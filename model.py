import dataclasses
import math
from collections import namedtuple
from dataclasses import field
from functools import partial
from typing import Any, Callable
import jax
import jax.numpy as jnp
import orbax.checkpoint as jnp
from flax import struct
from jax.experimental import mesh_utils 
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.experimental.shard_map import shard_map 
from jax.sharding import PartitionSpec as P

def create_mesh():
    """
    Always 1D because only care about FSDP.

    Purpose: Creates a one-dimensional device mesh for distributed training using Fully Sharded Data Parallelism (FSDP)
    This mesh defines how model parameters and computation will be distributed across available hardware accelerators

    Inputs: None

    Outputs: Returns a jax.sharding.Mesh object representing a 1D arrangement of devices
    The mesh has a single axis named "x" that spans across all available devices
    """
    devices = jax.devices()
    mesh_shape = (len(devices),)
    # Create a 1D mesh with all devices along the 'x' axis
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape, devices), ("x",))
    return mesh

ShardingRules = namedtuple(
    "FSDPRules",
    [ 
        "batch",
        "sequence",
        "d_model",
        "query_heads",
        "key_heads",
        "key_dim",
        "ffw",
        "vocab",
        "conv_window",
    ],
)

# Define sharding rules for Fully Sharded Data Parallelism (FSDP)
fsdp_rules = ShardingRules(
    batch="x",  # Shard batch dimension
    sequence=None,  # Don't shard sequence dimension
    d_model="x",  # Shard model dimension
    query_heads=None,
    key_heads=None,
    key_dim=None,
    ffw=None,
    vocab=None,
    conv_window=None,
)

# Define sharding rules for model parallelism
mdl_parallel_rules = ShardingRules(
    batch=None,
    sequence=None,
    d_model=None,
    query_heads="x",  # Shard query heads
    key_heads="x",  # Shard key heads
    key_dim=None,
    ffw="x",  # Shard feed-forward layer
    vocab=None,
    conv_window=None,
)

def _logical_to_physical(logical: P, rules: ShardingRules):
    """Converts logical to physical pspec."""
    return P(*(getattr(rules, axis) for axis in logical))


def _logical_to_sharding(logical: P, mesh: jax.sharding.Mesh, rules: ShardingRules):
    """Converts logical to sharding."""
    return jax.sharding.NamedSharding(mesh, _logical_to_physical(logical, rules))

@struct.dataclass
class Config:
    d_model: int
    ffw_multiplier: int
    query_heads: int
    key_heads: int
    num_layers: int
    key_dim: int
    vocab_size: int
    # Max seq len here can be a source of nasty bugs in incremental prefill
    # if we overflow (since dynamic slice will shunt left instead of erroring. Fix?
    max_seq_len: int
    causal: bool
    use_attn_kernel: bool
    weight_dtype_at_rest: jnp.float32
    active_weight_dtype: jnp.bfloat16
    # Sharding rules
    rules: ShardingRules
    mesh: jax.sharding.Mesh | None
    # Optimizer config
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 50
    total_steps: int = 10000
    # Rescale gradients which spike.
    grad_norm_clip: float = 0.1
    # MEGABYTE only https://openreview.net/pdf?id=JTmO2V9Xpz
    mega_byte: bool = False
    patch_size: int | None = 1
    # BERT
    mask_token: int | None = None

    @property
    def patch_d(self):
        return self.d_model // self.patch_size
    
@struct.dataclass
class TensorInfo:
    shape: jax.ShapeDtypeStruct
    logical_axes: tuple
    initializer: Callable | None = None
    metadata: dict = field(default_factory=dict)


def process_batch(batch, cfg, step_idx: int | None = None):
    del step_idx
    batch_size = batch["x"].shape[0]
    # Patch size lets us handle megabyte style methods to reduce effective seqlen.
    # E.g. mega|byte|tran|sfor predict |byte|tran|sfor|----
    dummy = np.zeros((batch_size, cfg.patch_size), dtype=jnp.int32)
    return {
        "x": np.concatenate([batch["x"][:, : -cfg.patch_size], dummy], axis=-1),
        "y": np.concatenate([batch["x"][:, cfg.patch_size :], dummy], axis=-1),
        # TODO(sholto): Maybe we should padd in dataset so we are't attending to next seq.
        "segment_ids": np.concatenate([batch["segment_ids"][:, : -cfg.patch_size], dummy], axis=-1),
        "aux": None,
    }

def process_batch_shae(batch, cfg, step_idx: int | None = None):
    batch_size = batch["x"].shape[0]
    aux = {
        "lad_category": batch["lad_category"],
        "lad_value": batch["lad_value"],
        "sad_category": batch["sad_category"],
        "sad_value": batch["sad_value"],
    }

    if cfg.causal:
        # Patch size lets us handle megabyte style methods to reduce effective seqlen.
        # E.g. mega|byte|tran|sfor predict |byte|tran|sfor|----
        dummy = np.zeros((batch_size, cfg.patch_size), dtype=jnp.int32)
        x = np.concatenate([batch["x"][:, : -cfg.patch_size], dummy], axis=-1)
        y = np.concatenate([batch["x"][:, cfg.patch_size :], dummy], axis=-1)
        segment_ids = np.concatenate([batch["segment_ids"][:, : -cfg.patch_size], dummy], axis=-1)
    else:
        segment_ids = batch["segment_ids"]
        # In this case we are doing BERT.
        bert_corruption = jax.random.randint(jax.random.key(step_idx), segment_ids.shape, 0, 100)
        # 15 of tokens masked
        masking = bert_corruption <= 15
        masked_batch = jnp.where(masking, cfg.mask_token, batch["x"])
        x = masked_batch
        y = batch["x"]
        aux["bert_mask"] = masking

    return {
        "x": x,
        "y": y,
        "segment_ids": segment_ids,
        "aux": aux,
    }

@struct.dataclass
class Layer:
    q: jax.Array | TensorInfo
    k: jax.Array | TensorInfo
    v: jax.Array | TensorInfo
    proj: jax.Array | TensorInfo
    w1: jax.Array | TensorInfo
    w2: jax.Array | TensorInfo
    # Extra layernorms like grok.
    attn_in_gamma: jax.Array | TensorInfo
    attn_out_gamma: jax.Array | TensorInfo
    ff_in_gamma: jax.Array | TensorInfo
    ff_out_gamma: jax.Array | TensorInfo

    @classmethod
    def abstract(cls, cfg: Config):
        return Layer(
            q=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.query_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
                ("d_model", "query_heads", "key_dim"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
            ),
            k=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
                ("d_model", "key_heads", "key_dim"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
            ),
            v=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), cfg.weight_dtype_at_rest),
                ("d_model", "key_heads", "key_dim"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2)),
            ),
            proj=TensorInfo(
                jax.ShapeDtypeStruct((cfg.query_heads, cfg.key_dim, cfg.d_model), cfg.weight_dtype_at_rest),
                ("query_heads", "key_dim", "d_model"),
                jax.nn.initializers.he_normal(in_axis=(0, 1), out_axis=2),
            ),
            w1=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model, cfg.d_model * cfg.ffw_multiplier), cfg.weight_dtype_at_rest),
                ("d_model", "ffw"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            w2=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model * cfg.ffw_multiplier, cfg.d_model), cfg.weight_dtype_at_rest),
                ("ffw", "d_model"),
                jax.nn.initializers.he_normal(in_axis=1, out_axis=0),
            ),
            attn_in_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            attn_out_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            ff_in_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            ff_out_gamma=TensorInfo(
                jax.ShapeDtypeStruct((cfg.d_model,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
        )

@struct.dataclass
class Weights:
    layers: list[Layer]
    embedding: jax.Array | TensorInfo
    vocab_proj: jax.Array | TensorInfo
    gamma_final: jax.Array | TensorInfo
    # MEGABYTE only.
    causal_convs: list[jax.Array | TensorInfo] | None
    mini_model: list[Layer]
    # For predicting LADs
    lad_hidden: jax.Array
    lad_predictor: jax.Array
    lad_regressor: jax.Array
    sad_hidden: jax.Array
    sad_predictor: jax.Array
    sad_regressor: jax.Array

    @classmethod
    def abstract(cls, cfg: Config):
        if cfg.mega_byte:
            embed_dim = cfg.patch_d
            windows = [3, 5, 7]
            causal_convs = [
                TensorInfo(
                    jax.ShapeDtypeStruct((window, cfg.patch_d, cfg.patch_d), cfg.weight_dtype_at_rest),
                    ("vocab", "d_model"),
                    jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
                )
                for window in windows
            ]
            mini_cfg = dataclasses.replace(cfg, d_model=cfg.patch_d)
            # 3 Little transformer layers on top!
            mini_model = [Layer.abstract(mini_cfg) for _ in range(3)]
        else:
            embed_dim = cfg.d_model
            causal_convs = None
            mini_model = None

        return Weights(
            layers=[Layer.abstract(cfg) for _ in range(cfg.num_layers)],
            embedding=TensorInfo(
                jax.ShapeDtypeStruct((cfg.vocab_size, embed_dim), cfg.weight_dtype_at_rest),
                ("vocab", "d_model"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            vocab_proj=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, cfg.vocab_size), cfg.weight_dtype_at_rest),
                ("d_model", "vocab"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            gamma_final=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim,), cfg.weight_dtype_at_rest),
                ("d_model",),
                jax.nn.initializers.constant(1.0),
            ),
            causal_convs=causal_convs,
            mini_model=mini_model,
            lad_hidden=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, embed_dim), cfg.weight_dtype_at_rest),
                ("d_model", "ffw"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            lad_predictor=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, 3), cfg.weight_dtype_at_rest),
                ("d_model", "vocab"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            lad_regressor=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, 1), cfg.weight_dtype_at_rest),
                ("d_model", "vocab"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            sad_hidden=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, embed_dim), cfg.weight_dtype_at_rest),
                ("d_model", "ffw"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            sad_predictor=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, 3), cfg.weight_dtype_at_rest),
                ("d_model", "vocab"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
            sad_regressor=TensorInfo(
                jax.ShapeDtypeStruct((embed_dim, 1), cfg.weight_dtype_at_rest),
                ("d_model", "vocab"),
                jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
            ),
        )

    @classmethod
    def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: dict):
        abstract = cls.abstract(cfg)
        return jax.tree.map(
            lambda info: _logical_to_sharding(info.logical_axes, mesh, rules),
            abstract,
            is_leaf=lambda x: isinstance(x, TensorInfo),
        )
    
    @classmethod
    def init(
        cls, cfg: Config, key: jax.random.PRNGKey, mesh: jax.sharding.Mesh, rules: dict, use_low_mem_init: bool = True
    ):
        def _init():
            abstract = cls.abstract(cfg)
            # Create one new RNG key per tensor.
            num_leaves = len(jax.tree_util.tree_leaves(abstract))
            key_iter = iter(jax.random.split(key, num_leaves))
            return jax.tree.map(
                lambda info: info.initializer(next(key_iter), info.shape.shape, info.shape.dtype),
                abstract,
                is_leaf=lambda x: isinstance(x, TensorInfo),
            )

        if use_low_mem_init:
            _init = jax.jit(_init, out_shardings=cls.shardings(cfg, mesh, rules))
        return jax.device_put(_init(), cls.shardings(cfg, mesh, rules))

@struct.dataclass
class KVCache:
    k: list[jax.Array]  # (batch_size, key_heads, max_seq_len, key_dim)
    v: list[jax.Array]  # (batch_size, key_heads, max_seq_len, key_dim)
    lengths: jax.Array  # [batch_size]

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int):
        return KVCache(
            k=[
                TensorInfo(
                    jax.ShapeDtypeStruct((batch_size, cfg.key_heads, max_seq_len, cfg.key_dim), jnp.bfloat16),
                    ("batch", "key_heads", "sequence", "key_dim"),
                )
                for _ in range(cfg.num_layers)
            ],
            v=[
                TensorInfo(
                    jax.ShapeDtypeStruct((batch_size, cfg.key_heads, max_seq_len, cfg.key_dim), jnp.bfloat16),
                    ("batch", "key_heads", "sequence", "key_dim"),
                )
                for _ in range(cfg.num_layers)
            ],
            lengths=TensorInfo(
                jax.ShapeDtypeStruct((batch_size,), jnp.int32),
                ("batch",),
            ),
        )

    @classmethod
    def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: ShardingRules):
        abstract = cls.abstract(
            cfg, batch_size=1, max_seq_len=cfg.max_seq_len
        )  # Batch size 1, since we just want the axes names.
        return jax.tree.map(
            lambda info: _logical_to_sharding(info.logical_axes, mesh, rules),
            abstract,
            is_leaf=lambda x: isinstance(x, TensorInfo),
        )

    @classmethod
    def init(cls, cfg: Config, batch_size: int, max_seq_len: int):
        abstract = cls.abstract(cfg, batch_size, max_seq_len)
        return jax.tree.map(
            lambda info: jnp.zeros(info.shape.shape, info.shape.dtype),
            abstract,
            is_leaf=lambda x: isinstance(x, TensorInfo),
        )

    @property
    def time_axis(self) -> int:
        return 2

def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")


def _generate_pos_embeddings(
    positions: jax.Array, features: int, min_timescale=1.0, max_timescale=16384.0
) -> tuple[jax.Array, jax.Array]:
    """Generate Sin/Cos for Rotary Embeddings.

    Generates sinusoids at (features//2) different timescales, where the
    timescales form a geometric series from min_timescale to max_timescale
    (max_timescale is not included, but would be the next element in the series).

    Sinusoids are evaluated at integer positions i in [0, length).

    The outputs are computed as:

    sin[b, t, j] = sin(rope_pos[b, t] / timescale[j])
    cos[b, t, j] = cos(rope_pos[b, t] / timescale[j])

    Args:
        postions: [batch, time]
        features: d_head.
        min_timescale: an optional float
        max_timescale: an optional float

    Returns:
        output_sin: a float32 Tensor with shape [length, features // 2]
        output_cos: a float32 Tensor with shape [length, features // 2]
    """
    # Forked from
    # flaxformer/components/embedding.py;l=592
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, since rounding off to a bfloat16 is
    # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
    # from sin(256).
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

def apply_rotary_embedding(x, sin, cos):
    """
    Purpose: Applies rotary position embeddings to input vectors using rotation matrices

    Inputs:
    - x: Input tensor [batch, heads, sequence_length, head_dim]
    - sin: Sine values [batch, sequence_length, head_dim/2]
    - cos: Cosine values [batch, sequence_length, head_dim/2]

    Output:
    - Rotated tensor with same shape as x
    """
    # Verify input dimensions
    assert x.ndim == 4 # Ensure x has 4 dimensions [batch, heads, seq_len, head_dim]
    assert sin.ndim == 3 and cos.ndim == 3 # Ensure sin/cos have 3 dimensions [batch, seq_len, head_dim/2]
    
    # split x into two halves along the last dimension
    # If head_dim = 128, x1 and x2 will each have head_dim=64
    x1, x2 = jnp.split(x, 2, axis=-1)

    # Add singleton dimension for heads
    # [B, T, head_dim] -> [B, 1, T, head_dim]
    sin, cos = sin[:, None, :, :, :]
    cos[:, None, :, :] # [B, T, head_dim] --> [B, h, T, head_dim]
    # Apply 2D rotation to each vector:
    # [x1] = [cos, -sin] [x1]
    # [x2]   [sin, cos] [x2]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal: bool):
    """
    Purpose: Creates an attention mask that handles both:
    1. Segment masking (prevents attention between different segments)
    2. Causal masking (prevents attention to future tokens)

    Inputs:
    - q_len: Query sequence length
    - k_len: Key sequence length
    - q_segment_ids: Segment IDs for queries [batch, q_len]
    - k_segment_ids: Segment IDs for keys [batch, k_len]
    - q_offset: Offset for query positions [batch]
    - causal: Whether to apply causal masking

    Output:
    - Attention mask of shape [batch, 1, q_len, k_len]
    - True where attention is allowed, False where it should be masked
    """

    # Create segment mask by comparing segment IDs
    # [:, :, None] and [:, None, :] enable broadcasting for comparison
    # [B, t, T] where True means tokens are in same segment
    segment_mask = q_segment_ids[:, :, None] == k_segment_ids[:, None, :]

    # Add singleton dimension for attention heads
    # [B, t, T] -> [B, 1, t, T]
    segment_mask = segment_mask[:, None, :]

    if causal:
        # Create causal mask for autoregressive generation

        # Define shape for broadcasting [batch, heads, query_len, key_len]
        qk = (1, 1, q_len, k_len)

        # Create position indices for queries and keys
        q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2) # Counts along query dimension
        k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3) # Counts along key dimension

        # Add offset to query positions (used in incremental decoding)
        q_positions = q_iota + q_offset[:, None, None, None]

        # Create causal mask: True where query position >= key position
        # This ensures each position can only attend to previous positions
        causal_mask = q_positions >= k_iota 

        # Combine segment and causal masks with logical AND
        # Both masks must allow attention for final mask to allow it
        combined_mask = jnp.logical_and(segment_mask, causal_mask)
        return combined_mask
    else:
        # For non-causal (e.g. BERT), just use segment mask
        return segment_mask

def attention(
        q: jax.Array, # Query vectors [batch, heads, q_len, head_dim]
        k: jax.Array, # Key vectors [batch, heads, k_len, head_dim]
        v: jax.Array, # Value vectors [batch, heads, k_len, head_dim]
        q_segment_ids: jax.Array, # Query segment IDs [batch, q_len]
        k_segment_ids: jax.Array, # Key segment IDs [batch, k_len]
        q_offset: jax.Array, # Query position offset [batch]
        cfg: Config, # Configuration object
        internals: Any, # Dictionary for storing internal values
        layer_idx: int, # Current layer index
) -> jax.Array:
    """
    Purpose: Computes scaled dot-product attention with masking

    Input shapes explained:
    - batch: number of sequences
    - heads: number of attention heads
    - q_len/k_len: sequence lengths
    - head_dim: dimension per head

    Output:
    - Attention output [batch, heads, q_len, head_len]
    """

    # Calculate scaling factor for numerical stability
    # Prevents dot products from growing too large
    scale = q.shape[-1] ** -0.5 # 1/sqrt(head_dim)

    # Verify input precision
    assert q.dtype == jnp.float32
    assert k.dtype == jnp.float32

    # Compute attention scores (dot product of queries and keys)
    # Einstein notation: batch, heads, query time, dim x batch, heads, key_time, dim
    # -> batch, heads, query time, key time
    qk = jnp.einsum("bhtd, bhTd->bhtT", q, k) * scale

    # Create attention mask combining segment and causal masks if needed
    mask = make_attention_mask(
        q.shape[2], k.shape[2],
        q_segment_ids, k_segment_ids,
        q_offset, cfg.causal
    )

    # Apply mask by replacing masked positions with large negative number
    # This ensures they become ~0 after softmax
    qk = jnp.where(mask, qk, -1e30)

    # Apply softmax to get attention weights
    # Note: JAX softmax handles numerical stability internally
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)

    # Store attention scores for visualization/analysis
    internals['layers'][layer_idx]['attn_scores'] = attn 

    # Compute weighted sum of values based on attention weights
    # Einstein notation: batch, heads, query time, key time x batch, heads, key time, dim
    # -> batch, heads, query time, dim
    return jnp.einsum("bhtT,bhTd->bhtd", attn, v).astype(jnp.bfloat16)

def attention_kernal(q, k, v, q_segment_ids, kv_segment_ids, cfg: Config):
    """
    Purpose: Implements Flash Attention, a memory-efficient attention algorithm that processes attention in blocks to reduce memory usage

    Inputs:
    - q: Query vectors [batch, heads, q_len, head_dim]
    - k: Key vectors [batch, heads, k_len, head_dim]
    - v: Value vectors [batch, heads, k_len, head_dim]
    - q_segment_ids: Query segment IDs [batch, q_len]
    - kv_segment_ids: Key/value segment IDs [batch, k_len]
    - cfg: Configuration object

    Output:
    - Attention output [batch, heads, q_len, head_dim] in bfloat16
    """

    # Convert inputs to float32 (required for TPU)
    q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)

    # Standard attention scaling factor
    scale = q.shape[-1] ** -0.5

    # Define sharded computation function
    @partial(
       shard_map,
       mesh=cfg.mesh,
       # Specify how inputs should be sharded across devices
       in_specs=(
           _logical_to_physical(P("batch", "query_heads", "sequence", "key_dim"), cfg.rules),  # q
           _logical_to_physical(P("batch", "key_heads", "sequence", "key_dim"), cfg.rules),    # k
           _logical_to_physical(P("batch", "key_heads", "sequence", "key_dim"), cfg.rules),    # v
           _logical_to_physical(P("batch", "sequence"), cfg.rules),                            # q_segment_ids
           _logical_to_physical(P("batch", "sequence"), cfg.rules),                            # kv_segment_ids
       ),
       # Specify how output should be sharded
       out_specs=_logical_to_physical(P("batch", "query_heads", "sequence", "key_dim"), cfg.rules),
       check_rep=False,  # Disable representation checks for performance
    )

    def _f(q, k, v, q_segment_ids, kv_segment_ids):
        # Create segment IDs object for Flash Attention
        segment_ids = flash_attention.SegmentIDs(q_segment_ids, kv_segment_ids)

        # Call Flash Attention implementation
        return flash_attention.flash_attention(
            q, k, v,
            segment_ids=segment_ids,
            causal=True, # Use causal masking
            sm_scale=scale, # Attention scaling factor
            # Define block sizes for tiled computation
            block_sizes=flash_attention.BlockSizes(
               # All block sizes set to 512 for this implementation
               block_q=512,            # Query block size
               block_k_major=512,      # Major key block size
               block_k=512,            # Key block size
               block_b=1,              # Batch block size
               block_q_major_dkv=512,  # Query block size for dkv computation
               block_k_major_dkv=512,  # Key block size for dkv computation
               block_k_dkv=512,        # Key block size for dkv computation
               block_q_dkv=512,        # Query block size for dkv computation
               block_k_major_dq=512,   # Key block size for dq computation
               block_k_dq=512,         # Key block size for dq computation
               block_q_dq=512,         # Query block size for dq computation
           ),
        )
    
    # Execute sharded computation and convert result to bloat16
    return _f(q, k, v, q_segment_ids, kv_segment_ids).astype(jnp.bfloat16)

def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    """
    Purpose: Applies Root Mean Square (RMS) normalization to input
    Similar to layer normalization but simpler and more efficient
    Used in modern transformers like PaLM andn Grok

    Inputs:
    - x: Input tensor to normalize [batch, seq_len, d_model]
    - gamma: Learned scale parameter [d_model]

    Output:
    - Normalized tensor with same shape as x in bfloat16
    """
    # 1. Cast to float32 for better numerical precision
    # 2. Square each element
    # 3. Take mean across last dimension (keeping dims for broadcasting)
    # 4. Add small epsilon (1e-6) for numerical stability
    # 5. Take square root to get RMS
    rms = jnp.sqrt(
        jnp.mean(
            jnp.astype(x, jnp.float32)**2,
            axis=-1 # Normalize across feature dimension
            keepdims=True # Keep dims for broadcasting
        ) + 1e-6
    )

    # 1. Divide input by RMS to normalize
    # 2. Multiply by learned scale (gamma)
    # 3. Cast back to bloat16 for efficiency
    return jnp.astype(gamma * x / rms, jnp.bfloat16)

def forward_layer(
        x: jax.Array,  # Input tensor [batch, seq_len, d_model]
        segment_ids: jax.Array, # Segment IDs [batch, seq_len]
        layer: Layer, # Layer parameters
        sin: jax.Array, # Sine positional embeddings
        cos: jax.Array, # Cosine positional embeddings
        idx: int, # Layer index
        cfg: Config, # Model config
        cache: KVCache | None = None, # Optional KV cache for inference
        internals: Any = None, # For storing intermediate values
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Modified forward_layer function to optionally store residual stream activation"""
    
    """
    Purpose: Processes one transformer layer, including attention and feed-forward

    Output: Tuple of (processed_tensor [batch, seq_len, d_model],
    key_cache [batch, heads, seq_len, dim],
    value_cache [batch, heads, seq_len, dim]
    )
    """
    # Store residual stream activations before any layer processing if requested
    if store_activations:
        internals['residual_stream'] = x.copy()

    # Initialize internals for this layer
    internals['layers'].append({})

    # Convert weights to bfloat16 for efficiency
    layer = dataclasses.replace(layer,
                                q=cfg.active_weight_dtype(layer.q),
                                k=cfg.active_weight_dtype(layer.k),
                                v=cfg.active_weight_dtype(layer.v),
                                w1=cfg.active_weight_dtype(layer.w1),
                                w2=cfg.active_weight_dtype(layer.w2))
    
    # Pre-attention normalization
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_in_gamma)
    
    # Compute Q, K, V matrices
    with jax.named_scope("qkv_matmul"):
        # btd, dhq->bhtq: batch, time, d_model x d_model, heads, query_dim
        q = jnp.einsum("btd, dhq->bhtq", attn_in, layer.q)
        k = jnp.einsum("btd, dhk->bhtk", attn_in, layer.k)
        v = jnp.einsum("btd, dhv->bhtv", attn_in, layer.v)
    
    # Apply rotary position embeddings
    with jax.named_scope("rope"):
        q = apply_rotary_embedding(q, sin, cos)
        k = apply_rotary_embedding(k, sin, cos)
    
    # Handle KV caching for inference
    with jax.named_scope("cache_update"):
        if cache is not None:
            # Get cached keys and values
            cache_k, cache_v = cache.k[idx], cache.v[idx]

            def update(original, update, at):
                # Update cache at specific positions
                return jax.lax.dynamic_update_slice_in_dim(
                    original, update, at, axis=cache.time_axis - 1
                )
            
            # Update cache with new k,v values
            k, v = jax.vmap(update, in_axes=(0, 0, 0))(
                cache_k, k.astype(cache_k.dtype), cache.lengths
            ), jax.vmap(update, in_axes=(0, 0, 0))(
                cache_v, v.astype(cache_v.dtype), cache.lengths
            )

            # Create masks for valid positions
            q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
            time_indices = jnp.arange(0, v.shape[cache.time_axis])[None, :]
            incremental_positions = jnp.sum(segment_ids != 0, axis=-1)
            k_segment_ids = jnp.where(
                time_indices < (cache.lengths + incremental_positions)[:, None], 
                1, 0
            )
            
            # Apply masks to k,v
            k = k * k_segment_ids[:, None, :, None]
            v = v * k_segment_ids[:, None, :, None]
            q_offset = cache.lengths
        else:
            q_segment_ids = segment_ids
            k_segment_ids = segment_ids
            q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)

    # Compute attention
    with jax.named_scope("attention"):
        if cfg.use_attn_kernel and cache is None:
            attn_out = attention_kernal(q, k, v, q_segment_ids, k_segment_ids, cfg)
        else:
            attn_out = attention(q, k, v, q_segment_ids, k_segment_ids, q_offset, cfg, internals, idx)

    # Project attention output
    with jax.named_scope("projection"):
        attn_out = jnp.einsum("bhtq, hqd->btd", attn_out, layer.proj)
    
    # First residual connection + normalization
    with jax.named_scope("residual"):
        attn_out = rms_norm(attn_out, layer.attn_out_gamma)
        x = x + attn_out
    
    # Pre-FFN normalization
    with jax.named_scope("ffn_pre_norm"):
        ff_in = rms_norm(x, layer.ff_in_gamma)
    
    # Feed-forward network
    with jax.named_scope("ffw"):
        ff_out = jnp.einsum("btd,df->btf", ff_in, layer.w1) # First linear
        ff_out = jax.nn.gelu(ff_out) # GELU activation
        ff_out = jnp.einsum("btf,fd->btd", ff_out, layer.w2) # Second linear

    # Second residual connection + normalization
    with jax.named_scope("residual"):
        ff_out = rms_norm(ff_out, layer.ff_out_gamma)
        x = x + ff_out
    
    return x, k, v

def causal_conv1d(x, weight):
    """
    Purpose: Part of MEGABYTE architecture for processing patched sequences
    Only used when cfg.mega_byte=True 

    Inputs:
    - x: Input tensor [batch_size, time, patch, in_channels]
    - weight: Convolutional kernel [kernel_size, in_channels, out_channels]

    Output: 
    - Convolved tensor [batch_size, time, patch, out_channels]
    """
    
    kernel_size, _, out_channels = weight.shape

    # Add causal padding (only pad beginning, not end)
    # This ensures convolution only sees past token
    padded_x = jnp.pad(x, ((0, 0), # don't pad batch
                            (kernel_size - 1, 0), # pad time dimension
                            (0, 0))) # don't pad channels
    
    # Perform convolution using JAX's low-level conv op
    out = jax.lax.conv_general_dilated(
        lhs=padded_x, # input
        rhs=weight, # kernel
        window_strides=(1,), # stride size
        padding="VALID", # no additional padding
        # Specify dimension layout:
        # NHC = batch, height(time), channels
        # HIO = height(kernel), input_channels, output_channels
        dimension_numbers=("NHC", "HIO", "NHC")
    )

    return out

def forward(
        x: jax.Array, # Input tokens [batch, seq_len]
        segment_ids: jax.Array, # Segment IDs [batch, seq_len]
        weights: Weights, # Model weights
        cfg: Config, # Configuration
        cache: KVCache | None = None, # Option KV cache for inference
        store_activations: bool = False,
        store_layer_idx: int = None,
        *, # Forces aux to be keyword-only
        aux: Any | None = None, # Optional auxiliary data for LAD/SPAD prediction
):
    """
    Purpose: Main forward pass of the transformer model
    Handles both standard and MEGABYTE processing paths
    """
    """Modified forward function with activation storage"""

    # Initialize storage for intermediate values
    internals = {'layers': [], 'stored_activations': None}

    # Convert token IDs to embeddings
    embeds = weights.embedding[x, :] # [B, T] -> [B, T, D]
    x = embeds
    batch, time = x.shape[0], x.shape[1]

    # MEGABYTE processing path
    if cfg.mega_byte:
        # Verify embedding dimension matches patch dimension
        assert x.shape[-1] == cfg.patch_d 

        # Reshape for patch processing
        # [B, T, patch_d] -> [B, T//patch-size, patch_size, patch_d]
        x = x.reshape(batch, time // cfg.patch_size, cfg.patch_size, cfg.patch_d)

        # Flatten for convolution
        x = x.reshape(-1, cfg.patch_size, cfg.patch_d)

        # Apply causal convolutions
        for filter in weights.causal_convs:
            x = causal_conv1d(x, filter)

        # Reshape back
        x = x.reshape(batch, time // cfg.patch_size, cfg.patch_size, cfg.patch_d)
        x = x.reshape(batch, time // cfg.patch_size, cfg.d_model)

        # Process segment IDs for patches
        main_block_segment_ids = segment_ids[:, :: cfg.patch_size]
        positions = segment_ids_to_positions(main_block_segment_ids)
        assert x.shape[1] == main_block_segment_ids.shape[1]
    
    # Standard processing path
    else:
        positions = segment_ids_to_positions(segment_ids)
        main_block_segment_ids = segment_ids

    # Handle positional embeddings
    if cache is not None:
        start_indices = cache.lengths # Use cache position for inference
    else:
        start_indices = jnp.zeroes((batch,), dtype=jnp.int32)
    
    positions = start_indices[:, None] + positions
    sin, cos = _generate_pos_embeddings(positions, cfg.key_dim,
                                        min_timescale=1.0,
                                        max_timescale=cfg.max_seq_len)
    
    # Process through transformer layers
    for idx, layer in enumerate(weights.layers):

        # Store activations at the specified middle layer
        should_store = store_activations and idx == store_layer_idx
        
        x, k, v = forward_layer(
            x, main_block_segment_ids, layer, sin, cos, idx, cfg, 
            cache, internals, store_activations=should_store)
        
        # If this was the layer we wanted to store, save the activations
        if should_store:
            internals['store_activations'] = {
                'residual_stream': internals['residual_stream'],
                'sequences': x,
                'metadata': {
                    'batch_idx': jnp.arange(batch),
                    'sequence_positions': jnp.arange(time),
                    'layer_idx': idx,
                }
            }

        if cache is not None:
            cache.k[idx] = k
            cache.v[idx] = v
    
    # Additional MEGABYTE processing
    if cfg.mega_byte:
        # Reshape for per-patch processing
        x = x.reshape(batch, time // cfg.patch_size, cfg.patch_size, cfg.patch_d)
        
        # Add previous token embeddings
        prev_token_embeds = embeds[:, :-1, :]
        prev_token_embeds = jnp.concatenate([jnp.zeros_like(embeds[:, 0:1, :]), 
                                           prev_token_embeds], axis=1)
        prev_token_embeds = prev_token_embeds.reshape(batch, time // cfg.patch_size, 
                                                     cfg.patch_size, cfg.patch_d)
        x = x + prev_token_embeds
        
        # Process each patch with mini-transformer
        x = x.reshape(-1, cfg.patch_size, cfg.patch_d)
        per_patch_segment_ids = segment_ids.reshape(batch, time // cfg.patch_size, 
                                                  cfg.patch_size)
        per_patch_segment_ids = per_patch_segment_ids.reshape(-1, cfg.patch_size)
        
        # Generate positional embeddings for patches
        positions = segment_ids_to_positions(segment_ids)
        per_patch_positions = positions.reshape(batch, time // cfg.patch_size, 
                                             cfg.patch_size)
        per_patch_positions = positions.reshape(-1, cfg.patch_size)
        sin, cos = _generate_pos_embeddings(per_patch_positions, cfg.key_dim, 
                                          min_timescale=1.0, 
                                          max_timescale=cfg.max_seq_len)
        
        # Process through mini-transformer
        mini_model_cfg = dataclasses.replace(cfg, use_attn_kernel=False)
        for idx, layer in enumerate(weights.mini_model):
            x, _, _ = forward_layer(x, per_patch_segment_ids, layer, sin, cos, 
                                  idx, mini_model_cfg, None)
        
        # Reshape back to sequence
        x = x.reshape(batch, time // cfg.patch_size, cfg.patch_size, cfg.patch_d)
        x = x.reshape(batch, time, cfg.patch_d)

    # Final layer norm and projection
    x = rms_norm(x, weights.gamma_final)
    logits = jnp.einsum("btd,dv->btv", x, weights.vocab_proj)

    # LAD/SPAD predictions (if aux data provided)
    if aux is not None:
        # Find last non-padding position
        last_nonzero = jnp.sum(segment_ids > 0, axis=-1)
        indices = last_nonzero[:, None, None] - 1
        last_xs = jnp.take_along_axis(x, indices, 1)
        internals['last_embed'] = last_xs
        assert last_xs.shape[1] == 1

        # LAD predictions
        lad = jax.nn.gelu(jnp.einsum("btd,df->btf", last_xs, weights.lad_hidden))
        internals["lad_pred"] = jnp.einsum("btf,fv", lad, weights.lad_predictor)[:, 0, :]
        internals["lad_reg"] = jnp.einsum("btf,fv", lad, weights.lad_regressor)[:, 0, 0]
        
        # SAD predictions
        sad = jax.nn.gelu(jnp.einsum("btd,df->btf", last_xs, weights.sad_hidden))
        internals["sad_pred"] = jnp.einsum("btf,fv", sad, weights.sad_predictor)[:, 0, :]
        internals["sad_reg"] = jnp.einsum("btf,fv", sad, weights.sad_regressor)[:, 0, 0]
    
    # Update cache for inference if provided
    if cache is not None:
        cache = dataclasses.replace(cache, lengths=cache.lengths + jnp.sum(segment_ids != 0, axis=-1))
        return logits, cache, internals, x
    
    return logits, internals, x

def save_activations(activation_dir: str, step: int, internals: dict):
    """Save activations and metadata to disk"""
    if internals.get('stored_activations') is None:
        return 
    
    activation_data = internals['stored_activations']

    # Create paths for this training step
    step_dir = os.path.join(activation_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    # Save activations
    jnp.save(
        os.path.join(step_dir, "residual_stream.npy"),
        activation_data['residual_stream']
    )

    # Save sequences
    jnp.save(
        os.path.join(step_dir, "sequences.npy"),
        activation_data['sequences']
    )

    # Save metadata
    with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump(activation_data['metadata'], f)

def get_lr_with_cosine_decay_and_warmup(
    step: int, # Current training step
    total_steps: int, # Total number of training steps
    max_lr: float, # Maximum learning rate
    min_lr: float, # Minimum learning rate
    warmup_steps: int # Number of warmup steps
):
    """
    Purpose: Calculates learning rate using cosine decay schedule with linear warmup
    This helps stabilize early training and then gradually reduce learning rate

    Returns: Learning rate for current step
    """

    def warmup(s):
        """Linear warmup from 0 to max_lr"""
        # s/warmup_steps goes from 0 to 1 during warmup period
        return max_lr * (s / warmup_steps)
    
    def cosine_decay(s):
        """Cosine decay from max_lr to min_lr"""
        # Calculate progress through decay period (0 to 1)
        progress = (s - warmup_steps) / (total_steps - warmup_steps)

        # Cosine decay formula:
        # min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
        # - When progress = 0: cos(0) = 1, so lr = max_lr
        # - When progress = 1: cos(π) = -1, so lr = min_lr
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress))
    
    return jax.lax.cond(step < warmup_steps, warmup, cosine_decay, step)

def adam_update(
        param: jax.Array, # Current parameter values
        grad: jax.Array, # Current gradients
        m: jax.Array, # First moment (momentum)
        v: jax.Array, # Second moment (variance)
        lr: float, # Learning rate
        t: int, # Current timestep
        beta1=0.9, # Momentum decay rate
        beta2=0.999, # Variance decay rate
        eps=1e-8 # Small constant for numerical stability
):
    """
    Purpose: Implements one step of the Adam optimization algorithm
    Adam combines momentum and adaptive learning rates per parameter

    Returns: Tuple of (updated_parameters, updated_momentum, updated_variance)
    """

    # Update momentum (first moment)
    # Exponential moving average of gradients
    # New momentum = beta1 * old_momentum + (1-beta1) * current_gradient
    m = beta1 * m + (1 - beta1) * grad

    # Update variance (second moment)
    # Expoential moving average of squared gradients 
    # New variance = beta2 * old_variance + (1-beta2) * gradient
    v = beta2 * v + (1 - beta2) * jnp.square(grad)

    # Bias correction
    # Early in training, moments are biased toward zero
    # This correction helps overcome the bias
    m_hat = m / (1 - beta1 ** (t + 1)) # Corrected momentum
    v_hat = v / (1 - beta2 ** (t + 1)) # Corrected variance

    # Compute parameter update
    # - Larger variance -> smaller update (more cautious)
    # - Smaller variance -> larger update (more aggressive)
    # - Momentum helps push through plateaus
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)

    # Return updated values
    return param - update, m, v

def init_optimizer_state(weights: Weights):
    """
    Purpose: Initializes Adam optimizer state (momentum and variance) for all model weights
    Creates two copies of zeros matching each weight tensor's shape - one for momentum, one for variance

    Inputs:
    - weights: Weights class containing all model parameters

    Output:
    - Tree of tuples matching weights structure, each containing (momentum, variance) initialized to zeros
    """ 

    def _zeros_like(old):
        """Helper function to create zeros matching input tensor structure"""
        if ininstance(old, jax.ShapeDtypeStruct):
            # For abstract tensors (shapes only), create matching structure
            return jax.ShapeDtypeStruct(
                old.shape, # Same shape as parameter
                old.dtype, # Same dtype
                sharding=old.sharding # Same sharding across devices
            )
        else:
            # For actual tensors (shapes only), create matching structure
            return jax.device_put(jnp.zeros_like(old), old.sharding)
    # For each parameter in weights tree:
    # - Create tuple of (momentum_zeros, variance_zeros)
    # - Maintain same tree structure as weights
    return jax.tree_map(
        lambda p: (_zeros_like(p), _zeros_like(p)),
        weights
    )   

def cross_entropy_loss(
        logits: jax.Array, # Raw model outputs [batch, seq_len, num_classes]
        labels: jax.Array, # True labels [batch, seq_len]
        mask: jax.Array, # Mask for valid positions [batch, seq_len]
        internals: jax.Array | None = None # Optional storage for intermediate values
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, Any]:
    """
    Purpose: Computes cross entropy loss and accuracy for model predictions
    Handles masked inputs (e.g., padding tokens)

    Returns: Tuple of (
        loss: mean cross entropy over valid tokens,
        accuracy: proportion of correct predictions in valid tokens,
        internals: optional dict with intermediate values
    )
    """

    # Get number of classes from logits shape
    num_classes = logits.shape[-1]

    # Convert integer labels to one-hot vectors
    labels_one_hot = jax.nn.one_hot(labels, num_classes)

    # Compute log probabilities using softmax
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Compute per-token loss
    loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)

    # Apply mask to zero out loss for padding tokens
    loss *= mask

    # Store per-token loss if internals provided
    if internals is not None:
        internals["per_token_loss"] = loss 
    
    # Count number of valid (non_padding) tokens
    valid_tokens = jnp.sum(mask)

    # Compute mean loss over valid tokens
    loss = loss.sum() / valid_tokens

    # Get predicted classes (highest logit)
    predictions = jnp.argmax(logits, axis=-1)

    # Count correct predictions (only for valid tokens)
    correct_predictions = jnp.sum((predictions == labels) * mask)

    # Compute accuracy over valid tokens
    accuracy = correct_predictions / valid_tokens

    # Return results (with or without internals)
    return (loss, accuracy) if internals is None else (loss, accuracy, internals)
def compute_loss(
        weights: Weights, # Model Parameters
        x: jax.Array, # Input tokens [batch, seq_len]
        segment_ids: jax.Array, # Segment IDs [batch, seq_len]
        y: jax.Array, # Target tokens [batch, seq_len]
        cfg: Config, # Model configuration
        aux: Any | None = None, # Optional auxiliary data
) -> tuple[jax.Array, Any]: # Returns (loss, internals)
    """
    Purpose: Main loss computation function that:
    1. Runs forward pass to get predictions
    2. Creates appropriate loss mask (different for causal vs BERT)
    3. Computes cross entropy loss and accuracy

    Returns: Tuple of (loss value, internal states/metrics)
    """

    # Run forward pass
    # * ignores additional returns (like cache)
    logits, internals, _ = forward(x, segment_ids, weights, cfg, aux=aux)

    # Create mask for loss computation
    # segment_ids of 0 indicates padding tokens
    loss_mask = jnp.where(segment_ids == 0, 0, 1) 

    # Handle BERT-style masked language modeling
    if not cfg.causal:
        # Verify we have masking information 
        assert "bert_mask" in aux

        # For BERT, we only want to compute loss on masked tokens
        # &= is bitwise and assignment
        # This combines padding mask with BERT mask
        loss_mask &= aux["bert_mask"]
    
    # Compute cross entropy loss with mask
    loss, accuracy, internals = cross_entropy_loss(
        logits, # Model predictions
        y, # True targets
        loss_mask, # Combined mask 
        internals # Store intermediate values
    )

    # Store loss and accuracy in internals

    internals["token_prediction_loss"] = loss
    internals["accuracy"] = accuracy

    return loss, internals

def update_weights(
        weights: Weights, # Current model parameters
        grads: Weights, # Computed gradients
        state: Any, # Optimizer state (momentum and variance)
        lr: float, # Current learning rate
        t: int, # Current timestep
        cfg: Config, # Model configuration
        internals: Any # Storage for metrics/debugging
):
    """
    Purpose: Updates model weights using gradients with:
    1. Gradient clipping for stability
    2. Adam optimization
    3. Per-parameter gradient norm computation

    Returns: Tuple of (new_weights, new_optimizer_state, internals
    )
    """

    def update_fn(param, grad, state, grad_norm):
        """Helper function to update a single parameter"""
        # Unpack optimizer state
        m, v = state # momentum and variance

        # Clip gradients to prevent explosive updates
        # Instead of global norm, clip per parameter
        # This allows overlap of weight sync during backward pass
        scale_factor = jnp.maximum(grad_norm, cfg.grad_norm_clip)
        scaled_grad = grad / scale_factor.astype(grad.dtype) * cfg.grad_norm_clip 

        # Apply Adam update
        param_update, m_new, v_new = adam_update(param, scaled_grad, m, v, lr, t)

        return param_update, (m_new, v_new)

    # Compute gradient norms for each parameter
    grad_norms = jax.tree.map(jnp.linalg.norm, grads)

    # Store gradient norms for monitoring
    internals["grad_norms"] = grad_norms

    # Apply updates to all parameters in parallel
    # Maps update_fn over the tree of weights/grads/states
    updated = jax.tree_map(update_fn, weights, grads, state, grad_norms)

    # Extract new weights and states from updated values
    # Use weights as structure template
    new_weights = jax.tree.map(lambda _, u: u[0], weights, updated) # First elements
    new_state = jax.tree.map(lambda _, u: u[1], weights, updated) # Second elements

    return new_weights, new_state, internals

def update_step(
        weights: Weights, # Current model parameters
        x: jax.Array, # Input tokens
        segment_ids: jax.Array, # Segment IDs
        y: jax.Array, # Target tokens
        opt_state: Any, # Optimizer state
        step: int, # Current training step
        cfg: Config, # Model configuration
        aux: Any | None, # Optional auxiliary data
        override_compute_loss_fn = None # Optional custom loss function
):
    """
    Purpose: Performs one complete training step:
    1. Computes loss and gradients
    2. Calculates learning rate
    3. Updates weights using Adam

    Returns: Tuple of (
    loss_value, updated_weights, updated_optimizer_state, internal_metrics
    )
    """

    # Select loss function
    if override_compute_loss_fn:
        compute_loss_fn = override_compute_loss_fn 
    else:
        compute_loss_fn = compute_loss 
    
    # Compute loss and gradients in one pass
    # value_and_grad returns both function value and gradients
    # has_aux=True because compute_loss returns (loss, internals)
    (loss, internals), grads = jax.value_and_grad(
        compute_loss_fn,
        has_aux=True
    )(weights, x, segment_ids, y, cfg, aux)

    # Calculate current learning rate using schedule
    lr = get_lr_with_cosine_decay_and_warmup(
        step=step,
        total_steps=cfg.total_steps,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps
    )

    # Update weights using gradients
    weights, opt_state, internals = update_weights(
        weights, grads, opt_state, lr, step, cfg, internals
    )

    # Store learning rate for monitoring
    internals["lr"] = lr

    return loss, weights, opt_state, internals

def input_shardings(
        mesh, # Device mesh for parallel computation
        rules # Sharding rules for distributing computation
) -> tuple[jax.sharding.NamedSharding, jax.sharding.NamedSharding, jax.sharding.NamedSharding]:
    """
    Purpose: Defines how input tensors should be distributed across multiple devices
    Specifies sharding for input tokens, segment IDs, and target tokens

    Returns: Dictionary of sharding specifications for each input tensor
    """

    # Define logical axes for each input tensor
    logical_axes = {
        "x": P("batch", "sequence"), # Input tokens
        # Shape example: [batch_size, sequence_length]
        # Could be sharded across batch dimension

        "segment_ids": P("batch", "sequence"), # Segment IDs
        # Same shape and sharding as input tokens
        # Keeps aligned with input tokens

        "y": P("batch", "sequence"), # Target tokens
        # Same shape and sharding as input tokens
        # Keeps aligned with input tokens
    }

    # Convert logical axes to physical device assignments
    physical_axes = jax.tree.map(
        partial(_logical_to_sharding, mesh=mesh, rules=rules),
        logical_axes
    )

    # Auxiliary data doesn't need sharding
    physical_axes["aux"] = None

    return physical_axes

# Checkpointing functions for saving/loading model state
def make_mngr(path="/tmp/checkpoint_manager_sharded", erase: bool = False):
    """
    Purpose: Creates a checkpoint manager for saving model state

    Args: 
    - path: Where to save checkpoints
    - erase: Whether to clear existing checkpoints

    Returns: Checkpoint manager that handles save/load
    """

    # Optionally erase existing checkpoints
    if erase:
        path = ocp.test_utils.erase_and_create_empty(path)
    
    # Create manager options
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3 # Only keep 3 most recent checkpoints
    )

    # Create and return manager
    mngr = ocp.CheckpointManager(path, options=options)
    return mngr

def save(mngr: ocp.CheckpointManager,
         weights: Weights,
         opt_state: Any,
         step: int):
    """
    Purpose: Saves model weights and optimizer state

    Args:
    - mngr: Checkpoint manager
    - weights: Model parameters
    - opt_state: Optimizer state (momentum/variance)
    """

    # Save weights and optimizer state
    mngr.save(
        step, # Checkpoint identifier
        args=ocp.args.StandardSave({
            "weights": weights,
            "opt_state": opt_state
        })
    )
    # Wait for save to complete
    mngr.wait_until_finished()

def load(mngr: ocp.CheckpointManager,
         cfg: Config,
         step: int | None = None):
    """
    Purpose: Loads model weights and optimizer state

    Args:
    - mngr: Checkpoint manager
    - cfg: Model configuration
    - step: Specific step to load (None for latest)

    Returns: Tuple of (weights, optimizer_state)
    """
    # Create abstract weight specifications
    abstract_weights = Weights.abstract(cfg)

    # Create concrete shapes and sharding specs for weights
    weights_shapes_shardings = jax.tree.map(
        lambda info: jax.ShapeDtypeStruct(
            info.shape.shape, # Tensor shape
            info.shape.dtype, # Data type
            sharding=jax.sharding.NamedSharding(
                cfg.mesh,
                _logical_to_physical(info.logical_axes, cfg.rules)
            ),
        ),
        abstract_weights,
        is_leaf=lambda x: isinstance(x, TensorInfo),
    )

    # Create matching optimizer state specifications
    opt_shapes_shardings = init_optimizer_state(weights_shapes_shardings)

    # Load checkpoint
    restored = mngr.restore(
        mngr.latest_step() if step is None else step,
        args=ocp.args.StandardRestore({
            "weights": weights_shapes_shardings,
            "opt_state": opt_shapes_shardings
        }),
    )

    return restored["weights"], restored["opt_state"]

# Inference functions for token generation
def prepare_chunk(chunk, pad_to: int, pad_id: int):
    """
    Purpose: Prepares input chunk for inference by padding and adding batch dimension

    Args:
    - chunk: Input tokens
    - pad_to: Target length to pad to
    - pad_id: Token ID to use for padding

    Returns: Tuple of (padded_chunk, segment_ids)
    """

    # Add padding and batch dimension
    # [length] -> [1, padded_length]
    chunk = jnp.pad(chunk, (0, pad_to - len(chunk)))[None, :]

    # Create segment IDs (1 for real tokens, 0 for padding)
    segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)

    return chunk, segment_ids

def sample_next_token(logits, temperature=1.0, greedy: bool = True):
    """
    Purpose: Samples next token from model logits

    Args:
    - logits: Model output probabilities
    - temperature: Controls randomness (lower = more deterministic)
    - greedy: Whether to take most likely token or sample

    Returns: Next token ID
    """
    if greedy:
        # Take most likely token
        return jnp.argmax(logits, -1)
    else: 
        # Temperature sampling
        logits = logits / temperature # Higher temp = more random
        probs = jax.nn.softmax(logits, axis=-1)
        return jax.random.categorical(jax.random.PRNGKey(0), probs, axis=-1)

def sample_from_prompt(
        tokens: jax.Array, # Initial prompt tokens
        weights: Weights, # Model weights
        cache: KVCache, # KV cache for efficient generation
        cfg: Config, # Model config
        batch_idx: int = 0, # Which batch elements to generate for
        num_steps: int = 20, # How many tokens to generate
        greedy: bool = True, # Whether to use greedy sampling
):
    """
    Purpose: Generates sequence continuation from a prompt
    Uses caching for effiient autoregressive generation

    Returns: Tuple of (generated_tokens, updated_cache)
    """
    # Ensures prompt isn't too long
    assert len(tokens) <= cfg.max_seq_len

    # Pad to next power of 2 for efficient computation
    pad_to = 2 ** math.ceil(math.log2((len(tokens))))

    # Prepare initial prompt
    prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=0)

    # Update cache lengths for this batch
    cache = dataclasses.replace(
        cache,
        lengths=jax.lax.dynamic_update_index_in_dim(cache.lengths, 0, batch_idx, axis=0)
    )

    # Process initial prompt
    logits, cache,_ = jax.jit(
        forward,
        static_argnames="cfg"
    )(prompt, prompt_segment_ids, weights, cfg, cache)

    # Get logits for next token
    next_token_logit = logits[batch_idx, cache.lengths[batch_idx] - 1, :]

    # Generate tokens autoregressively
    tokens = []
    for _ in range(0, num_steps):
        # Sample next token
        next_token = sample_next_token(next_token_logit, greedy=greedy)[None]
        tokens.append(next_token[0])
        
        # Prepare single token for next step
        prompt, prompt_segment_ids = prepare_chunk(next_token, pad_to=1, pad_id=0)
        
        # Get next token logits
        logits, cache, _ = jax.jit(
            forward, 
            static_argnames="cfg"
        )(prompt, prompt_segment_ids, weights, cfg, cache)
        next_token_logit = logits[batch_idx, 0, :]

    return tokens, cache
