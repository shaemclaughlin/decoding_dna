"""
Training script for DNA sequence transformer with enhanced tracking

For open genome:

python3 projects/bio/train.py --checkpoint_dir=/tmp/bio_checkpoints/test_run --checkpoint_interval=10000 --max_seq_len=16384 --data_dir=gs://minformer_data/open-genome-imgpr/tfrecords/stage1/train_v3/ --log_every=10
python3 projects/bio/train.py --checkpoint_dir=/tmp/bio_checkpoints/test_run_fp32norm --checkpoint_interval=1000 --max_seq_len=8192 --dataset=shae_8k --log_every=10

"""

import argparse
import functools
import os
from datetime import datetime
from typing import Any
from collections import defaultdict 
import json
from pathlib import Path

# Assuming these are in the same directory or in the Python path
import data
import data_shae
import jax
import jax.numpy as jnp
import model
import numpy as np
from jax.profiler import trace
from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter

# New class for tracking training stats
class TrainingStats:
    def __init__(self):
        self.total_tokens = 0
        self.tokens_per_dataset = defaultdict(int)
        self.sequences_per_dataset = defaultdict(int)
        self.tokens_per_step = []
        self.sequence_lengths = []
        self.steps_without_progress = 0

    def update(self, batch, dataset_name):
        """Update stats with new batch information"""
        num_tokens = np.sum(batch["segment_ids"] != 0)
        seq_length = batch["x"].shape[1]

        self.total_tokens += num_tokens
        self.tokens_per_dataset[dataset_name] += num_tokens
        self.sequences_per_dataset[dataset_name] += batch["x"].shape[0]
        self.tokens_per_step.append(num_tokens)
        self.sequence_lengths.append(seq_length)

    def get_summary(self):
        """Get current stats summary"""
        return {
            "total_tokens": self.total_tokens,
            "tokens_per_dataset": dict(self.tokens_per_dataset),
            "sequences_per_dataset": dict(self.sequences_per_dataset),
            "avg_tokens_per_step": np.mean(self.tokens_per_step),
            "avg_sequence_length": np.mean(self.sequence_lengths),
            "max_sequence_length": max(self.sequence_lengths),
            "min_sequence_length": min(self.sequence_lengths)
        }

    def save_stats(self, save_dir: str):
        """Save statistics to file"""
        stats = self.get_summary()
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

def parse_args():
    """Parse command line arguments with added tracking options"""
    parser = argparse.ArgumentParser(description="DNA Sequence Training Script")

    # Model architecture arguments
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension")
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier")
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--vocab_size", type=int, default=8, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=16384, help="Maximum sequence length")
    
    # Training hyperparameters
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=30000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Logging and checkpoint arguments
    parser.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate model every N steps")
    parser.add_argument("--data_dir", type=str, default="data/tfrecords/", help="Directory containing TFRecord files")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/shae", help="Base directory for TensorBoard logs")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/tmp/dna_checkpoints", help="Directory for saving checkpoints"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["human-genome-8192", "open-genome-imgpr", "shae_8k"],
        default="open-genome-imgpr",
        help="Type of dataset to download and process",
    )

    # New arguments for enhanced tracking
    parser.add_argument("--stats_dir", type=str, default="training_stats",
                        help="Directory to save training statistics")
    parser.add_argument("--track_memory", action="store_true",
                        help="Track memory usage during training")
    parser.add_argument("--detailed_logging", action="store_true",
                        help="Enable more detailed logging of training progress")
    
    # Add new arguments for activation storage
    parser.add_argument("--store_activations", action="store_true", 
                       help="Whether to store activations for SAE training")
    parser.add_argument("--activation_dir", type=str, default=None,
                       help="Directory to store activations")
    parser.add_argument("--store_activations_every", type=int, default=1000,
                       help="Store activations every N steps")
    parser.add_argument("--activations_per_store", type=int, default=1000,
                       help="Number of activations to store each time")
    parser.add_argument("--middle_layer_idx", type=int, default=6,
                       help="Which layer to extract activations from")

    return parser.parse_args()


def clean_key(key: str) -> str:
    """
    Clean metric keys for TensorBoard logging

    Purpose: Convert internal key names to Tensorboard-friendly format
    Input: Raw key string (e.g. "['layer.0.attention']")
    Output: Clean key string (e.g. "layer/0/attention")
    """
    cleaned = key.replace("['", "").replace("']", "")
    cleaned = cleaned.replace(".", "/")
    return cleaned


def flatten_pytree(tree):
    """
    Flatten JAX pytree for logging

    Purpose: Convert nested JAX structures into flat key-value pairs
    Input: JAX pytree structure
    Output: List of (key, value) tuples
    """
    leaves = jax.tree_util.tree_map_with_path(lambda p, x: (clean_key(jax.tree_util.keystr(p)), x), tree)
    return jax.tree_util.tree_leaves(leaves, is_leaf=lambda x: isinstance(x, tuple))


def log_metrics(writer: SummaryWriter, metrics: dict, step: int, training_stats: TrainingStats = None):
    """
    Log metrics to TensorBoard with enhanced tracking

    Purpose: Record training metrics and statistics
    Inputs:
    - writer: TensorBoard SummaryWriter
    - metrics: Dictionary of metrics to log
    - step: Current training step 
    - training_stats: Optional TrainingStats object for enhanced tracking
    """

    # Log standard metrics
    flat_metrics = flatten_pytree(metrics)
    for key, value in flat_metrics:
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)
    
    # Log training stats if available
    if training_stats is not None:
        stats = training_stats.get_summary()
        writer.add_scalar("tokens/total", stats["total_tokens"], step)
        writer.add_scalar("tokens/avg_per_step", stats["avg_tokens_per_step"], step)
        writer.add_scalar("sequences/avg_length", stats["avg_sequence_length"], step)

def save_activations(activation_dir: str, step: int, activations: Any, metadata: Any):
    """Save activations and metadata to storage"""
    step_dir = os.path.join(activation_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Save activations array
    np.save(os.path.join(step_dir, "activations.npy"), activations)
    
    # Save metadata (sequences, positions, etc)
    with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

def main():
    """Main training function with enhanced tracking and debugging"""
    args = parse_args()

    # Create directories including activation directory if needed
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.store_activations:
        os.makedirs(args.activation_dir, exist_ok=True)

    # Initialize training statistics tracker
    training_stats = TrainingStats()

    # Create a unique log directory name with key configuration parameters
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

    # Data setup with dataset tracking
    if args.dataset == "open-genome-imgpr":
        dataset_pattern = str(args.data_dir) + "record_*.tfrecord"
        print(f"Loading open-genome data from: {dataset_pattern}")
        iter = data_hf.create_iterator(dataset_pattern, batch_size=args.batch_size, shuffle=True)
        process_batch = model.process_batch
        dataset_type = "open-genome"


    elif args.dataset == "shae_8k":
        # List of eukaryote datasets
        eukaroytes = ['drosophila_genome_8192bp_bins_no_N',
                    'macaque_genome_8192bp_bins_no_N',
                    'mouse_genome_8192bp_bins_no_N',
                    'zebrafish_genome_8192bp_bins_no_N',
                    '8kb_genomic_bins_with_sequences_GW17IPC']
        
        # Create file patterns for each dataset
        stage_1 = [f"gs://minformer_data/{e}/tfrecords/record_*.tfrecord" for e in eukaroytes]
        # Do second stage on human only.
        stage_2 = ["gs://minformer_data/shae_8k/tfrecords/record_*.tfrecord"]
        
        print("\nDataset Setup:")
        print("Stage 1 (Eukaryotes):")
        for pattern in stage_1:
            print(f"- {pattern}")
        print("\nStage 2 (Human):")
        print(f"- {stage_2[0]}")
        
        iter = data_shae.create_iterator(
           stage_1=stage_1, 
           stage_2=stage_2, 
           batch_size=args.batch_size, 
           shuffle=True
        )
        process_batch = model.process_batch_shae
        dataset_type = "eukaryotes+human"

    # Model configuration
    cfg = model.Config(
        d_model=args.d_model,
        ffw_multiplier=args.ffw_multiplier,
        query_heads=args.query_heads,
        key_heads=args.key_heads,
        num_layers=args.num_layers,
        key_dim=args.key_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        causal=True,
        use_attn_kernel=True,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.bfloat16,
        rules=model.fsdp_rules,
        mesh=model.create_mesh(),
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )

    # Print detailed configuration
    print("\nModel Configuration:")
    print("=" * 50)
    for field in cfg.__dataclass_fields__:
        value = getattr(cfg, field)
        print(f"{field:20}: {value}")
    print("=" * 50)

    # Calculate and print theoretical token counts
    theoretical_tokens = args.batch_size * args.max_seq_len * args.total_steps
    print(f"\nTheoretical Maximum Tokens:")
    print(f"Per batch: {args.batch_size * args.max_seq_len:,}")
    print(f"Total: {theoretical_tokens:,}")
    print("=" * 50)

    # Checkpoint manager setup
    ckpt_manager = model.make_mngr(path=args.checkpoint_dir)
    print(f"\nCheckpoint directory: {args.checkpoint_dir}")

    # Initialize or load weights and optimizer state
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        weights, opt_state = model.load(ckpt_manager, cfg)
        start_step = ckpt_manager.latest_step()
        print(f"Resumed from step: {start_step}")
    else:
        print("Initializing new weights...")
        weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
        opt_state = model.init_optimizer_state(weights)
        start_step = 0

    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames="cfg")
    step = functools.partial(step, cfg=cfg)

    # Training loop with enhanced tracking
    with SummaryWriter(log_dir) as writer:
        # Log hyperparameters
        writer.add_hparams(
            {
                "d_model": cfg.d_model,
                "num_layers": cfg.num_layers,
                "query_heads": cfg.query_heads,
                "key_heads": cfg.key_heads,
                "max_lr": cfg.max_lr,
                "min_lr": cfg.min_lr,
                "warmup_steps": cfg.warmup_steps,
                "total_steps": cfg.total_steps,
                "batch_size": args.batch_size,
                "max_seq_len": cfg.max_seq_len,
                "dataset_type": dataset_type
            },
            {},
        )

        print("\nStarting training loop...")
        print(f"Will train for {cfg.total_steps:,} steps")
        print(f"Logging every {args.log_every} steps")
        print(f"Checkpoing every {args.checkpoint_interval} steps")
        print("=" * 50 + "\n")

        # Training loop with progress tracking
        for i in range(start_step, cfg.total_steps):
            try:
                # Get and process next batch
                batch = next(iter)
                processed_batch = process_batch(batch, cfg, step_idx=i)
                batch = jax.device_put(processed_batch, model.input_shardings(cfg.mesh, cfg.rules))

                # Determine if we should store activations this step 
                should_store_activations = (
                    args.store_activations and 
                    i % args.store_activations_every == 0
                )

                # Update training stats
                current_tokens = np.sum(batch["segment_ids"] != 0)
                training_stats.update(batch, dataset_type)

                # Always profile on the first step so that we can think about optimisations.
                if i == 0:
                    with trace(log_dir):
                        loss, weights, opt_state, internals = step(
                            weights,
                            batch["x"],
                            batch["segment_ids"],
                            batch["y"],
                            opt_state,
                            i,
                            aux=batch["aux"],
                            store_activations=should_store_activations,
                            middle_layer_idx=args.middle_layer_idx
                        )
                        jax.block_until_ready(loss)
                else:
                    loss, weights, opt_state, internals = step(
                        weights, 
                        batch["x"], 
                        batch["segment_ids"], 
                        batch["y"], 
                        opt_state, 
                        i, 
                        aux=batch["aux"],
                        store_activations=should_store_activations,
                        middle_layer_idx=args.middle_layer_idx
                    )
                
                # NEW: Store activations if they were collected
                if should_store_activations and internals.get("stored_activations") is not None:
                    metadata = {
                        "step": i,
                        "sequences": batch["x"].tolist(),  # Convert to list for JSON
                        "segment_ids": batch["segment_ids"].tolist(),
                        "batch_size": args.batch_size,
                        "sequence_length": args.max_seq_len,
                        "layer_idx": args.middle_layer_idx,
                    }
            
                    save_activations(
                        args.activation_dir,
                        i,
                        internals["stored_activations"],
                        metadata
                    )


                # Detailed logging every N steps
                if i % args.log_every == 0:
                    stats = training_stats.get_summary()

                    # Log standard metrics
                    writer.add_scalar("loss", loss, i)
                    writer.add_scalar("accuracy", internals["accuracy"], i)
                    writer.add_scalar("num_tokens_per_batch", current_tokens, i)

                    # Log detailed statistics
                    writer.add_scalar("tokens/total", stats["total_tokens"], i)
                    writer.add_scalar("tokens/avg_per_step", stats["avg_tokens_per_step"], i)
                    writer.add_scalar("sequences/avg_length", stats["avg_sequence_length"], i)
                    
                    # Print progress
                    progress = (i / cfg.total_steps) * 100
                    print(f"\nStep {i:,}/{cfg.total_steps:,} ({progress:.1f}%)")
                    print(f"Loss: {loss:.4f}, Accuracy: {internals['accuracy']:.4f}")
                    print(f"Tokens this batch: {current_tokens:,}")
                    print(f"Total tokens: {stats['total_tokens']:,}")
                    print(f"Average tokens/batch: {stats['avg_tokens_per_step']:.1f}")
                    
                    # Memory tracking if enabled
                    if args.track_memory:
                        memory_stats = jax.live_buffers()
                        total_memory = sum(x.size * x.dtype.itemsize for x in memory_stats)
                        writer.add_scalar("memory/total_gb", total_memory / 1e9, i)
                        print(f"Memory usage: {total_memory/1e9:.2f} GB")

                    log_metrics(writer, internals, i, training_stats)
                
                # Save checkpoint
                if i > 0 and i % args.checkpoint_interval == 0:
                    print(f"\nSaving checkpoint at step {i}")
                    model.save(ckpt_manager, weights, opt_state, i)
                    
                    # Also save current statistics
                    stats_path = os.path.join(args.stats_dir, f"stats_step_{i}.json")
                    training_stats.save_stats(stats_path)
            
            except Exception as e:
                print(f"\nError at step {i}:")
                print(str(e))

                # Save emergency statistics before raising error
                emergency_stats_path = os.path.join(args.stats_dir, f"emergency_stats_step_{i}.json")
                training_stats.save_stats(emergency_stats_path)
                raise e
        
        # Training complete - save final statistics
        final_stats_path = os.path.join(args.stats_dir, "final_training_stats.json")
        training_stats.save_stats(final_stats_path)
        
        print("\nTraining completed!")
        print(f"TensorBoard logs saved in: {log_dir}")
        print(f"Final statistics saved in: {final_stats_path}")
        
        # Print final summary
        final_stats = training_stats.get_summary()
        print("\nFinal Training Statistics:")
        print("=" * 50)
        print(f"Total tokens processed: {final_stats['total_tokens']:,}")
        print(f"Average tokens per batch: {final_stats['avg_tokens_per_step']:.1f}")
        print(f"Average sequence length: {final_stats['avg_sequence_length']:.1f}")
        print(f"Maximum sequence length: {final_stats['max_sequence_length']:,}")
        print("=" * 50)

if __name__ == "__main__":
    main()
