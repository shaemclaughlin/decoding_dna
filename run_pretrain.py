"""
Pre-training script for DNA transformer model on eukaryotic genomes
"""

import os
import datetime

# List of eukaryote datasets for pre-training
EUKARYOTES = [
    'drosophila_genome_8192bp_bins_no_N',
    'macaque_genome_8192bp_bins_no_N', 
    'mouse_genome_8192bp_bins_no_N',
    'zebrafish_genome_8192bp_bins_no_N',
    '8kb_genomic_bins_with_sequences_GW17IPC'
]

# Get timestamp for unique identification
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CONFIG = {
    # Model architecture
    "d_model": 2048,
    "num_layers": 12,
    "query_heads": 8,
    "key_heads": 8,
    "ffw_multiplier": 4,
    "key_dim": 128,
    "vocab_size": 8,
    "max_seq_len": 8192,
    
    # Training params
    "batch_size": 16,
    "max_lr": 3e-4,
    "min_lr": 1e-5,
    "warmup_steps": 1000,
    "total_steps": 100000,
    
    # Paths - using Google Cloud Storage bucket with unique paths
    "data_dir": "gs://minformer_data",
    "checkpoint_dir": f"gs://minformer_data/pretrained_ckpt/shae_v1_{timestamp}",
    "log_dir": f"gs://minformer_data/logs/shae_v1_{timestamp}",
    
    # New: Activation storage configuration
    "activation_dir": f"gs://minformer_data/activations/shae_v1_{timestamp}",
    "store_activations_every": 1000,  # Store activations every N steps
    "activations_per_store": 1000,    # Number of activations to store each time
    "middle_layer_idx": 6,            # Index of middle layer (for 12 layers, this is 6)
    
    # Logging
    "log_every": 100,
    "eval_every": 1000,
    "checkpoint_interval": 5000
}

def main():
    # Create file patterns for pre-training data
    stage_1 = [f"{CONFIG['data_dir']}/{e}/tfrecords/record_*.tfrecord" 
               for e in EUKARYOTES]
    
    # Human genome for stage 2
    stage_2 = [f"{CONFIG['data_dir']}/shae_8k/tfrecords/record_*.tfrecord"]

    # Convert config to command line arguments
    cmd_args = [
        f"python train.py",
        f"--d_model {CONFIG['d_model']}",
        f"--num_layers {CONFIG['num_layers']}",
        f"--query_heads {CONFIG['query_heads']}",
        f"--key_heads {CONFIG['key_heads']}",
        f"--ffw_multiplier {CONFIG['ffw_multiplier']}",
        f"--key_dim {CONFIG['key_dim']}",
        f"--vocab_size {CONFIG['vocab_size']}",
        f"--max_seq_len {CONFIG['max_seq_len']}",
        f"--batch_size {CONFIG['batch_size']}",
        f"--max_lr {CONFIG['max_lr']}",
        f"--min_lr {CONFIG['min_lr']}",
        f"--warmup_steps {CONFIG['warmup_steps']}",
        f"--total_steps {CONFIG['total_steps']}",
        f"--checkpoint_dir {CONFIG['checkpoint_dir']}",
        f"--log_dir {CONFIG['log_dir']}",
        f"--log_every {CONFIG['log_every']}",
        f"--eval_every {CONFIG['eval_every']}",
        f"--checkpoint_interval {CONFIG['checkpoint_interval']}",
        # New arguments for activation storage
        f"--activation_dir {CONFIG['activation_dir']}",
        f"--store_activations_every {CONFIG['store_activations_every']}",
        f"--activations_per_store {CONFIG['activations_per_store']}",
        f"--middle_layer_idx {CONFIG['middle_layer_idx']}",
        "--store_activations",  # Flag to enable activation storage
        "--dataset shae_8k",  # Use the shae dataset loader
    ]

    # Print settings for verification
    print("\nTraining Configuration:")
    print("=" * 50)
    for key, value in CONFIG.items():
        print(f"{key:20}: {value}")
    print("=" * 50)

    # Execute training command
    cmd = " ".join(cmd_args)
    print("\nExecuting command:")
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    main()
