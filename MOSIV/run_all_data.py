#!/usr/bin/env python
import os
import subprocess
import json
from pathlib import Path
import argparse
from glob import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def run_single_training(cmd_info):
    """Run a single training job on a specific GPU"""
    dataset_id = cmd_info['dataset_id']
    cmd = cmd_info['cmd']

    # Set CUDA_VISIBLE_DEVICES if gpu_id is specified
    env = os.environ.copy()
    if 'gpu_id' in cmd_info:
        env['CUDA_VISIBLE_DEVICES'] = str(cmd_info['gpu_id'])

    try:
        start_time = time.time()
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time

        return {'dataset_id': dataset_id, 'status': 'success', 'time': elapsed, 'gpu_id': cmd_info.get('gpu_id')}
    except subprocess.CalledProcessError as e:
        # Include stdout and stderr in error info
        error_msg = f"Exit code: {e.returncode}"

        # Get the last part of stderr which usually contains the actual error
        if e.stderr:
            stderr_lines = e.stderr.strip().split('\n')
            # Get last 20 lines which should contain the actual error
            important_lines = stderr_lines[-20:]
            error_msg += f"\nLast error lines:\n" + '\n'.join(important_lines)

        return {'dataset_id': dataset_id, 'status': 'failed', 'error': error_msg, 'gpu_id': cmd_info.get('gpu_id')}

def main():
    parser = argparse.ArgumentParser(description='Run training for all GenesisMO datasets')
    parser.add_argument('--config', type=str, nargs='+', required=True,
                       help='Config files or directory pattern (e.g., config/genesismo or config/genesismo/*)')
    parser.add_argument('--data', type=str, required=True,
                       help='Root data directory (e.g., data/Genesis_MO)')
    parser.add_argument('--output', type=str, required=True,
                       help='Root output directory (e.g., output)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only print commands without running')
    parser.add_argument('--skip', nargs='+', default=[],
                       help='Dataset IDs to skip (e.g., --skip 01 07)')
    parser.add_argument('--only', nargs='+', default=None,
                       help='Only run these dataset IDs (e.g., --only 02 03)')
    parser.add_argument('--nproc_per_node', type=int, default=1,
                       help='Number of processes per node for torchrun')
    parser.add_argument('--parallel', action='store_true',
                       help='Run different datasets in parallel on different GPUs')
    parser.add_argument('--gpus', type=str, default=None,
                       help='GPU IDs for parallel execution (e.g., "0,1,2,3")')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if output JSON already exists')

    args = parser.parse_args()

    # Handle config files - could be multiple files or a directory
    config_files = []
    for config_item in args.config:
        if os.path.isdir(config_item):
            # If it's a directory, find all JSON files in it
            config_files.extend(sorted(glob(os.path.join(config_item, '*.json'))))
        elif os.path.isfile(config_item) and config_item.endswith('.json'):
            # If it's a JSON file, add it directly
            config_files.append(config_item)
        else:
            # Try as a glob pattern
            matched = glob(config_item)
            if matched:
                config_files.extend([f for f in matched if f.endswith('.json')])

    config_files = sorted(list(set(config_files)))  # Remove duplicates and sort

    if not config_files:
        print(f"Error: No config files found in: {args.config}")
        sys.exit(1)

    print(f"Found {len(config_files)} config files")

    commands = []
    skipped = []
    missing_data = []
    already_completed = []

    for config_path in config_files:
        # Extract dataset ID from config filename (e.g., "07" from "config/genesismo/07.json")
        config_path = Path(config_path)
        dataset_id = config_path.stem

        # Check if should skip
        if dataset_id in args.skip:
            skipped.append(dataset_id)
            continue

        if args.only and dataset_id not in args.only:
            skipped.append(dataset_id)
            continue

        # Construct data and output paths
        data_path = Path(args.data) / dataset_id
        output_path = Path(args.output) / dataset_id

        # Check if data exists
        if not data_path.exists():
            print(f"Warning: Data not found for {dataset_id} at {data_path}")
            missing_data.append(dataset_id)
            continue

        # Check if debug folder exists (indicating completed training)
        debug_path = output_path / "debug"
        if debug_path.exists() and not args.force:
            print(f"Found existing debug folder for {dataset_id} at {debug_path}, skipping...")
            already_completed.append(dataset_id)
            continue

        # Build the training command
        if args.nproc_per_node > 1:
            # Use torchrun for distributed training
            cmd = [
                "torchrun",
                f"--nproc_per_node={args.nproc_per_node}",
                "train_dynamic_MO.py",
                "-c", str(config_path),
                "-s", str(data_path),
                "-m", str(output_path),
                "--reg_scale",
                "--reg_alpha",
                "--use_wandb"
            ]
        else:
            # Regular python command
            cmd = [
                "python", "train_dynamic_MO.py",
                "-c", str(config_path),
                "-s", str(data_path),
                "-m", str(output_path),
                "--reg_scale",
                "--reg_alpha",
                "--use_wandb"
            ]

        commands.append({
            'dataset_id': dataset_id,
            'config': str(config_path),
            'data': str(data_path),
            'output': str(output_path),
            'cmd': cmd,
            'cmd_str': ' '.join(cmd)
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total configs found: {len(config_files)}")
    print(f"Commands to run: {len(commands)}")
    if already_completed:
        print(f"Already completed: {len(already_completed)} - {', '.join(already_completed)}")
    if skipped:
        print(f"Skipped: {len(skipped)} - {', '.join(skipped)}")
    if missing_data:
        print(f"Missing data: {len(missing_data)} - {', '.join(missing_data)}")
    print(f"{'='*60}\n")

    if not commands:
        print("No commands to run!")
        sys.exit(0)

    # Print all commands
    print("Commands to execute:")
    print("-" * 60)
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] Dataset: {cmd_info['dataset_id']}")
        print(f"  Config: {cmd_info['config']}")
        print(f"  Data:   {cmd_info['data']}")
        print(f"  Output: {cmd_info['output']}")
        print(f"  Command: {cmd_info['cmd_str']}")

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No commands executed")
        print("Remove --dry_run to actually run these commands")
        sys.exit(0)

    # Execute commands
    print("\n" + "="*60)
    print(f"Starting execution of {len(commands)} training jobs...")

    # Create output directories
    for cmd_info in commands:
        Path(cmd_info['output']).mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    if args.parallel and args.gpus:
        # Parallel execution on multiple GPUs
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        num_gpus = len(gpu_ids)
        print(f"Parallel mode: Using {num_gpus} GPUs: {gpu_ids}")
        print(f"Total datasets to process: {len(commands)}")
        print("="*60 + "\n")

        # Create a pool of workers, one per GPU
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            # Prepare futures dictionary
            future_to_info = {}

            # Submit initial batch of jobs (one per GPU)
            pending_commands = list(commands)
            active_gpus = {}  # Maps GPU ID to active future

            # Start initial jobs
            for gpu_id in gpu_ids:
                if pending_commands:
                    cmd_info = pending_commands.pop(0)
                    cmd_info['gpu_id'] = gpu_id
                    print(f"[GPU {gpu_id}] Starting dataset {cmd_info['dataset_id']}")

                    future = executor.submit(run_single_training, cmd_info)
                    future_to_info[future] = cmd_info
                    active_gpus[gpu_id] = future

            # Process jobs as they complete
            while future_to_info:
                # Wait for any job to complete
                for future in as_completed(future_to_info):
                    cmd_info = future_to_info.pop(future)
                    gpu_id = cmd_info['gpu_id']

                    # Remove from active GPUs
                    if gpu_id in active_gpus and active_gpus[gpu_id] == future:
                        del active_gpus[gpu_id]

                    # Process result
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            successful.append(result['dataset_id'])
                            print(f"[GPU {gpu_id}] ✓ Dataset {result['dataset_id']} completed in {result['time']:.1f}s")
                        else:
                            failed.append(result['dataset_id'])
                            print(f"[GPU {gpu_id}] ✗ Dataset {result['dataset_id']} failed")
                            # Print error details
                            if 'error' in result:
                                print(f"    Error details: {result['error']}")
                    except Exception as e:
                        failed.append(cmd_info['dataset_id'])
                        print(f"[GPU {gpu_id}] ✗ Dataset {cmd_info['dataset_id']} failed with exception: {e}")

                    # Submit new job to this GPU if available
                    if pending_commands:
                        new_cmd = pending_commands.pop(0)
                        new_cmd['gpu_id'] = gpu_id
                        print(f"[GPU {gpu_id}] Starting dataset {new_cmd['dataset_id']}")

                        new_future = executor.submit(run_single_training, new_cmd)
                        future_to_info[new_future] = new_cmd
                        active_gpus[gpu_id] = new_future
                    else:
                        print(f"[GPU {gpu_id}] Finished - no more datasets to process")

                    # Print progress
                    completed = len(successful) + len(failed)
                    remaining = len(pending_commands) + len(active_gpus)
                    print(f"Progress: {completed}/{len(commands)} completed, {remaining} remaining")
                    print("-" * 40)

    else:
        # Sequential execution (original behavior)
        print("Sequential mode")
        print("="*60)

        for i, cmd_info in enumerate(commands, 1):
            dataset_id = cmd_info['dataset_id']

            print(f"\n{'='*60}")
            print(f"[{i}/{len(commands)}] Running dataset: {dataset_id}")
            print(f"Command: {cmd_info['cmd_str']}")
            print(f"{'='*60}\n")

            try:
                result = subprocess.run(cmd_info['cmd'], check=True)
                print(f"\n✓ Dataset {dataset_id} completed successfully")
                successful.append(dataset_id)
            except subprocess.CalledProcessError as e:
                print(f"\n✗ Dataset {dataset_id} failed with exit code {e.returncode}")
                failed.append(dataset_id)

                if i < len(commands):
                    response = input("\nContinue with remaining datasets? (y/n): ")
                    if response.lower() != 'y':
                        print("Aborted by user")
                        break
            except KeyboardInterrupt:
                print("\n\nInterrupted by user (Ctrl+C)")
                break

    # Final summary
    print(f"\n{'='*60}")
    print("EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(commands)}")
    if successful:
        print(f"  {', '.join(successful)}")
    print(f"Failed: {len(failed)}/{len(commands)}")
    if failed:
        print(f"  {', '.join(failed)}")

if __name__ == "__main__":
    main()