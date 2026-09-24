#!/usr/bin/env python3
"""
Run predict_multi.py for multiple datasets in parallel across all available GPUs
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse
import torch
import threading
from queue import Queue
import json
import re


def extract_metrics(output):
    """Extract metrics from the command output"""
    metrics = {}

    try:
        # Extract PSNR
        psnr_match = re.search(r'PSNR - Train:\s*([0-9.]+),\s*Test:\s*([0-9.]+)', output)
        if psnr_match:
            metrics['psnr_train'] = float(psnr_match.group(1))
            metrics['psnr_test'] = float(psnr_match.group(2))

        # Extract SSIM
        ssim_match = re.search(r'SSIM - Train:\s*([0-9.]+),\s*Test:\s*([0-9.]+)', output)
        if ssim_match:
            metrics['ssim_train'] = float(ssim_match.group(1))
            metrics['ssim_test'] = float(ssim_match.group(2))

        # Extract CD
        cd_match = re.search(r'CD\s*- Train:\s*([0-9.]+),\s*Test:\s*([0-9.]+)', output)
        if cd_match:
            metrics['cd_train'] = float(cd_match.group(1))
            metrics['cd_test'] = float(cd_match.group(2))

        # Extract EMD
        emd_match = re.search(r'EMD\s*- Train:\s*([0-9.]+),\s*Test:\s*([0-9.]+)', output)
        if emd_match:
            metrics['emd_train'] = float(emd_match.group(1))
            metrics['emd_test'] = float(emd_match.group(2))

    except Exception as e:
        print(f"Warning: Could not extract metrics: {e}")

    return metrics if metrics else None


def run_command(gpu_id, cmd, dataset_id, output_queue):
    """Run a single command on a specific GPU"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting: {dataset_id}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time

        # Extract metrics from output
        metrics = extract_metrics(result.stdout)

        print(f"[GPU {gpu_id}] ✓ Completed: {dataset_id} ({elapsed:.1f}s)")
        if metrics:
            print(f"         PSNR: Train={metrics['psnr_train']:.2f}, Test={metrics['psnr_test']:.2f}")
            print(f"         CD:   Train={metrics['cd_train']:.4f}, Test={metrics['cd_test']:.4f}")
            print(f"         EMD:  Train={metrics['emd_train']:.4f}, Test={metrics['emd_test']:.4f}")

        output_queue.put((dataset_id, True, elapsed, metrics))

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ Failed: {dataset_id} ({elapsed:.1f}s)")

        # Print complete error message
        error_lines = e.stderr.strip().split('\n')
        if error_lines:
            # Print last few lines which usually contain the actual error
            relevant_lines = error_lines[-10:]  # Last 10 lines
            print(f"  Error details:")
            for line in relevant_lines:
                if line.strip():  # Skip empty lines
                    print(f"    {line}")

        # Also save full error to a log file
        error_log = f"error_{dataset_id}_gpu{gpu_id}.log"
        with open(error_log, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"STDOUT:\n{e.stdout}\n")
            f.write(f"STDERR:\n{e.stderr}\n")
        print(f"  Full error saved to: {error_log}")

        output_queue.put((dataset_id, False, elapsed, None))


def worker(gpu_id, task_queue, output_queue):
    """Worker thread for each GPU"""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill to stop worker
            task_queue.task_done()
            break

        cmd, dataset_id = task
        run_command(gpu_id, cmd, dataset_id, output_queue)
        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description='Run predict_multi.py in parallel across GPUs')
    parser.add_argument('-c', '--config', type=str, default='config/genesismo',
                        help='Config directory path')
    parser.add_argument('-d', '--data', type=str, default='data/Genesis_MO',
                        help='Data directory path')
    parser.add_argument('-o', '--output', type=str, default='output/old',
                        help='Output directory path')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Only run these dataset IDs')
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Skip these dataset IDs')
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')
    parser.add_argument('--debug', action='store_true',
                        help='Run single test command to check for errors')
    parser.add_argument('--verbose', action='store_true',
                        help='Show full command output')

    args = parser.parse_args()

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available!")
        sys.exit(1)

    if args.gpus is not None:
        num_gpus = min(args.gpus, num_gpus)

    print(f"Found {num_gpus} GPU(s)")

    # Find all config files
    config_path = Path(args.config)
    config_files = sorted(config_path.glob('*.json'))

    if not config_files:
        print(f"No config files found in {config_path}")
        sys.exit(1)

    # Build command list
    commands = []
    skipped = []
    already_completed = []

    for config_file in config_files:
        dataset_id = config_file.stem

        # Skip if requested
        if dataset_id in args.skip:
            skipped.append(dataset_id)
            continue

        if args.only and dataset_id not in args.only:
            skipped.append(dataset_id)
            continue

        # Check paths
        data_path = Path(args.data) / dataset_id
        output_path = Path(args.output) / dataset_id

        if not data_path.exists():
            print(f"Warning: Data not found for {dataset_id} at {data_path}")
            skipped.append(dataset_id)
            continue

        if not output_path.exists():
            print(f"Warning: Output not found for {dataset_id} at {output_path}")
            skipped.append(dataset_id)
            continue

        # Check if already completed by looking for render output folders
        # These folders are created at the END of successful prediction
        img_render_path = output_path / f"{dataset_id}_img_render"
        img_gt_path = output_path / f"{dataset_id}_img_gt"

        if img_render_path.exists() and img_gt_path.exists() and not args.force:
            # Also check if they contain images
            render_images = list(img_render_path.glob("*.png"))
            if len(render_images) > 0:
                print(f"Already completed: {dataset_id} (found {len(render_images)} rendered images)")
                already_completed.append(dataset_id)
                continue

        # Build command
        gt_path = data_path / "point_clouds"
        cmd = [
            "python", "predict_multi.py",
            "-c", str(config_file),
            "-s", str(data_path),
            "-m", str(output_path),
            "--gt_path", str(gt_path),
            "--predict_frames", "30",
            "--train_frames", "20",
            "--reg_alpha",
            "--reg_scale"
        ]

        commands.append((cmd, dataset_id))

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total configs: {len(config_files)}")
    print(f"To run: {len(commands)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Already completed: {len(already_completed)}")
    print(f"GPUs to use: {num_gpus}")
    print(f"{'='*60}\n")

    if not commands:
        print("No commands to run!")
        return

    # Debug mode: run single test
    if args.debug:
        print("DEBUG MODE: Running single test command...")
        cmd, dataset_id = commands[0]
        print(f"Testing with dataset: {dataset_id}")
        print(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'

        try:
            if args.verbose:
                # Run with visible output
                result = subprocess.run(cmd, env=env, check=True)
            else:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                print("Test successful!")
        except subprocess.CalledProcessError as e:
            print(f"Test failed with return code: {e.returncode}")
            print(f"STDERR:\n{e.stderr}")
            print(f"STDOUT:\n{e.stdout}")
        return

    # Confirm
    response = input(f"Run {len(commands)} predictions on {num_gpus} GPUs? (y/n): ")
    if response.lower() != 'y':
        print("Aborted")
        return

    # Create task queue and output queue
    task_queue = Queue()
    output_queue = Queue()

    # Add all tasks to queue
    for cmd, dataset_id in commands:
        task_queue.put((cmd, dataset_id))

    # Add poison pills (one per worker)
    for _ in range(num_gpus):
        task_queue.put(None)

    # Start worker threads
    threads = []
    print(f"\nStarting {num_gpus} worker threads...")
    for gpu_id in range(num_gpus):
        t = threading.Thread(target=worker, args=(gpu_id, task_queue, output_queue))
        t.start()
        threads.append(t)
        print(f"  Worker for GPU {gpu_id} started")

    # Wait for all tasks to complete
    print("\nProcessing tasks...")
    start_time = time.time()
    task_queue.join()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Collect results
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())

    # Print final summary
    total_time = time.time() - start_time
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {len(successful)}/{len(results)}")

    if failed:
        print(f"\nFailed datasets ({len(failed)}):")
        for dataset_id, _, elapsed, _ in failed:
            print(f"  - {dataset_id} ({elapsed:.1f}s)")

    if successful:
        avg_time = sum(r[2] for r in successful) / len(successful)
        print(f"\nAverage time per successful run: {avg_time:.1f}s")

        # Show metrics summary
        metrics_available = [r for r in successful if r[3] is not None]
        if metrics_available:
            print(f"\n{'='*60}")
            print("METRICS SUMMARY (Test Set)")
            print(f"{'='*60}")
            print(f"{'Dataset':<10} {'PSNR':<8} {'SSIM':<8} {'CD':<10} {'EMD':<10}")
            print("-" * 60)

            for dataset_id, _, _, metrics in sorted(metrics_available, key=lambda x: x[0]):
                psnr = metrics.get('psnr_test', -1)
                ssim = metrics.get('ssim_test', -1)
                cd = metrics.get('cd_test', -1)
                emd = metrics.get('emd_test', -1)
                print(f"{dataset_id:<10} {psnr:>7.2f}  {ssim:>7.4f}  {cd:>9.4f}  {emd:>9.4f}")

            # Calculate averages
            avg_psnr = sum(r[3].get('psnr_test', 0) for r in metrics_available) / len(metrics_available)
            avg_ssim = sum(r[3].get('ssim_test', 0) for r in metrics_available) / len(metrics_available)
            avg_cd = sum(r[3].get('cd_test', 0) for r in metrics_available) / len(metrics_available)
            avg_emd = sum(r[3].get('emd_test', 0) for r in metrics_available) / len(metrics_available)

            print("-" * 60)
            print(f"{'AVERAGE':<10} {avg_psnr:>7.2f}  {avg_ssim:>7.4f}  {avg_cd:>9.4f}  {avg_emd:>9.4f}")
            print(f"{'='*60}\n")

    # Save results summary
    summary_file = Path(args.output) / "prediction_summary.json"
    summary_data = {
        'total_time': total_time,
        'num_gpus': num_gpus,
        'successful': [r[0] for r in successful],
        'failed': [r[0] for r in failed],
        'skipped': skipped,
        'already_completed': already_completed,
        'average_time': avg_time if successful else 0
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print("Done!")


if __name__ == "__main__":
    main()