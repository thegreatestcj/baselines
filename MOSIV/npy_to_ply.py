import numpy as np
import os
from plyfile import PlyData, PlyElement

def npy_to_ply_frames(npy_path, output_dir):
    """Convert NPY trajectory file to individual PLY files per frame"""

    # Load the trajectory
    print(f"Loading trajectory from: {npy_path}")
    trajectory = np.load(npy_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get number of frames
    n_frames = len(trajectory)
    print(f"Total frames: {n_frames}")

    # Convert each frame to PLY
    for frame_idx in range(n_frames):
        # Get particle positions for this frame
        positions = trajectory[frame_idx]

        # Filter out invalid particles (those at origin or with all zeros)
        valid_mask = np.any(positions != 0, axis=1)
        valid_positions = positions[valid_mask]

        if len(valid_positions) == 0:
            print(f"Frame {frame_idx}: No valid particles, skipping")
            continue

        # Create vertex array with positions
        num_particles = len(valid_positions)
        vertex_data = np.zeros(num_particles, dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4')
        ])

        vertex_data['x'] = valid_positions[:, 0]
        vertex_data['y'] = valid_positions[:, 1]
        vertex_data['z'] = valid_positions[:, 2]

        # Create PLY element and data
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_element])

        # Save to file
        output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.ply')
        ply_data.write(output_path)

        print(f"Frame {frame_idx}: Saved {num_particles} particles to {output_path}")

    print(f"Conversion complete! {n_frames} frames saved to {output_dir}")

if __name__ == "__main__":
    # Input NPY file
    npy_file = "/mnt/c/Users/Chunjiang/Downloads/02-pred_traj.npy"

    # Check if file exists
    if not os.path.exists(npy_file):
        # Try alternative name
        npy_file = "/mnt/c/Users/Chunjiang/Downloads/02-pred.npy"

    if not os.path.exists(npy_file):
        print(f"Error: Could not find NPY file at {npy_file}")
        print("Please check the file path and name")
        exit(1)

    # Output directory
    output_dir = "/mnt/c/Users/Chunjiang/Downloads/02_ply_frames"

    # Convert
    npy_to_ply_frames(npy_file, output_dir)