"""
Step 1: Generate quick pressure visualizations for all subjects/OMs.
This helps identify which trials have faulty sensors that need fixing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import WORK_PRESSURE, QC_PRESSURE  # noqa: E402

def save_pressure_video(data, output_path, fps=50, max_frames=None):
    """
    Create side-by-side visualization of left and right foot pressure.
    
    Args:
        data: (N, 60, 21, 2) array - [frames, rows, cols, feet]
        output_path: where to save the video
        fps: frames per second
        max_frames: maximum number of frames to render (None = all frames)
    """
    N = data.shape[0]
    
    # Limit frames if requested
    if max_frames is not None:
        N = min(N, max_frames)
        data = data[:N]
    
    # Set up colormap
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='black')
    
    # Use 99th percentile for vmax to avoid scaling issues from outliers
    valid_data = data[~np.isnan(data) & (data > 0)]
    if len(valid_data) > 0:
        vmin = 0  # Always start at 0 for pressure
        vmax = np.percentile(valid_data, 99)
        actual_max = np.nanmax(data)
        print(f"    Data range: 0 to {actual_max:.1f} (display capped at 99th percentile: {vmax:.1f})")
    else:
        vmin = 0
        vmax = 1
        print(f"    Warning: No valid pressure data found")
    
    def combine_frames(frame):
        """Combine left and right foot side-by-side."""
        left = frame[:, :, 0]
        right = frame[:, :, 1]
        combined = np.ma.masked_invalid(np.hstack((left, right)))
        return combined
    
    # Set up figure
    initial_frame = combine_frames(data[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(initial_frame, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(10.5, -2, 'Left', ha='center', va='center', fontsize=10)
    ax.text(31.5, -2, 'Right', ha='center', va='center', fontsize=10)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pressure")
    
    # Add frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', color='white', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def update(frame_idx):
        combined = combine_frames(data[frame_idx])
        im.set_data(combined)
        frame_text.set_text(f'Frame {frame_idx}/{N-1}')
        return [im, frame_text]
    
    ani = animation.FuncAnimation(fig, update, frames=N, blit=True, interval=1000/fps)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    print(f"    → Video saved to: {output_path}")

def process_subject(subject_name, data_dir, output_dir, max_frames=None, append=None):
    """Generate pressure visualizations for all OMs for a subject."""
    
    subject_data_dir = Path(data_dir) / subject_name
    subject_output_dir = Path(output_dir) / subject_name / "pressure_viz" if append is None else Path(output_dir) / subject_name / f"pressure_viz_{append}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all OM directories
    om_dirs = sorted([d for d in subject_data_dir.iterdir() 
                     if d.is_dir() and d.name.startswith("OM")])
    
    if not om_dirs:
        print(f"No OM directories found for {subject_name}")
        return
    
    print(f"\nProcessing {subject_name}")
    print("="*60)
    
    for om_dir in om_dirs:
        om_idx = om_dir.name
        pressure_file = om_dir / "Original_Pressure.npy"
        
        if not pressure_file.exists():
            print(f"  ⚠ Skipping {om_idx}: pressure file not found")
            continue
        
        print(f"  Processing {om_idx}...")
        
        # Load data
        data = np.load(pressure_file)
        print(f"    Loaded data shape: {data.shape}")
        
        # Generate video
        output_path = subject_output_dir / f"{om_idx}_pressure.mp4"
        save_pressure_video(data, output_path, fps=50, max_frames=max_frames)
        
        print(f"  ✓ Completed {om_idx}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate pressure visualization videos for all subjects"
    )
    parser.add_argument("--data_dir", default=WORK_PRESSURE,
                       help="Directory containing pressure data")
    parser.add_argument("--output_dir", default=QC_PRESSURE,
                       help="Directory to save visualizations")
    parser.add_argument("--subjects", nargs="+", required=True,
                       help="List of subjects to process")
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process per video (default: all frames)')
    parser.add_argument('--append', type=str, default=None,
                       help='Optional string to append to output directory name')
    args = parser.parse_args()
    
    for subject in args.subjects:
        process_subject(subject, args.data_dir, args.output_dir, args.max_frames, args.append)  
    print("\n" + "="*60)
    print("✓ All visualizations generated!")
    print(f"Review videos in: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()