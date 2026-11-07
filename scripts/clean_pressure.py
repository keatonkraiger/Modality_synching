import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from scipy.ndimage import median_filter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import RAW_PRESSURE, WORK_PRESSURE  # noqa: E402

def load_tekscan_csv(filepath, skip_header_lines=31, rows=60, cols=21):
    frames = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    i = skip_header_lines
    while i < len(lines):
        if lines[i].strip().startswith("Frame"):
            i += 1
            matrix_lines = lines[i:i+rows]
            frame_data = []
            for line in matrix_lines:
                entries = line.strip().split(',')
                entries = entries[:cols] + ['B'] * (cols - len(entries))
                row = [np.nan if val == 'B' else float(val) for val in entries]
                frame_data.append(row)
            frames.append(frame_data)
            i += rows
        else:
            i += 1

    return np.array(frames)  # shape: (N, 60, 21)

def remove_isolated_pixels(frame, min_neighbors=2, kernel_size=3):
    """
    Remove isolated pixels that don't have enough active neighbors.
    
    Args:
        frame: 2D array of pressure values
        min_neighbors: minimum number of non-zero neighbors required
        kernel_size: size of neighborhood to check (3x3, 5x5, etc.)
    """
    # Create binary mask of active pixels (non-zero, non-NaN)
    active_mask = (frame > 0) & (~np.isnan(frame))
    
    # Count active neighbors for each pixel
    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 0  # Don't count the pixel itself
    neighbor_count = ndimage.convolve(active_mask.astype(int), kernel, mode='constant')
    
    # Keep pixels that have enough neighbors OR are part of a larger structure
    keep_mask = (neighbor_count >= min_neighbors) | (~active_mask)
    
    # Apply the mask
    cleaned_frame = np.copy(frame)
    cleaned_frame[~keep_mask] = 0
    
    return cleaned_frame

def remove_temporal_outliers(data, window_size=5, threshold_std=3):
    """
    Remove temporal outliers by looking at each pixel's behavior over time.
    
    Args:
        data: 3D array (frames, rows, cols)
        window_size: size of temporal window to analyze
        threshold_std: number of standard deviations for outlier detection
    """
    cleaned_data = np.copy(data)
    
    for i in range(data.shape[1]):  # rows
        for j in range(data.shape[2]):  # cols
            pixel_time_series = data[:, i, j]
            
            # Skip if all NaN or all zero
            if np.all(np.isnan(pixel_time_series)) or np.all(pixel_time_series == 0):
                continue
                
            # Use rolling window to detect outliers
            for frame_idx in range(len(pixel_time_series)):
                # Define window around current frame
                start_idx = max(0, frame_idx - window_size // 2)
                end_idx = min(len(pixel_time_series), frame_idx + window_size // 2 + 1)
                window = pixel_time_series[start_idx:end_idx]
                
                # Remove NaN and zeros for statistics
                valid_window = window[(~np.isnan(window)) & (window > 0)]
                
                if len(valid_window) < 3:  # Need at least 3 points for meaningful stats
                    continue
                    
                current_val = pixel_time_series[frame_idx]
                if np.isnan(current_val) or current_val == 0:
                    continue
                    
                # Check if current value is an outlier
                window_mean = np.mean(valid_window)
                window_std = np.std(valid_window)
                
                if window_std > 0 and abs(current_val - window_mean) > threshold_std * window_std:
                    # Additional check: is this value much higher than neighbors?
                    if current_val > window_mean + threshold_std * window_std:
                        cleaned_data[frame_idx, i, j] = 0
    
    return cleaned_data

def apply_median_filter_selective(data, kernel_size=3, high_threshold=500):
    """
    Apply median filter selectively to high-value pixels that might be noise.
    
    Args:
        data: 3D array (frames, rows, cols)
        kernel_size: size of median filter kernel
        high_threshold: threshold above which to apply median filtering
    """
    cleaned_data = np.copy(data)
    
    for frame_idx in range(data.shape[0]):
        frame = data[frame_idx]
        
        # Find pixels above threshold
        high_pixels = frame > high_threshold
        
        if np.any(high_pixels):
            # Apply median filter to the entire frame
            filtered_frame = median_filter(frame, size=kernel_size)
            
            # Only replace the high-value pixels with filtered values
            # But only if the filtered value is significantly lower
            replacement_mask = high_pixels & (filtered_frame < frame * 0.7)
            cleaned_data[frame_idx][replacement_mask] = filtered_frame[replacement_mask]
    
    return cleaned_data

def clip_pressure_values_enhanced(data, min_val, max_val, impossible_threshold=1500):
    clipped = np.copy(data)
    
    # Set impossible high values to zero instead of max_val
    clipped = np.where(clipped > impossible_threshold, 0, clipped)
    
    # Normal clipping for reasonable range
    clipped = np.where(clipped < min_val, 0, clipped)
    clipped = np.where(clipped > max_val, max_val, clipped)
    
    return clipped

def clean_pressure_data(data, min_val=10, max_val=862, impossible_threshold=1500,
                       remove_isolated=True, remove_temporal=True, apply_median=True):
    print("  → Starting data cleaning pipeline...")
    
    # Step 1: Enhanced clipping (sets impossible values to 0)
    cleaned = clip_pressure_values_enhanced(data, min_val, max_val, impossible_threshold)
    impossible_count = np.sum(data > impossible_threshold)
    if impossible_count > 0:
        print(f"    • Set {impossible_count} impossible readings (>{impossible_threshold}) to zero")

    
    return cleaned

def save_pressure_video(data, output_path, fps=100):
    N = data.shape[0]
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='black')
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    def combine_frames(frame):
        left = frame[:, :, 0]
        right = frame[:, :, 1]
        combined = np.ma.masked_invalid(np.hstack((left, right)))
        return combined

    initial_frame = combine_frames(data[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(initial_frame, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(10.5, -2, 'Left', ha='center', va='center', fontsize=10)
    ax.text(31.5, -2, 'Right', ha='center', va='center', fontsize=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pressure")

    def update(frame_idx):
        combined = combine_frames(data[frame_idx])
        im.set_data(combined)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=N, blit=True)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    print(f"  → Video saved to: {output_path}")

from pathlib import Path
def process_all_enhanced(data_dir, save_dir=None, min_val=10, max_val=862, 
                        impossible_threshold=1500, target_fps=50):
    """Enhanced processing with comprehensive cleaning."""
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    files = os.listdir(data_dir)
    processed = []

    # A take is any <take>_L.csv or <take>_R.csv (take names need not start with OM).
    bases = sorted({f[:-6] for f in files if f.endswith(("_L.csv", "_R.csv"))})

    for base in bases:
        left_path = os.path.join(data_dir, f"{base}_L.csv")
        right_path = os.path.join(data_dir, f"{base}_R.csv")

        print(f"[✓] Processing: {base}")
        left = load_tekscan_csv(left_path) if os.path.exists(left_path) else None
        right = load_tekscan_csv(right_path) if os.path.exists(right_path) else None

        # A foot that was not recorded becomes all-NaN ("no sensor anywhere"),
        # same shape as the recorded one, so every take stays (N, 60, 21, 2).
        for side, arr in (("left", left), ("right", right)):
            if arr is None:
                print(f"  [!] No {side} foot CSV for {base}: {side} foot will be all NaN")
        if left is None:
            left = np.full_like(right, np.nan)
        if right is None:
            right = np.full_like(left, np.nan)

        min_frames = min(len(left), len(right))
        
        # Apply enhanced cleaning to each foot separately
        print("  → Cleaning left foot data...")
        left_cleaned = clean_pressure_data(left[:min_frames], min_val, max_val, impossible_threshold)
        
        print("  → Cleaning right foot data...")
        right_cleaned = clean_pressure_data(right[:min_frames], min_val, max_val, impossible_threshold)
        
        combined = np.stack([left_cleaned, right_cleaned], axis=-1)  # Shape: (N, 60, 21, 2)
        
        # Downsample to the target fps, we assume its recorded at 50 fps.
        combined = combined[::int(100/target_fps)]
        processed.append((base, combined))

        if save_dir:
            save_path = os.path.join(save_dir, f"{base}/Original_Pressure.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, combined)
            print(f"  → Saved to: {save_path}")

            video_path = os.path.join(save_dir, f"{base}/Original_Pressure.mp4")
            print(f"  → Generating video: {video_path}")
            save_pressure_video(combined, video_path, fps=target_fps)

    return processed

# Example usage with enhanced cleaning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pressure data")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject name; sets --data_dir raw_data/Pressure/<S> and --save_dir work/Pressure/<S>")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to the save directory")
    args = parser.parse_args()
    if args.subject:
        args.data_dir = args.data_dir or os.path.join(RAW_PRESSURE, args.subject)
        args.save_dir = args.save_dir or os.path.join(WORK_PRESSURE, args.subject)
    if not args.data_dir or not args.save_dir:
        parser.error("give --subject, or both --data_dir and --save_dir")

    data_dir = args.data_dir
    save_dir = args.save_dir

    # Process with enhanced cleaning
    target_fps = 50
    all_data = process_all_enhanced(data_dir, save_dir,
                                   min_val=10, max_val=862,
                                   impossible_threshold=1500, target_fps=target_fps)