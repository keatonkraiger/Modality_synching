import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import WORK_PRESSURE  # noqa: E402

# ----------------- Helper Functions -----------------

def neighbor_means_3x3(foot_data):
    """3×3 neighbor mean per frame/pixel, ignoring NaNs. foot_data: (T,H,W)."""
    T, H, W = foot_data.shape
    neigh_sum = np.zeros_like(foot_data, dtype=float)
    neigh_cnt = np.zeros_like(foot_data, dtype=float)

    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            if di == 0 and dj == 0: 
                continue
            s = np.full_like(foot_data, np.nan)
            ii0, ii1 = max(0,di), H + min(0,di)
            jj0, jj1 = max(0,dj), W + min(0,dj)
            s[:, ii0:ii1, jj0:jj1] = foot_data[:, ii0-di:ii1-di, jj0-dj:jj1-dj]
            valid = ~np.isnan(s)
            neigh_sum += np.where(valid, s, 0.0)
            neigh_cnt += valid.astype(float)

    with np.errstate(invalid='ignore', divide='ignore'):
        neigh_mean = neigh_sum / neigh_cnt
    neigh_mean[neigh_cnt == 0] = np.nan
    return neigh_mean


def detect_faulty_sensors(pressure, ratio_thresh=3.0, min_frac=0.30, thresh=100.0):
    """
    Detect faulty sensors in pressure data.
    
    - pressure: (T,H,W,2), NaN = no sensor, 0 = valid inactive
    - Flags sensors if value > ratio_thresh × neighbors
      in > min_frac fraction of active frames,
      and if value >= thresh.
    - Neighbors with mean == 0 are excluded from ratio checks.
    """
    T, H, W, F = pressure.shape
    faulty = np.zeros((H, W, F), dtype=bool)

    for foot in range(F):
        s = pressure[..., foot]                # (T,H,W)
        nm = neighbor_means_3x3(s)             # (T,H,W)

        # Exclude neighbors that are zero (nm must be > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(nm > 0, s / nm, np.nan)

        active = (~np.isnan(ratio)) & (nm > 0)
        cond = (ratio > ratio_thresh)
        if thresh is not None:
            cond &= (s >= thresh)

        num = np.nansum(cond, axis=0)
        den = np.nansum(active, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(den > 0, num / den, 0.0)

        faulty[..., foot] = frac > min_frac

    return faulty


def plot_first_frame_with_faulty(pressure, faulty_mask, title="Faulty Sensor Frames"):
    """
    For each faulty sensor, find the first frame where it's active and plot it with a red outline.
    """
    T, H, W, F = pressure.shape
    footnames = ["Left", "Right"]

    # Iterate over each foot
    for f in range(F):
        fy, fx = np.where(faulty_mask[:, :, f])
        for y, x in zip(fy, fx):
            # Find the first frame where this sensor is non-NaN and >0
            sensor_series = pressure[:, y, x, f]
            valid_frames = np.where(~np.isnan(sensor_series) & (sensor_series > 0))[0]
            if len(valid_frames) == 0:
                continue
            t0 = valid_frames[0]

            # Plot this frame
            frame = pressure[t0]
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            for ff in range(F):
                im = axs[ff].imshow(frame[:, :, ff], origin="upper")
                axs[ff].set_title(f"{footnames[ff]} — Frame {t0}")
                plt.colorbar(im, ax=axs[ff])

                # Only outline this faulty sensor on its corresponding foot
                if ff == f:
                    axs[ff].add_patch(Rectangle((x-0.5, y-0.5), 1, 1,
                                                fill=False, edgecolor="red", lw=2))
            plt.suptitle(f"Faulty sensor at (y={y}, x={x}), foot={footnames[f]}")
            plt.tight_layout()
            plt.show()


def correct_faulty_sensors(pressure, faulty_mask):
    """Replace faulty sensors with neighbor mean values."""
    T, H, W, F = pressure.shape
    corrected = pressure.copy()
    for foot in range(F):
        s = corrected[..., foot]                # (T,H,W)
        nm = neighbor_means_3x3(s)              # (T,H,W)
        for y in range(H):
            for x in range(W):
                if faulty_mask[y, x, foot]:
                    s[:, y, x] = nm[:, y, x]
        corrected[..., foot] = s
    return corrected


def save_pressure_video(pressure, save_path, n_frames=200, fps=30):
    """Save MP4 of corrected pressure data (side-by-side feet, tight spacing, same colorbar scale)."""
    T, H, W, F = pressure.shape
    vmax = np.nanmax(pressure)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    footnames = ["Left", "Right"]

    ims = []
    for f in range(F):
        im = axs[f].imshow(pressure[0,:,:,f], origin='upper', vmin=0, vmax=vmax)
        axs[f].set_title(footnames[f])
        plt.colorbar(im, ax=axs[f])
        ims.append(im)

    plt.tight_layout()

    def update(t):
        for f in range(F):
            ims[f].set_data(pressure[t,:,:,f])
        return ims

    ani = animation.FuncAnimation(fig, update, frames=min(n_frames,T), interval=1000/fps, blit=False)
    ani.save(save_path, writer="ffmpeg", fps=fps)
    plt.close(fig)


# ----------------- Main Runner -----------------

thresh_dict = {
    'Keaton':{
        'OM2': 90,
        'OM3': 50
    }
}
def process_base_dir(base_dir, subject, remove_faulty=True, make_video=True, n_frames=200):
    for dir in os.listdir(base_dir):
        path = os.path.join(base_dir, dir, "Original_Pressure.npy")
        if not os.path.exists(path):
            continue

        print(f"Processing {path}")
        pressure = np.load(path)
        plot=True
        if dir in thresh_dict[subject]:
            plot=True
            thresh = thresh_dict[subject][dir]
        else:
            thresh = 120
        
        faulty_mask = detect_faulty_sensors(pressure, thresh=thresh)

        plot=False 
        if faulty_mask.sum() > 0:
            print(f"Thresh {thresh} - feaulty sensors found:", np.argwhere(faulty_mask))
            if plot:
                plot_first_frame_with_faulty(pressure, faulty_mask, title="First Faulty Frame")

            if remove_faulty:
                pressure = correct_faulty_sensors(pressure, faulty_mask)
                print("Corrected faulty sensors.")

            if make_video:
                save_path = os.path.join(base_dir, dir, "pressure.mp4")
                save_pressure_video(pressure, save_path, n_frames=n_frames)
                print(f"Saved video to {save_path}")
               
        else:
            print("No faulty sensors found.")
            

import argparse
# ----------------- Example Usage -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and correct faulty pressure sensors in pressure mat data.")
    parser.add_argument("--data_dir", default=WORK_PRESSURE,
                    help="Directory containing pressure data")
    parser.add_argument('--subject', type=str, required=True, help="Subject name (e.g., 'Keaton')")
    parser.add_argument('--remove_faulty', action='store_true', help="Whether to remove faulty sensors")
    parser.add_argument('--no_remove_faulty', action='store_false', dest='remove_faulty', help="Do not remove faulty sensors")
    parser.add_argument('--make_video', action='store_true', help="Whether to make videos")
    parser.add_argument('--json_file', type=str, default='post_processing/sensor_configs.json', help="Path to sensor config JSON")
    args = parser.parse_args()  
  
    data_dir = os.path.join(args.data_dir, args.subject)
    process_base_dir(data_dir, subject=args.subject, remove_faulty=args.remove_faulty,
                        make_video=args.make_video, n_frames=1000)

