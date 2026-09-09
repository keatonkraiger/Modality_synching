#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaning pipeline for faulty sensors across all OMs.

Reads faulty_sensors.json annotations and applies a 3×3 neighborhood smoothing
to all faulty sensors (union across all OMs) in Complete/<Subject>/OM*/Pressure.npy.

Optionally:
  • Creates cleaned videos for each OM
  • Overwrites cleaned arrays inside Complete/<Subject>/OMx/Pressure.npy
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndi

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def global_smooth_3x3(data, iterations=1):
    """
    Reapply the same 3×3 neighbor mean logic globally across the foot.
    Essentially a mild local smoothing pass (safe for valid data).
    """
    smoothed = data.copy()
    for _ in range(iterations):
        nm = neighbor_means_3x3(smoothed)  # reuse same local mean logic
        # Replace NaNs or zero values; otherwise blend original and neighbor mean
        smoothed = np.where(
            np.isnan(smoothed) | (smoothed == 0),
            nm,
            (smoothed + nm) / 2  # soft average for valid pixels
        )
    return smoothed


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def neighbor_means_3x3(foot_data, faulty_mask=None):
    """
    3×3 neighbor mean per frame/pixel, ignoring NaNs and (optionally) faulty pixels.
    Supports both (T,H,W) and (H,W) inputs.
    Operates purely spatially (no temporal smoothing).
    """
    # Handle single-frame (H,W)
    if foot_data.ndim == 2:
        foot_data = foot_data[None, ...]
        single_frame = True
    else:
        single_frame = False

    T, H, W = foot_data.shape
    neigh_sum = np.zeros_like(foot_data, dtype=float)
    neigh_cnt = np.zeros_like(foot_data, dtype=float)

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue

            s = np.full_like(foot_data, np.nan)
            ii0, ii1 = max(0, di), H + min(0, di)
            jj0, jj1 = max(0, dj), W + min(0, dj)
            s[:, ii0:ii1, jj0:jj1] = foot_data[:, ii0-di:ii1-di, jj0-dj:jj1-dj]

            valid = ~np.isnan(s)
            if faulty_mask is not None:
                fmask = np.full((H, W), False)
                fmask[ii0:ii1, jj0:jj1] = faulty_mask[ii0-di:ii1-di, jj0-dj:jj1-dj]
                valid &= ~fmask[None, :, :]

            neigh_sum += np.where(valid, s, 0.0)
            neigh_cnt += valid.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        neigh_mean = neigh_sum / neigh_cnt
    neigh_mean[neigh_cnt == 0] = np.nan

    # Return same shape as input
    if single_frame:
        neigh_mean = neigh_mean[0]
    return neigh_mean
def correct_faulty_sensors(pressure, faulty_mask, apply_global_smooth=False):
    """
    Replace faulty sensors with neighbor means, excluding all faulty pixels.
    Optionally apply a global 3×3 smoothing pass to reduce strip artifacts.
    """
    T, H, W, F = pressure.shape
    corrected = pressure.copy()

    for foot in range(F):
        s = corrected[..., foot]
        nm = neighbor_means_3x3(s, faulty_mask=faulty_mask[..., foot])

        # Step 1: repair explicitly faulty sensors
        for y in range(H):
            for x in range(W):
                if faulty_mask[y, x, foot]:
                    s[:, y, x] = nm[:, y, x]

        # Step 2: optional global smoothing pass
        if apply_global_smooth:
            print(f"    Applying global 3×3 smoothing for foot {foot} ...")
            for t in range(T):
                s[t] = global_smooth_3x3(s[t])

        corrected[..., foot] = s

    return corrected



def save_pressure_video(data, output_path, fps=50, max_frames=None):
    """Create side-by-side visualization of left and right foot pressure."""
    N = data.shape[0]
    if max_frames is not None:
        N = min(N, max_frames)
        data = data[:N]

    cmap = plt.get_cmap("jet")
    cmap.set_bad(color="black")

    valid_data = data[~np.isnan(data) & (data > 0)]
    if len(valid_data) > 0:
        vmin = 0
        vmax = np.percentile(valid_data, 99)
        actual_max = np.nanmax(data)
        print(f"    Data range: 0–{actual_max:.1f} (display capped at 99th percentile: {vmax:.1f})")
    else:
        vmin, vmax = 0, 1
        print("    Warning: No valid pressure data found")

    def combine_frames(frame):
        left = frame[:, :, 0]
        right = frame[:, :, 1]
        combined = np.ma.masked_invalid(np.hstack((left, right)))
        return combined

    initial_frame = combine_frames(data[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(initial_frame, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(10.5, -2, "Left", ha="center", va="center", fontsize=10)
    ax.text(31.5, -2, "Right", ha="center", va="center", fontsize=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pressure")

    frame_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                         verticalalignment="top", color="white", fontsize=10,
                         bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

    def update(frame_idx):
        combined = combine_frames(data[frame_idx])
        im.set_data(combined)
        frame_text.set_text(f"Frame {frame_idx}/{N-1}")
        return [im, frame_text]

    ani = animation.FuncAnimation(fig, update, frames=N, blit=True, interval=1000/fps)
    ani.save(output_path, writer="ffmpeg", fps=fps)
    plt.close()
    print(f"    → Video saved to: {output_path}")


# ---------------------------------------------------------------------
# Cleaning Logic
# ---------------------------------------------------------------------

def build_union_faulty_mask(subject, faulty_data, H, W):
    """Combine all faulty coordinates across OMs for this subject."""
    faulty_mask = np.zeros((H, W, 2), dtype=bool)
    subj_info = faulty_data.get(subject, {})

    for om_name, om_faults in subj_info.items():
        for side_idx, side_name in enumerate(["left", "right"]):
            for y, x in om_faults.get(side_name, []):
                if 0 <= y < H and 0 <= x < W:
                    faulty_mask[y, x, side_idx] = True
    return faulty_mask


def run_cleaning_all(subject, create_video=False, save_cleaned=False, fps=50, max_frames=None, apply_global_smooth=False):
    """
    Run the smoothing-based correction for all OMs of a subject
    using the union of all faulty sensors.
    """
    json_path = os.path.join("post_processing_output", "faulty_sensors.json")
    faulty_data = load_json(json_path)
    if subject not in faulty_data:
        print(f"[!] No faulty sensor data found for {subject} in {json_path}")
        return

    subj_dir = os.path.join("Complete", subject)
    if not os.path.exists(subj_dir):
        print(f"[!] Subject directory not found: {subj_dir}")
        return

    om_dirs = sorted([d for d in os.listdir(subj_dir) if d.startswith("OM")])
    if not om_dirs:
        print(f"[!] No OM directories found in {subj_dir}")
        return

    output_dir = os.path.join("post_processing_output", subject, "cleaned_pressure_viz")
    ensure_dir(output_dir)

    # Get representative shape from the first OM
    first_om_path = os.path.join(subj_dir, om_dirs[0], "Pressure.npy")
    sample = np.load(first_om_path)
    H, W = sample.shape[1:3]

    # Build global faulty mask (union of all OMs)
    union_mask = build_union_faulty_mask(subject, faulty_data, H, W)
    print(f"[{subject}] Union faulty sensors: {union_mask.sum()} total")

    # Process each OM
    for om in om_dirs:
        path = os.path.join(subj_dir, om, "Pressure.npy")
        if not os.path.exists(path):
            print(f"  Skipping missing {path}")
            continue

        print(f"Processing {om} ...")
        pressure = np.load(path)
        cleaned = correct_faulty_sensors(pressure, union_mask, apply_global_smooth=apply_global_smooth)
        print(f"  → Cleaned {union_mask.sum()} faulty pixels")

        if save_cleaned:
            np.save(path, cleaned)
            print(f"  Overwrote cleaned data in {path}")

        if create_video:
            video_path = os.path.join(output_dir, f"{om}_pressure.mp4")
            save_pressure_video(cleaned, video_path, fps=fps, max_frames=max_frames)
        else:
            print("  Skipping video generation.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean all OMs using union of faulty sensors.")
    parser.add_argument("--subject", type=str, required=True, help="Subject name (e.g., 'Kyle')")
    parser.add_argument("--create-video", action="store_true", help="Create cleaned visualization videos.")
    parser.add_argument("--save-cleaned", action="store_true",
                        help="Overwrite cleaned Pressure.npy files inside Complete/<Subject>/OMx/.")
    parser.add_argument("--fps", type=int, default=50, help="FPS for video output.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per video.")
    parser.add_argument("--apply-global-smooth", action="store_true", help="Apply global 3x3 smoothing to handle strip artifacts.")
    args = parser.parse_args()

    run_cleaning_all(args.subject, create_video=args.create_video,
                     save_cleaned=args.save_cleaned, fps=args.fps, max_frames=args.max_frames,
                     apply_global_smooth=args.apply_global_smooth)

if __name__ == "__main__":
    main()
