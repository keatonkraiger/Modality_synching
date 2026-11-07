#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive faulty sensor annotator.
Stores all annotations in one global file: raw_data/faulty_sensors.json (data_paths.FAULTY_SENSORS)

Usage:
    python post_processing/2_find_faulty_sensors.py --subject Kyle --om OM1 --frame 221
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import WORK_PRESSURE, FAULTY_SENSORS  # noqa: E402

plt.rcParams["toolbar"] = "None"  # cleaner GUI window


# -------------------- Utilities --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def merge_faulty(existing, new):
    """Merge two dicts of faulty sensors {left: [[y,x],...], right: [[y,x],...]}."""
    merged = {}
    for side in ["left", "right"]:
        old = existing.get(side, [])
        new_points = new.get(side, [])
        all_pts = {tuple(pt) for pt in old} | {tuple(pt) for pt in new_points}
        merged[side] = [list(pt) for pt in sorted(all_pts)]
    return merged


# -------------------- Core Functions --------------------

def load_pressure(subject, om, base_dir=WORK_PRESSURE):
    path = os.path.join(base_dir, subject, om, "Original_Pressure.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pressure file not found: {path}")
    return np.load(path), path


def annotate_faulty_sensors(pressure, frame_idx, subject, om, existing_faulty=None, show_prev=True, vmax=None):
    """
    Interactive click-based faulty sensor annotator.
    Left click = toggle faulty sensor.
    'q' = quit and save.
    """
    H, W = pressure.shape[1:3]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title(f"{subject} — {om} — Frame {frame_idx}")

    footnames = ["Left", "Right"]
    images, faulty = [], [set(), set()]

    # Determine color scale
    if vmax is None:
        vmax = np.nanpercentile(pressure, 99)
        print(f"Auto-selected vmax={vmax:.2f}")
    else:
        print(f"Using provided vmax={vmax:.2f}")

    frame = pressure[frame_idx]

    for i in range(2):
        im = axs[i].imshow(frame[:, :, i], origin="upper", vmin=0, vmax=vmax)
        axs[i].set_title(f"{footnames[i]} — Frame {frame_idx}")
        plt.colorbar(im, ax=axs[i])
        axs[i].set_xticks([]); axs[i].set_yticks([])
        images.append(im)

    rects = [[], []]

    # Load previously saved faulty sensors (if requested)
    if show_prev and existing_faulty:
        for f, side in enumerate(["left", "right"]):
            for (y, x) in [tuple(pt) for pt in existing_faulty.get(side, [])]:
                faulty[f].add((y, x))

    def redraw():
        for f in range(2):
            for r in rects[f]:
                r.remove()
            rects[f].clear()
            for (y, x) in faulty[f]:
                r = Rectangle((x - 0.5, y - 0.5), 1, 1,
                              fill=False, edgecolor="red", lw=1.5)
                axs[f].add_patch(r)
                rects[f].append(r)
        fig.canvas.draw_idle()

    def onclick(event):
        for f in range(2):
            if event.inaxes == axs[f]:
                if event.xdata is None or event.ydata is None:
                    return
                x, y = int(round(event.xdata)), int(round(event.ydata))
                key = (y, x)
                if key in faulty[f]:
                    faulty[f].remove(key)
                else:
                    faulty[f].add(key)
                redraw()

    def onkey(event):
        if event.key.lower() == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    redraw()
    plt.tight_layout()
    plt.show()

    return {
        "left": sorted(list(map(list, faulty[0]))),
        "right": sorted(list(map(list, faulty[1]))),
    }


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive faulty sensor annotator.")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--om", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--base_dir", type=str, default=WORK_PRESSURE)
    parser.add_argument("--labels", type=str, default=FAULTY_SENSORS, help="faulty-sensor JSON to read and update")
    parser.add_argument("--no-show-prev", dest="show_prev", action="store_false",
                        help="Disable showing previously labeled sensors in red.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Optional maximum value for colorbar scaling.")
    parser.set_defaults(show_prev=True)
    args = parser.parse_args()

    # Load pressure data
    pressure, path = load_pressure(args.subject, args.om, args.base_dir)
    print(f"Loaded {path}, shape={pressure.shape}")
    print(f"Displaying frame {args.frame}. Click sensors to toggle faulty. Press 'q' to save and quit.")

    # Load existing JSON
    json_path = str(args.labels)
    data = load_json(json_path)

    # ------------------------------------------------------------------
    # Build union of all faulty sensors for this subject across all OMs
    # ------------------------------------------------------------------
    existing_faulty = {"left": [], "right": []}
    if args.show_prev and args.subject in data:
        subj_info = data[args.subject]
        union_left, union_right = set(), set()
        for om_name, om_faults in subj_info.items():
            for y, x in om_faults.get("left", []):
                union_left.add((y, x))
            for y, x in om_faults.get("right", []):
                union_right.add((y, x))
        existing_faulty = {
            "left": [list(pt) for pt in sorted(union_left)],
            "right": [list(pt) for pt in sorted(union_right)],
        }
        print(f"Found {len(union_left)} left + {len(union_right)} right previously labeled faulty sensors.")

    # ------------------------------------------------------------------
    # Run interactive annotation
    # ------------------------------------------------------------------
    new_faulty = annotate_faulty_sensors(
        pressure,
        args.frame,
        args.subject,
        args.om,
        existing_faulty=existing_faulty if args.show_prev else {},
        show_prev=args.show_prev,
        vmax=args.vmax
    )

    # ------------------------------------------------------------------
    # Merge and save
    # ------------------------------------------------------------------
    subj_dict = data.get(args.subject, {})
    om_dict = subj_dict.get(args.om, {"left": [], "right": []})
    merged = merge_faulty(om_dict, new_faulty)
    subj_dict[args.om] = merged
    data[args.subject] = subj_dict

    save_json(json_path, data)
    print(f"Saved annotations → {json_path}")


if __name__ == "__main__":
    main()
