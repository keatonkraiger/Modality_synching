#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Side-by-side comparison video: RGB view | repaired pressure | current pressure.

Renders one frame at a time to JPEGs in a temporary directory and muxes them with
ffmpeg, so nothing larger than a single frame is ever held in memory.  All three
panels are drawn at the same pixel height and each keeps its own aspect ratio.
Frames stage under the system temp dir by default (see --tmp_dir).

    python viz/compare_pressure_video.py --subject Varad --om OM11
    python viz/compare_pressure_video.py --subject Varad --om OM11 \
        --start 1200 --max_frames 500 --out /tmp/preview.mp4

The third panel defaults to `Complete/<S>/OM<i>/Pressure.npy` -- note that this is
itself the old stage-D3 output for any cell that was hand-labelled, not the raw
recording.  Use `--old_source original` to compare against the pristine
`untrimmed/.../Original_Pressure.npy` (offset-cut the same way combine_all.py cuts).
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path

import numpy as np
import cv2
import matplotlib

# ------------------------------------------------------------------ appearance
BG = (24, 24, 27)              # BGR page background
INK = (235, 235, 238)          # primary text
INK_MUTED = (150, 150, 158)    # secondary text
NO_SENSOR = BG                 # no sensor at this cell -> merge into the page
ZERO_LOAD = (44, 41, 39)       # BGR: a real sensel reading exactly zero
CMAP_FLOOR = 0.18              # lift the ramp so the faintest load outranks ZERO_LOAD
COP_COLOUR = (240, 201, 76)    # BGR cyan-ish (#4cc9f0) marker, absent from magma
FAULTY_COLOUR = (73, 174, 237) # BGR amber (#edae49) outline on the "current" panel
GAP_CELLS = 2                  # blank columns between the two feet
FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_lut(name="magma"):
    rgb = matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3]
    return (rgb[:, ::-1] * 255).astype(np.uint8).reshape(256, 1, 3)


# ------------------------------------------------------------------------ data
def load_offset(offsets_dir, subject, om):
    p = Path(offsets_dir) / subject / f"{om}.json"
    if not p.exists():
        p = Path(offsets_dir) / subject / f"{om}_cfg.json"
    with open(p) as f:
        return int(json.load(f)["offset"])


def open_pressure(args, which):
    """Memory-mapped (T,60,21,2) array for 'new' or 'old'."""
    take = Path(args.complete_dir) / args.subject / args.om
    if which == "new":
        return np.load(take / args.new_name, mmap_mode="r"), 0
    if args.old_source == "complete":
        return np.load(take / "Pressure.npy", mmap_mode="r"), 0
    src = Path(args.untrimmed_dir) / "Pressure" / args.subject / args.om / "Original_Pressure.npy"
    return np.load(src, mmap_mode="r"), max(load_offset(args.offsets_dir, args.subject, args.om), 0)


def scan_vmax(arr, offset, n_frames, pct, stride):
    """99.x percentile of positive readings from a strided sample -- never loads it all."""
    idx = range(offset, offset + n_frames, max(stride, 1))
    vals = []
    for i in idx:
        f = np.asarray(arr[i])
        v = f[np.isfinite(f) & (f > 0)]
        if v.size:
            vals.append(v)
    if not vals:
        return 1.0
    return float(np.percentile(np.concatenate(vals), pct))


def cop_of(frame):
    """Per-foot (row, col) centroid; NaN when that foot carries no load."""
    out = np.full((2, 2), np.nan)
    for f in range(2):
        v = np.nan_to_num(frame[..., f])
        tot = v.sum()
        if tot <= 0:
            continue
        yy, xx = np.indices(v.shape)
        out[f] = ((v * yy).sum() / tot, (v * xx).sum() / tot)
    return out


# ---------------------------------------------------------------------- panels
def pressure_panel(frame, vmax, cell, lut, faulty=None, cop=None):
    """One (60,21,2) frame -> BGR image of height 60*cell, both feet side by side."""
    H, W, _ = frame.shape
    grid_w = W * 2 + GAP_CELLS
    small = np.full((H, grid_w), np.nan, dtype=np.float64)
    small[:, :W] = frame[..., 0]
    small[:, W + GAP_CELLS:] = frame[..., 1]

    scaled = np.clip(np.nan_to_num(small) / max(vmax, 1e-6), 0, 1)
    scaled = CMAP_FLOOR + (1.0 - CMAP_FLOOR) * scaled
    img = cv2.applyColorMap((scaled * 255).astype(np.uint8), lut)
    img[np.nan_to_num(small) <= 0] = ZERO_LOAD          # sensel present, no load
    img[np.isnan(small)] = NO_SENSOR                    # no sensel here at all
    img[:, W:W + GAP_CELLS] = BG

    big = cv2.resize(img, (grid_w * cell, H * cell), interpolation=cv2.INTER_NEAREST)

    def to_px(row, col, foot):
        x = int(round((col + (0 if foot == 0 else W + GAP_CELLS) + 0.5) * cell))
        return x, int(round((row + 0.5) * cell))

    if faulty is not None:
        for f in range(2):
            for y, x in np.argwhere(faulty[..., f]):
                x0, y0 = to_px(y - 0.5, x - 0.5, f)
                cv2.rectangle(big, (x0, y0), (x0 + cell, y0 + cell), FAULTY_COLOUR, 1, cv2.LINE_AA)

    if cop is not None:
        for f in range(2):
            if not np.isfinite(cop[f, 0]):
                continue
            x, y = to_px(cop[f, 0], cop[f, 1], f)
            r = max(cell // 2, 5)
            cv2.circle(big, (x, y), r, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(big, (x, y), r, COP_COLOUR, 2, cv2.LINE_AA)
            cv2.drawMarker(big, (x, y), COP_COLOUR, cv2.MARKER_CROSS, r, 1, cv2.LINE_AA)
    return big


def colourbar(height, vmax, lut, width=18):
    ramp = np.linspace(255, CMAP_FLOOR * 255, height).astype(np.uint8).reshape(height, 1)
    bar = cv2.applyColorMap(np.repeat(ramp, width, axis=1), lut)
    pad = np.full((height, width + 62, 3), BG, np.uint8)
    pad[:, :width] = bar
    cv2.putText(pad, f"{vmax:.0f}", (width + 6, 14), FONT, 0.42, INK_MUTED, 1, cv2.LINE_AA)
    cv2.putText(pad, "0", (width + 6, height - 5), FONT, 0.42, INK_MUTED, 1, cv2.LINE_AA)
    return pad


def fit_height(img, height):
    """Resize keeping aspect ratio; area for photos, nearest for anything blocky."""
    h, w = img.shape[:2]
    if h == height:
        return img
    interp = cv2.INTER_AREA if h > height else cv2.INTER_LINEAR
    return cv2.resize(img, (max(int(round(w * height / h)), 1), height), interpolation=interp)


def compose(panels, titles, height, header=46, footer=40):
    widths = [p.shape[1] for p in panels]
    total_w = sum(widths) + 24 * (len(panels) - 1)
    canvas = np.full((header + height + footer, total_w, 3), BG, np.uint8)
    x = 0
    for p, t in zip(panels, titles):
        canvas[header:header + height, x:x + p.shape[1]] = p
        if t:
            cv2.putText(canvas, t, (x + 2, header - 16), FONT, 0.62, INK, 1, cv2.LINE_AA)
        x += p.shape[1] + 24
    return canvas


# ------------------------------------------------------------------------ main
def render(args):
    take = Path(args.complete_dir) / args.subject / args.om
    video_path = take / f"Video_{args.view}.mp4"
    for p in (video_path, take / args.new_name):
        if not p.exists():
            sys.exit(f"missing {p}")

    new, new_off = open_pressure(args, "new")
    old, old_off = open_pressure(args, "old")
    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_total = min(n_video, new.shape[0] - new_off, old.shape[0] - old_off)
    start = max(args.start, 0)
    n = n_total - start if args.max_frames is None else min(args.max_frames, n_total - start)
    if n <= 0:
        sys.exit(f"nothing to render (video {n_video}, pressure {new.shape[0]}, start {start})")
    print(f"{args.subject}/{args.om}: {n} frames from {start} "
          f"(video {n_video}, new {new.shape[0]}, old {old.shape[0]})")

    faulty = None
    fp = Path(args.output_dir) / args.subject / "pressure_clean" / "faulty_mask.npy"
    if args.mark_cells and fp.exists():
        faulty = np.load(fp)
        print(f"  outlining {int(faulty.sum())} repaired cells on the '{args.old_label}' panel")

    lut = colour_lut(args.cmap)
    cell = max(args.height // new.shape[1], 1)
    panel_h = cell * new.shape[1]
    vmax = args.vmax or scan_vmax(new, new_off, min(n_total, 4000), args.vmax_pct, args.vmax_stride)
    print(f"  colour scale 0-{vmax:.0f} (shared by both pressure panels), cell {cell}px")

    if args.tmp_dir:
        tmp = Path(args.tmp_dir)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    else:
        # Default to the system temp dir, not next to the output: under WSL a
        # /mnt/... target is a Windows drive and writing thousands of small JPEGs
        # there is ~2x slower, and worse still with several renders in parallel.
        tmp = Path(tempfile.mkdtemp(prefix=f"ompress_{args.subject}_{args.om}_"))
    print(f"  staging frames in {tmp}")

    if start:
        print(f"  skipping to frame {start} ...", flush=True)
        for _ in range(start):
            if not cap.grab():
                sys.exit(f"video ended before start frame {start}")
    bar = colourbar(panel_h, vmax, lut)
    written = 0
    try:
        for k in range(n):
            ok, rgb = cap.read()
            if not ok:
                print(f"  video ended early at frame {start + k}")
                break
            i = start + k
            fn = np.asarray(new[new_off + i], dtype=np.float64)
            fo = np.asarray(old[old_off + i], dtype=np.float64)
            cn, co = cop_of(fn), cop_of(fo)

            panels = [fit_height(rgb, panel_h),
                      pressure_panel(fn, vmax, cell, lut, None, cn),
                      pressure_panel(fo, vmax, cell, lut, faulty, co),
                      bar]
            titles = [f"{args.view}  {args.subject}/{args.om}", args.new_label, args.old_label, ""]
            canvas = compose(panels, titles, panel_h)

            d = np.linalg.norm(cn - co, axis=1)
            parts = []
            for f, name in enumerate(("L", "R")):
                if np.isfinite(d[f]):
                    parts.append(f"{name} {d[f]:.2f}")
                elif np.isfinite(cn[f, 0]) != np.isfinite(co[f, 0]):
                    parts.append(f"{name} load only in '{args.old_label if np.isfinite(co[f,0]) else args.new_label}'")
                else:
                    parts.append(f"{name} unloaded")
            foot = (f"frame {i}/{n_total - 1}   t={i / args.fps:6.2f}s   "
                    f"CoP shift (sensels): {'   '.join(parts)}")
            cv2.putText(canvas, foot, (2, canvas.shape[0] - 14), FONT, 0.52, INK_MUTED, 1, cv2.LINE_AA)
            key = "ring = centre of pressure    grey = sensel reading zero    box = repaired cell"
            (kw, _), _ = cv2.getTextSize(key, FONT, 0.45, 1)
            cv2.putText(canvas, key, (max(canvas.shape[1] - kw - 4, 4), canvas.shape[0] - 14),
                        FONT, 0.45, INK_MUTED, 1, cv2.LINE_AA)

            if canvas.shape[0] % 2 or canvas.shape[1] % 2:
                canvas = canvas[:canvas.shape[0] // 2 * 2, :canvas.shape[1] // 2 * 2]
            cv2.imwrite(str(tmp / f"{k:06d}.jpg"), canvas,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            written += 1
            if written % 250 == 0:
                print(f"  {written}/{n} frames", flush=True)
    finally:
        cap.release()

    if written == 0:
        sys.exit("no frames written")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
           "-i", str(tmp / "%06d.jpg"), "-c:v", "libx264", "-preset", "medium",
           "-crf", str(args.crf), "-pix_fmt", "yuv420p", str(args.out)]
    print("  muxing:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not args.keep_frames:
        shutil.rmtree(tmp)
    else:
        print(f"  frames kept in {tmp}")
    print(f"  -> {args.out}  ({written} frames, {written / args.fps:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--om", required=True)
    ap.add_argument("--view", default="V1", choices=["V1", "V2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--height", type=int, default=720, help="panel height in px (all panels share it)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--vmax_pct", type=float, default=99.5)
    ap.add_argument("--vmax_stride", type=int, default=5)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--tmp_dir", default=None,
                    help="where to stage JPEG frames; default is a fresh dir under the system "
                         "temp dir (fastest under WSL). Point it elsewhere if temp is small.")
    ap.add_argument("--keep_frames", action="store_true")
    ap.add_argument("--no_mark_cells", dest="mark_cells", action="store_false")
    ap.add_argument("--new_name", default="Pressure_clean.npy")
    ap.add_argument("--new_label", default="repaired - Pressure_clean.npy")
    ap.add_argument("--old_label", default=None)
    ap.add_argument("--old_source", default="complete", choices=["complete", "original"])
    ap.add_argument("--complete_dir", default="Complete")
    ap.add_argument("--untrimmed_dir", default="untrimmed")
    ap.add_argument("--offsets_dir", default="raw_data/offsets")
    ap.add_argument("--output_dir", default="post_processing_output")
    args = ap.parse_args()
    if args.old_label is None:
        args.old_label = ("current - Pressure.npy" if args.old_source == "complete"
                          else "as recorded - Original_Pressure.npy")
    if args.out is None:
        args.out = str(Path(args.output_dir) / args.subject / "pressure_clean" / f"{args.om}_compare_{args.view}.mp4")
    render(args)


if __name__ == "__main__":
    main()
