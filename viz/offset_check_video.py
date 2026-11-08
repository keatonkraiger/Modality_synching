#!/usr/bin/env python3
"""
Watch a sync offset: V1 | V2 | pressure, exactly as syncher.py's "Test Offset Playback".

Uses the same inputs and the same start rule as the syncher, so nothing past
step 2 is needed (no OpenPose, no Complete/):
    raw_data/Video/<S>/<take>_V1.mp4, <take>_V2.mp4
    work/Pressure/<S>/<take>/Original_Pressure.mp4
    RGB starts at max(0, -offset), pressure at max(0, offset).

Offsets come from --offsets_dir (default raw_data/offsets, i.e. the saved
hand offsets). Point it at another folder of <S>/<take>.json files to render a
different set, and give it a --name so the outputs don't collide:

    python viz/offset_check_video.py --subject Anu
    python viz/offset_check_video.py --subject Anu --offsets_dir work/offset_compare/offsets_estimated --name estimated
    python viz/offset_check_video.py --subject Anu --oms OM3 --offset -52 --name try52

Output: qc/offset_check/<S>/<name>/<take>.mp4 (name defaults to "saved"),
first --seconds (30) of the synced overlap, at real-time speed by default.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_paths import OFFSETS, QC, RAW_VIDEO, WORK_PRESSURE  # noqa: E402
import takes  # noqa: E402

BG = (24, 24, 27)
INK = (235, 235, 238)
MUTED = (150, 150, 158)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render(args, take, offset, out):
    paths = [Path(RAW_VIDEO) / args.subject / f"{take}_V1.mp4",
             Path(RAW_VIDEO) / args.subject / f"{take}_V2.mp4",
             Path(WORK_PRESSURE) / args.subject / take / "Original_Pressure.mp4"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"{take}: missing {', '.join(missing)}, skipped")
        return
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    rgb_start, prs_start = max(0, -offset), max(0, offset)
    starts = [rgb_start, rgb_start, prs_start]
    avail = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) - s for c, s in zip(caps, starts))
    n = min(avail, int(round(args.seconds * 50)))
    if n <= 0:
        print(f"{take}: offset {offset} leaves no overlap, skipped")
        return
    for c, s in zip(caps, starts):
        c.set(cv2.CAP_PROP_POS_FRAMES, s)

    h, header, footer, gap = args.height, 34, 30, 12
    widths = [round(c.get(cv2.CAP_PROP_FRAME_WIDTH) * h / c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    W = sum(widths) + gap * (len(caps) - 1)
    H = header + h + footer
    W, H = W + W % 2, H + H % 2
    titles = ["V1", "V2", "Original_Pressure.mp4"]

    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(args.playback_fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    written = 0
    try:
        for k in range(n):
            frames = [c.read() for c in caps]
            if not all(ok for ok, _ in frames):
                break
            canvas = np.full((H, W, 3), BG, np.uint8)
            x = 0
            for (_, img), w, t in zip(frames, widths, titles):
                canvas[header:header + h, x:x + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                cv2.putText(canvas, t, (x + 4, header - 10), FONT, 0.6, INK, 1, cv2.LINE_AA)
                x += w + gap
            cv2.putText(canvas, f"{args.subject}/{take}   {args.name} offset {offset:+d}   "
                        f"video frame {rgb_start + k}   pressure frame {prs_start + k}   t={k / 50:5.2f}s",
                        (4, H - 10), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
            ff.stdin.write(canvas.tobytes())
            written += 1
    finally:
        ff.stdin.close()
        ff.wait()
        for c in caps:
            c.release()
    print(f"{take}: offset {offset:+d} -> {out}  ({written / 50:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--oms", nargs="+", default=None, help="takes (numbers or names); default all with video")
    ap.add_argument("--offsets_dir", default=OFFSETS, help="folder of <S>/<take>.json (default raw_data/offsets)")
    ap.add_argument("--offset", type=int, default=None, help="use this offset instead of a file (with --oms)")
    ap.add_argument("--name", default="saved", help="output subfolder / label for this set of offsets")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--height", type=int, default=420, help="panel height in px")
    ap.add_argument("--playback_fps", type=float, default=50, help="50 = real time, 25 = half speed")
    ap.add_argument("--out_dir", default=None, help="default qc/offset_check/<subject>/<name>/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or Path(QC) / "offset_check" / args.subject / args.name)
    sel = takes.parse_selection(args.oms)
    for t in takes.video_takes(Path(RAW_VIDEO) / args.subject):
        if not takes.selected(t, sel):
            continue
        if args.offset is not None:
            off = args.offset
        else:
            f = Path(args.offsets_dir) / args.subject / f"{t}.json"
            if not f.exists():
                print(f"{t}: no offset in {f.parent}, skipped")
                continue
            off = int(json.load(open(f))["offset"])
        render(args, t, off, out_dir / f"{t}.mp4")


if __name__ == "__main__":
    main()
