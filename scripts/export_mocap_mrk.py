"""
Export uncropped Vicon markers for MoSh, straight from the raw Vicon JSON.

    raw_data/Model_output/<S>/<take>.json  ->  raw_data/MOCAP_MRK/<S>/<take>.npy
                                               raw_data/MOCAP_MRK/<S>/<take>.json  (sidecar)

The array is built exactly like Complete/<S>/<take>/MOCAP_MRK.npy (same 39-marker
order, same Rz(-90) world fix, (N, 39, 4) = [X, Y, Z, valid] in mm) but it is
NOT offset-cut to the pressure and NOT resampled: it keeps every Vicon frame at
the native rate, which is 100 Hz for some takes and 50 Hz for others (even
within one subject -- check the sidecar's "fps"). This lets MoSh run before
offsets / OpenPose / step 8 exist.

Mapping back to Complete/ (whose Vicon streams are always 50 fps):
    step = 2 if fps == 100 else 1
    complete_mrk == mrk[::step][-offset : -offset + frames]   (offset < 0)
i.e. decimate the same way combine_all.py does, then apply the take's offset
from raw_data/offsets/ (offsets are in 50 fps frames). Verified byte-identical
on Keaton OM2.

By default only takes with video are exported (same rule as get_model_output.py,
so Calibration is included when it has video — handy as MoSh's static pose).
--all_jsons exports every JSON in the folder.

    python scripts/export_mocap_mrk.py --subjects Keaton Anu
    python scripts/export_mocap_mrk.py --all [--overwrite]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_paths import RAW_MODEL_OUTPUT, RAW_MOCAP_MRK, RAW_VIDEO  # noqa: E402
import takes  # noqa: E402
from get_model_output import marker_order, apply_rz_minus90_transformation  # noqa: E402


def markers_from_json(data, apply_transform=True):
    """(N, 39, 4) marker array in get_model_output.marker_order, plus missing names."""
    by_name = {m["name"]: m for m in data.get("Markers", []) if m["name"] in marker_order}
    if not by_name:
        return None, list(marker_order)
    n = len(np.asarray(next(iter(by_name.values()))["valid"]))
    out = np.zeros((n, len(marker_order), 4))
    missing = []
    for i, name in enumerate(marker_order):
        m = by_name.get(name)
        if m is None:
            missing.append(name)
            continue
        traj = np.array(m["trajectories"]).T  # (N, 3)
        if apply_transform:
            traj = apply_rz_minus90_transformation(traj)
        out[:, i, :3] = traj
        out[:, i, 3] = np.array(m["valid"])
    return out, missing


def export_subject(subject, args):
    src = Path(args.model_output_dir) / subject
    dst = Path(args.out_dir) / subject
    jsons = sorted(src.glob("*.json"), key=lambda p: takes.sort_key(p.stem))
    if not jsons:
        print(f"[{subject}] no Vicon JSONs in {src}")
        return 0
    if not args.all_jsons:
        have_video = set(takes.video_takes(Path(args.video_dir) / subject))
        skipped = [p.stem for p in jsons if p.stem not in have_video]
        if skipped:
            print(f"[{subject}] skipping (no video): {', '.join(skipped)}")
        jsons = [p for p in jsons if p.stem in have_video]
    selection = takes.parse_selection(args.specific_oms)
    jsons = [p for p in jsons if takes.selected(p.stem, selection)]

    n_done = 0
    for j in jsons:
        out_npy = dst / f"{j.stem}.npy"
        if out_npy.exists() and not args.overwrite:
            print(f"[{subject}] {j.stem}: exists, skipping (--overwrite to redo)")
            continue
        with open(j) as f:
            data = json.load(f)
        mrk, missing = markers_from_json(data, apply_transform=not args.no_world_fix)
        if mrk is None:
            print(f"[{subject}] {j.stem}: no markers in JSON, skipping")
            continue
        fps = data.get("FrameRate")
        valid_frac = mrk[..., 3].mean(axis=0)
        dst.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, mrk)
        meta = {
            "subject": subject,
            "take": j.stem,
            "source": f"raw_data/Model_output/{subject}/{j.name}",
            "frames": int(mrk.shape[0]),
            "fps": fps,
            "units": "mm",
            "world_fix": "none" if args.no_world_fix else "Rz(-90): (X,Y,Z)->(Y,-X,Z)",
            "layout": "(frames, markers, [X, Y, Z, valid]); missing marker = all zeros",
            "marker_order": marker_order,
            "missing_markers": missing,
            "valid_fraction": {n: round(float(v), 4) for n, v in zip(marker_order, valid_frac)},
            "vicon_start_frame": data.get("StartFrame"),
            "cropped": False,
        }
        with open(dst / f"{j.stem}.json", "w") as f:
            json.dump(meta, f, indent=2)
        low = [n for n, v in zip(marker_order, valid_frac) if 0 < v < 0.5]
        note = ""
        if missing:
            note += f"  missing: {', '.join(missing)}"
        if low:
            note += f"  <50% valid: {', '.join(low)}"
        print(f"[{subject}] {j.stem}: {mrk.shape} @ {fps} Hz{note}")
        n_done += 1
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subjects", nargs="+")
    g.add_argument("--all", action="store_true", help="every subject in raw_data/Model_output/")
    ap.add_argument("--specific_oms", nargs="+", default=None,
                    help="only these takes: numbers and/or names (e.g. 1 3 Pick-up)")
    ap.add_argument("--all_jsons", action="store_true",
                    help="export every JSON, not just takes that have video")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_world_fix", action="store_true", help="skip the Rz(-90) transform")
    ap.add_argument("--model_output_dir", default=RAW_MODEL_OUTPUT)
    ap.add_argument("--video_dir", default=RAW_VIDEO)
    ap.add_argument("--out_dir", default=RAW_MOCAP_MRK)
    args = ap.parse_args()

    subjects = (sorted(d.name for d in Path(args.model_output_dir).iterdir() if d.is_dir())
                if args.all else args.subjects)
    total = sum(export_subject(s, args) for s in subjects)
    print(f"\nWrote {total} take(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
