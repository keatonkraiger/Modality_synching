"""
Pull / push the two synced trees between this repo and OneDrive.

    raw_data/   <->  onedrive:Datasets/OrdinaryMovements/raw_data
    Complete/   <->  onedrive:Datasets/OrdinaryMovements/Complete

work/ and qc/ are never synced: they are rebuilt from raw_data/.

Uses `rclone copy` only, so neither direction ever deletes a file on the other
side. Everything is a dry run unless you pass --yes.

    python scripts/sync_onedrive.py pull raw --subjects Kyle --kinds Pressure offsets
    python scripts/sync_onedrive.py pull complete --subjects Kyle --no-video --yes
    python scripts/sync_onedrive.py push raw --subjects Anu --yes
    python scripts/sync_onedrive.py check complete           # what differs (sizes only)

Remote root defaults to data_paths.REMOTE; override with $OM_REMOTE or --remote.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_paths import REMOTE, SYNCED  # noqa: E402

TREES = {"raw": "raw_data", "complete": "Complete"}

# raw_data/<Kind>/<Subject>/...   ("SMPL" nests subjects deeper, handled below)
RAW_KINDS = ["Video", "Pressure", "XCPs", "Model_output", "SMPL", "openpose", "offsets"]
ALWAYS_EXCLUDE = ["*.part", "*.zip.part", "__pycache__/**", "desktop.ini", "Thumbs.db", ".DS_Store"]


def build_filters(tree, subjects, kinds, no_video):
    rules = [f"- {p}" for p in ALWAYS_EXCLUDE]
    if no_video:
        rules.append("- *.mp4")
    if tree == "raw_data":
        rules.append("+ /*.json")  # faulty_sensors.json, subject_info.json
        for kind in kinds or RAW_KINDS:
            if subjects is None:
                rules.append(f"+ /{kind}/**")
            elif kind == "SMPL":
                rules += [f"+ /SMPL/**/{s}/**" for s in subjects]
            else:
                rules += [f"+ /{kind}/{s}/**" for s in subjects]
    else:
        rules.append("+ /*.json")  # om_sub_info.json
        rules += [f"+ /{s}/**" for s in subjects] if subjects else ["+ /**"]
    rules.append("- **")
    return rules


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direction", choices=["pull", "push", "check"])
    ap.add_argument("tree", choices=["raw", "complete", "all"])
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--kinds", nargs="+", default=None, choices=RAW_KINDS,
                    help="raw only: restrict to these raw_data/ subfolders")
    ap.add_argument("--no-video", action="store_true", help="skip *.mp4 (Complete/ without video is ~4 GB instead of 14)")
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--yes", action="store_true", help="actually transfer (default is --dry-run)")
    ap.add_argument("--transfers", type=int, default=8)
    a, extra = ap.parse_known_args()  # anything unrecognised goes straight to rclone

    if shutil.which("rclone") is None:
        sys.exit("rclone not found on PATH")
    if a.kinds and a.tree == "complete":
        ap.error("--kinds only applies to the raw tree")

    trees = ["raw", "complete"] if a.tree == "all" else [a.tree]
    for t in trees:
        name = TREES[t]
        local, remote = str(SYNCED[name]), f"{a.remote.rstrip('/')}/{name}"
        src, dst = (remote, local) if a.direction == "pull" else (local, remote)
        if a.direction == "check":
            cmd = ["rclone", "check", local, remote, "--size-only"]
        else:
            cmd = ["rclone", "copy", src, dst, "--transfers", str(a.transfers), "--progress"]
            if not a.yes:
                cmd.append("--dry-run")
        for r in build_filters(name, a.subjects, a.kinds, a.no_video):
            cmd += ["--filter", r]
        cmd += extra
        print(f"\n### {a.direction} {name}: {src if a.direction != 'check' else local} -> "
              f"{dst if a.direction != 'check' else remote}{'' if a.yes or a.direction == 'check' else '  (dry run)'}")
        print(" ".join(f"'{c}'" if " " in c or "*" in c else c for c in cmd))
        rc = subprocess.call(cmd)
        if rc not in (0,) and not (a.direction == "check" and rc == 1):
            sys.exit(rc)
    if a.direction != "check" and not a.yes:
        print("\nDry run only. Re-run with --yes to transfer.")


if __name__ == "__main__":
    main()
