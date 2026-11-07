"""
OpenPose JSONs are ~7k tiny files per take, which OneDrive handles badly, so
raw_data/ keeps one zip per take and the pipeline works from an unzipped copy:

    raw_data/openpose/<S>/OM<i>.zip   ->   work/openpose/<S>/OM<i>/OM<i>_V{1,2}_jsons/*.json

Zip members are stored relative to the OM folder (`OM<i>_V1_jsons/...`).

    python scripts/openpose_zip.py pack   --src <dir of unzipped subjects> [--subjects ...]
    python scripts/openpose_zip.py unpack [--subjects ...]
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_paths import RAW_OPENPOSE, WORK_OPENPOSE  # noqa: E402

OM_RE = re.compile(r"^OM(\d+)$")


def _om_ok(om, oms):
    """`oms` may hold ints (1) or names ('OM1'); None means all."""
    if oms is None:
        return True
    n = int(OM_RE.match(om).group(1))
    return n in oms or om in oms


def pack_take(om_dir, zip_path):
    om_dir, zip_path = Path(om_dir), Path(zip_path)
    files = sorted(p for p in om_dir.rglob("*") if p.is_file())
    if not files:
        return 0
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, f.relative_to(om_dir).as_posix())
    tmp.replace(zip_path)  # never leave a half-written zip under the real name
    return len(files)


def pack_subject(src_root, subject, dst_root=RAW_OPENPOSE, oms=None, overwrite=False):
    src = Path(src_root) / subject
    if not src.is_dir():
        print(f"  [skip] no {src}")
        return
    for om_dir in sorted(d for d in src.iterdir() if d.is_dir() and OM_RE.match(d.name)):
        if not _om_ok(om_dir.name, oms):
            continue
        zp = Path(dst_root) / subject / f"{om_dir.name}.zip"
        if zp.exists() and not overwrite:
            print(f"  {subject}/{om_dir.name}: {zp.name} exists, skipping")
            continue
        n = pack_take(om_dir, zp)
        print(f"  {subject}/{om_dir.name}: {n} files -> {zp}")


def unpack_subject(subject, src_root=RAW_OPENPOSE, dst_root=WORK_OPENPOSE, oms=None, overwrite=False):
    """Extract every take zip whose target folder is missing (or all, with overwrite)."""
    src = Path(src_root) / subject
    if not src.is_dir():
        return
    for zp in sorted(src.glob("OM*.zip")):
        om = zp.stem
        if not OM_RE.match(om) or not _om_ok(om, oms):
            continue
        dst = Path(dst_root) / subject / om
        if dst.is_dir() and any(dst.iterdir()) and not overwrite:
            continue
        print(f"  unzipping {zp} -> {dst}")
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dst)


def _subjects(root, given):
    if given:
        return given
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir()) if Path(root).is_dir() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pack", "unpack"])
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--oms", nargs="+", default=None, help="e.g. OM1 OM3")
    ap.add_argument("--src", default=None, help="pack: unzipped root (default work/openpose); unpack: zip root")
    ap.add_argument("--dst", default=None, help="pack: zip root (default raw_data/openpose); unpack: unzipped root")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.action == "pack":
        src, dst = a.src or WORK_OPENPOSE, a.dst or RAW_OPENPOSE
        for s in _subjects(src, a.subjects):
            pack_subject(src, s, dst, a.oms, a.overwrite)
    else:
        src, dst = a.src or RAW_OPENPOSE, a.dst or WORK_OPENPOSE
        for s in _subjects(src, a.subjects):
            unpack_subject(s, src, dst, a.oms, a.overwrite)


if __name__ == "__main__":
    main()
