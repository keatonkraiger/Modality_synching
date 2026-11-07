"""
Take names.

A take is identified by a name, which is also its file stem everywhere:
    raw_data/Video/<S>/<take>_V{1,2}.mp4, Pressure/<S>/<take>_{L,R}.csv,
    XCPs/<S>/<take>.xcp, Model_output/<S>/<take>.json, offsets/<S>/<take>.json
    work/.../<S>/<take>/,  Complete/<S>/<take>/

Names are usually the standard activities `OM<n>`, possibly with a suffix
(`OM5_T2` = second attempt), but any name is allowed (`Pick-up`, `Shelf`).
**A name counts as a take when the subject has video for it**, which is what
keeps calibration files (`cal01.xcp`, `Calibration.json`) out.

On the CLI, `--specific_oms` / `--oms` take any mix of bare numbers and names:
`--specific_oms 1 3 Pick-up OM5_T2` (a bare `3` means `OM3`). Matching is exact,
so `1` never selects `OM10`.
"""
import re
from pathlib import Path

_OM = re.compile(r"^OM(\d+)(.*)$")
VIDEO_RE = re.compile(r"^(?P<take>.+)_V(?P<view>[12])\.mp4$", re.IGNORECASE)

# Folders that sit next to take folders but are not takes.
NON_TAKE_DIRS = {"original_fps", "superseded", "OG_oms", "Cleaned", "data"}


def sort_key(take):
    """OM takes first in numeric order (OM2 < OM10, OM5 < OM5_T2), then the rest alphabetically."""
    m = _OM.match(take)
    return (0, int(m.group(1)), m.group(2).lower(), "") if m else (1, 0, "", take.lower())


def sort_takes(names):
    return sorted(names, key=sort_key)


def normalize(value):
    """CLI value -> take name: 3 / '3' -> 'OM3'; anything else unchanged."""
    s = str(value).strip()
    return f"OM{s}" if s.isdigit() else s


def parse_selection(values):
    """None or [] -> None (= all takes); otherwise the set of normalised names."""
    if not values or values == ["all"]:
        return None
    return {normalize(v) for v in values}


def selected(take, selection):
    return selection is None or take in selection


def take_dirs(parent):
    """Take sub-directories of `parent`, sorted."""
    parent = Path(parent)
    if not parent.is_dir():
        return []
    dirs = [d for d in parent.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_")) and d.name not in NON_TAKE_DIRS]
    return sorted(dirs, key=lambda d: sort_key(d.name))


def video_takes(subject_video_dir):
    """Takes that have at least one view in raw_data/Video/<S>/ (not recursive)."""
    d = Path(subject_video_dir)
    if not d.is_dir():
        return []
    names = {m.group("take") for f in d.iterdir() if f.is_file() and (m := VIDEO_RE.match(f.name))}
    return sort_takes(names)
