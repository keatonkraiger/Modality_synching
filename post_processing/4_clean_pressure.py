#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage D2 -- automatic pressure-sensel repair (replaces the hand-label-only D3).

Reads the pristine per-take arrays
    work/Pressure/<Subject>/OM<i>/Original_Pressure.npy
applies the same offset cut as scripts/combine_all.py, detects faulty sensels
per subject (statistics pooled over every take of that subject, because the
same physical insoles were worn for the whole session), repairs them, and writes

    Complete/<Subject>/OM<i>/Pressure_clean.npy      (new file; Pressure.npy is never touched)

plus a report under qc/pressure/<Subject>/pressure_clean/.

Fault classes (all detected automatically, all merged with raw_data/faulty_sensors.json):

  hot       reads >= --hot-gain times the mean of its *active* neighbours whenever
            it is loaded (median ratio over the pooled frames).  These are
            over-gain / saturating sensels and strips; they drag the CoP toward
            themselves.
  spurious  active while every valid neighbour reads zero, in >= --spur-frac of
            its active frames.  These are the "pressure while the foot is in the
            air" cells; they define the CoP by themselves in swing phase.
  dead      reads zero while its neighbours are loaded: either zero in
            >= --dead-zero-frac of the frames where both direct (H or V)
            neighbours exceed --bracket, or a median gain <= --weak-gain with
            >= 5 active neighbours.  Single dead cells and dead strips.

Repair:
  * faulty cells are filled by harmonic inpainting from the good cells around
    them (iterated 3x3 normalised convolution over the faulty region only, so
    2- and 3-wide strips fill smoothly and faulty cells never launder each
    other).  --hot-mode rescale divides hot cells by their gain instead.
  * per frame, any active cell with no active valid neighbour is zeroed
    (isolated-pixel removal; disable with --keep-isolated).
  * NaN (no sensor) is preserved exactly; nothing temporal is applied.

Usage:
    python post_processing/4_clean_pressure.py --subjects Kyle Spencer
    python post_processing/4_clean_pressure.py --all --video --max-frames 1500
    python post_processing/4_clean_pressure.py --subjects Kyle --dry-run     # report only
"""

import os
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import COMPLETE, WORK, OFFSETS, QC_PRESSURE, FAULTY_SENSORS  # noqa: E402
import takes as take_names  # noqa: E402  (`takes` is a local variable name here)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors as mcolors
from matplotlib.patches import Rectangle, Patch

OFFS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
FOOT = ["left", "right"]
CLASS_ORDER = ["spurious", "hot", "dead", "manual"]          # fixed order, fixed colours
CLASS_COLOUR = {"spurious": "#d1495b", "hot": "#edae49", "dead": "#30638e", "manual": "#66a182"}
RATIO_BINS = np.logspace(np.log10(0.02), np.log10(50.0), 241)   # log-spaced, ~3.3 % per bin
RATIO_MID = np.sqrt(RATIO_BINS[:-1] * RATIO_BINS[1:])


# --------------------------------------------------------------------------- io

def load_offset(offsets_dir, subject, om):
    p = Path(offsets_dir) / subject / f"{om}.json"
    if not p.exists():
        alt = Path(offsets_dir) / subject / f"{om}_cfg.json"
        if not alt.exists():
            raise FileNotFoundError(f"offset file not found: {p} (or {alt})")
        p = alt
    with open(p) as f:
        return int(json.load(f)["offset"])


def aligned_original(subject, om, work_dir, complete_dir, offsets_dir):
    """Original_Pressure.npy cut exactly like combine_all.py cuts Pressure.npy."""
    src = Path(work_dir) / "Pressure" / subject / om / "Original_Pressure.npy"
    p = np.load(src)
    offset = load_offset(offsets_dir, subject, om)
    start = max(offset, 0)
    ref = Path(complete_dir) / subject / om / "Pressure.npy"
    if ref.exists():
        T = np.load(ref, mmap_mode="r").shape[0]
        if start + T > p.shape[0]:
            raise ValueError(f"{subject}/{om}: Complete has {T} frames but only "
                             f"{p.shape[0] - start} pressure frames remain after offset {offset}")
    else:
        T = p.shape[0] - start
    return p[start:start + T].astype(np.float64)


def load_manual_labels(path, subject):
    """Union of faulty_sensors.json over every take key (case-insensitive) -> (H,W,2) bool."""
    mask = np.zeros((60, 21, 2), dtype=bool)
    if not os.path.exists(path):
        return mask
    with open(path) as f:
        data = json.load(f)
    for om, sides in data.get(subject, {}).items():
        for fi, side in enumerate(FOOT):
            for y, x in sides.get(side, []):
                if 0 <= y < 60 and 0 <= x < 21:
                    mask[y, x, fi] = True
    return mask


# ------------------------------------------------------------------- helpers

def shift(a, dy, dx):
    """out[t, y, x] = a[t, y+dy, x+dx], NaN outside. a is (T,H,W)."""
    T, H, W = a.shape
    out = np.full_like(a, np.nan)
    ys, ye = max(0, -dy), min(H, H - dy)
    xs, xe = max(0, -dx), min(W, W - dx)
    out[:, ys:ye, xs:xe] = a[:, ys + dy:ye + dy, xs + dx:xe + dx]
    return out


def neighbour_stats(s):
    """s: (T,H,W) with NaN for cells that must not be used as neighbours.
    Returns active-neighbour mean, active-neighbour count, valid-neighbour count and
    the direct-neighbour shifts (for bracketing)."""
    nsum = np.zeros_like(s)
    ncnt = np.zeros_like(s)
    nval = np.zeros(s.shape[1:], dtype=np.int32)
    direct = {}
    valid0 = ~np.isnan(s[0])
    for dy, dx in OFFS:
        sh = shift(s, dy, dx)
        v = np.nan_to_num(sh) > 0
        nsum += np.where(v, np.nan_to_num(sh), 0.0)
        ncnt += v
        nval += shift(valid0[None].astype(float), dy, dx)[0] == 1
        if abs(dy) + abs(dx) == 1:
            direct[(dy, dx)] = np.nan_to_num(sh)
    anm = nsum / np.maximum(ncnt, 1)
    return anm, ncnt, nval, direct


def hist_median(hist, n):
    """Median of a per-cell log-histogram (H,W,B) with n valid samples per cell."""
    cum = np.cumsum(hist, axis=-1)
    target = np.ceil(n / 2.0)[..., None]
    idx = np.argmax(cum >= np.maximum(target, 1), axis=-1)
    med = RATIO_MID[idx]
    med[idx == 0] = 0.0            # lowest bin holds the exact zeros
    med[n == 0] = np.nan
    return med


class FootAccumulator:
    """Per-cell statistics pooled over takes for one foot."""

    def __init__(self, H, W):
        nb = len(RATIO_MID)
        self.n_frames = 0
        self.n_act = np.zeros((H, W))
        self.n_sat = np.zeros((H, W))
        self.spur = np.zeros((H, W))
        self.spur_val = np.zeros((H, W))
        self.n_ok = np.zeros((H, W))
        self.hist_r = np.zeros((H, W, nb))
        self.n_ok5 = np.zeros((H, W))
        self.hist_g5 = np.zeros((H, W, nb))
        self.n_br = np.zeros((H, W))
        self.zero_br = np.zeros((H, W))
        self.max_val = np.zeros((H, W))

    def _add_hist(self, hist, ratio, sel):
        H, W, nb = hist.shape
        b = np.digitize(ratio, RATIO_BINS) - 1
        b = np.clip(b, 0, nb - 1)
        cell = np.broadcast_to(np.arange(H * W).reshape(1, H, W), ratio.shape)
        flat = (cell * nb + b)[sel]
        hist += np.bincount(flat, minlength=H * W * nb).reshape(H, W, nb)

    def add(self, s, good, P):
        """s: (T,H,W) values, NaN outside mask.  good: (H,W) cells usable as neighbours."""
        sv = np.nan_to_num(s)
        s_nb = np.where(good[None], s, np.nan)
        anm, ncnt, nval, direct = neighbour_stats(s_nb)
        act = sv > 0
        self.n_frames += s.shape[0]
        self.n_act += act.sum(0)
        self.n_sat += (sv >= P.sat_value).sum(0)
        self.max_val = np.maximum(self.max_val, sv.max(0))
        # spurious: active while no valid neighbour is active
        sp = act & (ncnt == 0) & (nval[None] > 0)
        self.spur += sp.sum(0)
        self.spur_val += (sv * sp).sum(0)
        # gain vs active neighbours (>= 2 active neighbours, clearly loaded)
        ok = act & (ncnt >= 2) & (anm > P.support)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, sv / anm, np.nan)
        self.n_ok += ok.sum(0)
        self._add_hist(self.hist_r, np.nan_to_num(r, nan=1.0), ok)
        # gain incl. zeros with >= 5 active neighbours (weak / dead)
        ok5 = (ncnt >= 5) & (anm > P.support)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(ok5, sv / anm, np.nan)
        self.n_ok5 += ok5.sum(0)
        self._add_hist(self.hist_g5, np.nan_to_num(g, nan=1.0), ok5)
        # bracketed by loaded direct neighbours (H or V)
        brH = (direct[(0, 1)] > P.bracket) & (direct[(0, -1)] > P.bracket)
        brV = (direct[(1, 0)] > P.bracket) & (direct[(-1, 0)] > P.bracket)
        br = brH | brV
        self.n_br += br.sum(0)
        self.zero_br += (br & ~act).sum(0)

    def metrics(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return dict(
                r50=hist_median(self.hist_r, self.n_ok),
                n_ok=self.n_ok,
                g5=hist_median(self.hist_g5, self.n_ok5),
                n_ok5=self.n_ok5,
                deadfrac=np.where(self.n_br > 0, self.zero_br / np.maximum(self.n_br, 1), np.nan),
                n_br=self.n_br,
                spurfrac=np.where(self.n_act > 0, self.spur / np.maximum(self.n_act, 1), np.nan),
                spur_n=self.spur,
                spur_mean=np.where(self.spur > 0, self.spur_val / np.maximum(self.spur, 1), 0.0),
                satfrac=np.where(self.n_act > 0, self.n_sat / np.maximum(self.n_act, 1), 0.0),
                n_act=self.n_act,
                max_val=self.max_val,
            )


def classify(m, mask, P):
    """Return dict of (H,W) bool class masks from one foot's metrics."""
    hot = mask & (m["n_ok"] >= P.hot_min_frames) & (m["r50"] >= P.hot_gain)
    spurious = mask & (m["spur_n"] >= P.spur_min) & (m["spurfrac"] >= P.spur_frac)
    dead_br = mask & (m["n_br"] >= P.dead_min_frames) & (m["deadfrac"] >= P.dead_zero_frac)
    weak = mask & (m["n_ok5"] >= P.weak_min_frames) & (m["g5"] <= P.weak_gain)
    suspect = mask & (m["n_ok"] >= P.hot_min_frames) & (m["r50"] >= P.suspect_gain) & ~hot
    return dict(hot=hot, spurious=spurious, dead=dead_br | weak, suspect=suspect)


# ------------------------------------------------------------------- repair

def inpaint(s, faulty, mask, max_iter=100, tol=1e-3):
    """Harmonic fill of `faulty` cells from the good cells of `mask`. s: (T,H,W)."""
    good = mask & ~faulty
    if not faulty.any():
        return s
    out = np.where(good[None], np.nan_to_num(s), 0.0)
    known = np.broadcast_to(good[None], out.shape).copy()
    fl = faulty[None]
    for _ in range(max_iter):
        nsum = np.zeros_like(out)
        ncnt = np.zeros_like(out)
        for dy, dx in OFFS:
            k = shift(known.astype(float), dy, dx) == 1
            v = shift(out, dy, dx)
            nsum += np.where(k, np.nan_to_num(v), 0.0)
            ncnt += k
        upd = fl & (ncnt > 0)
        new = np.where(upd, nsum / np.maximum(ncnt, 1), out)
        delta = np.abs(new - out)[upd].max() if upd.any() else 0.0
        out = new
        known = known | upd
        if delta < tol:
            break
    out = np.where(mask[None], out, np.nan)
    return out


def remove_isolated(s, mask):
    """Zero active cells that have no active valid neighbour. s: (T,H,W) NaN outside mask."""
    sv = np.nan_to_num(s)
    ncnt = np.zeros_like(sv)
    for dy, dx in OFFS:
        ncnt += np.nan_to_num(shift(s, dy, dx)) > 0
    iso = (sv > 0) & (ncnt == 0)
    out = np.where(iso, 0.0, sv)
    return np.where(mask[None], out, np.nan), int(iso.sum())


def repair_take(p, flags, gains, P):
    """p: (T,H,W,2). flags: dict class->(H,W,2). Returns cleaned array and stats."""
    out = p.copy()
    stats = {}
    for f in range(2):
        mask = ~np.isnan(p[0, ..., f])
        s = np.where(mask[None], np.nan_to_num(p[..., f]), np.nan)
        faulty = flags["all"][..., f].copy()
        if P.hot_mode == "rescale":
            hot_only = flags["hot"][..., f] & ~flags["spurious"][..., f] & ~flags["dead"][..., f] & ~flags["manual"][..., f]
            g = np.where(hot_only, gains[..., f], 1.0)
            g = np.where(np.isnan(g) | (g <= 0), 1.0, g)
            s = s / g[None]
            faulty &= ~hot_only
        s = inpaint(s, faulty, mask)
        n_iso = 0
        if not P.keep_isolated:
            s, n_iso = remove_isolated(s, mask)
        out[..., f] = s
        stats[FOOT[f]] = dict(n_faulty=int(faulty.sum()), isolated_zeroed=n_iso)
    return out, stats


# ---------------------------------------------------------------------- CoP

def cop(p):
    """Per-foot weighted centroid in sensel units. Returns (T,2,2) [foot][row, col] (NaN when unloaded) and totals (T,2)."""
    T, H, W, F = p.shape
    v = np.nan_to_num(p)
    yy = np.arange(H)[None, :, None, None]
    xx = np.arange(W)[None, None, :, None]
    tot = v.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        cy = (v * yy).sum(axis=(1, 2)) / tot
        cx = (v * xx).sum(axis=(1, 2)) / tot
    c = np.stack([cy, cx], axis=-1)
    c[tot == 0] = np.nan
    return c, tot


def cop_report(orig, clean):
    c0, t0 = cop(orig)
    c1, t1 = cop(clean)
    rep = {}
    for f in range(2):
        both = ~np.isnan(c0[:, f, 0]) & ~np.isnan(c1[:, f, 0])
        d = np.linalg.norm(c0[both, f] - c1[both, f], axis=1) if both.any() else np.zeros(0)
        rep[FOOT[f]] = dict(
            frames=int(orig.shape[0]),
            frames_loaded_orig=int((t0[:, f] > 0).sum()),
            frames_loaded_clean=int((t1[:, f] > 0).sum()),
            frames_spurious_only=int(((t0[:, f] > 0) & (t1[:, f] == 0)).sum()),
            cop_shift_mean=float(d.mean()) if d.size else 0.0,
            cop_shift_p95=float(np.percentile(d, 95)) if d.size else 0.0,
            cop_shift_max=float(d.max()) if d.size else 0.0,
            frames_shift_gt_1=int((d > 1.0).sum()),
            load_change_pct=float((t1[:, f].sum() / t0[:, f].sum() - 1.0) * 100.0) if t0[:, f].sum() > 0 else 0.0,
        )
    return rep, (c0, c1, t0, t1)


# --------------------------------------------------------------------- plots

def plot_summary(subject, metrics, flags, mask, P, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 12))
    gain_norm = mcolors.LogNorm(vmin=0.33, vmax=3.0)
    for f in range(2):
        m = metrics[f]
        msk = mask[..., f]
        panels = [
            (np.where(msk & (m["n_ok"] >= 30), m["r50"], np.nan), "RdBu_r", gain_norm, "gain vs active neighbours"),
            (np.where(msk & (m["n_ok5"] >= 30), m["g5"], np.nan), "Blues_r", mcolors.Normalize(0, 1.5), "gain, >=5 loaded neighbours"),
            (np.where(msk, m["spurfrac"], np.nan), "Reds", mcolors.Normalize(0, 1), "isolated-activity fraction"),
        ]
        for j, (img, cmap, norm, title) in enumerate(panels):
            ax = axes[f, j]
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#e8e8e8")
            im = ax.imshow(np.ma.masked_invalid(img), cmap=cm, norm=norm, interpolation="nearest")
            ax.set_title(f"{FOOT[f]}: {title}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = axes[f, 3]
        base = np.where(msk, 0.0, np.nan)
        cm = mcolors.ListedColormap(["#f4f4f4"]); cm.set_bad("#e8e8e8")
        ax.imshow(np.ma.masked_invalid(base), cmap=cm, interpolation="nearest")
        for cls in CLASS_ORDER:
            for y, x in np.argwhere(flags[cls][..., f]):
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=CLASS_COLOUR[cls], edgecolor="none", alpha=0.9))
        for y, x in np.argwhere(flags["suspect"][..., f]):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="none", edgecolor=CLASS_COLOUR["hot"], lw=1.2, ls="--"))
        ax.set_title(f"{FOOT[f]}: repaired cells ({int(flags['all'][..., f].sum())})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if f == 0:
            handles = [Patch(color=CLASS_COLOUR[c], label=c) for c in CLASS_ORDER]
            handles.append(Patch(facecolor="none", edgecolor=CLASS_COLOUR["hot"], ls="--", label=f"suspect (gain {P.suspect_gain}-{P.hot_gain}, not repaired)"))
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.suptitle(f"{subject} - pooled sensel statistics over {metrics[0]['_n_frames']} frames", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_cop(subject, om, traces, path):
    c0, c1, t0, t1 = traces
    T = c0.shape[0]
    t = np.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for f in range(2):
        axes[0, f].plot(t, c0[:, f, 0], color="#9a9a9a", lw=2, label="original")
        axes[0, f].plot(t, c1[:, f, 0], color="#30638e", lw=1.2, label="cleaned")
        axes[0, f].set_ylabel("CoP row (toe=0)")
        axes[0, f].set_title(f"{FOOT[f]} foot")
        axes[1, f].plot(t, c0[:, f, 1], color="#9a9a9a", lw=2)
        axes[1, f].plot(t, c1[:, f, 1], color="#30638e", lw=1.2)
        axes[1, f].set_ylabel("CoP col")
        axes[2, f].plot(t, t0[:, f], color="#9a9a9a", lw=2)
        axes[2, f].plot(t, t1[:, f], color="#30638e", lw=1.2)
        axes[2, f].set_ylabel("total load")
        axes[2, f].set_xlabel("frame")
        for ax in axes[:, f]:
            ax.grid(color="#eeeeee", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"{subject}/{om}: centre of pressure before / after repair", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_video(orig, clean, flags, path, fps=50, max_frames=None):
    N = orig.shape[0] if max_frames is None else min(orig.shape[0], max_frames)
    valid = orig[:N][~np.isnan(orig[:N]) & (orig[:N] > 0)]
    vmax = np.percentile(valid, 99) if valid.size else 1.0
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")

    def frame(arr, i):
        return np.ma.masked_invalid(np.hstack([arr[i, :, :, 0], arr[i, :, :, 1]]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    ims = []
    for ax, arr, title in zip(axes, (orig, clean), ("original", "cleaned")):
        ims.append(ax.imshow(frame(arr, 0), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"))
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        for f in range(2):
            for y, x in np.argwhere(flags["all"][..., f]):
                ax.add_patch(Rectangle((x + 21 * f - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="#4cc9f0", lw=0.8))
    txt = axes[0].text(0.02, 0.98, "", transform=axes[0].transAxes, va="top", color="white", fontsize=9)
    fig.tight_layout()

    def update(i):
        ims[0].set_data(frame(orig, i)); ims[1].set_data(frame(clean, i))
        txt.set_text(f"frame {i}/{N - 1}")
        return ims + [txt]

    ani = animation.FuncAnimation(fig, update, frames=N, blit=True, interval=1000 / fps)
    ani.save(path, writer="ffmpeg", fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------- main

def process_subject(subject, P):
    subj_dir = Path(P.complete_dir) / subject
    selection = take_names.parse_selection(P.oms)
    takes = [d.name for d in take_names.take_dirs(subj_dir) if take_names.selected(d.name, selection)]
    if not takes:
        print(f"[{subject}] no takes found in {subj_dir}")
        return
    out_dir = Path(P.output_dir) / subject / "pressure_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{subject}] {len(takes)} takes: {' '.join(takes)}")

    first = aligned_original(subject, takes[0], P.work_dir, P.complete_dir, P.offsets_dir)
    H, W = first.shape[1:3]
    mask = ~np.isnan(first[0])                      # (H,W,2), static for every take
    manual = load_manual_labels(P.manual_labels, subject) & mask if not P.no_manual else np.zeros_like(mask)

    # ---- detection: pooled over takes, P.passes rounds (flagged cells drop out of the neighbourhoods)
    flagged = manual.copy()
    metrics = None
    for pas in range(P.passes):
        accs = [FootAccumulator(H, W) for _ in range(2)]
        for om in takes:
            p = aligned_original(subject, om, P.work_dir, P.complete_dir, P.offsets_dir)
            for f in range(2):
                s = np.where(mask[None, ..., f], np.nan_to_num(p[..., f]), np.nan)
                accs[f].add(s, mask[..., f] & ~flagged[..., f], P)
        metrics = [a.metrics() for a in accs]
        for f in range(2):
            metrics[f]["_n_frames"] = accs[f].n_frames
        cls = [classify(metrics[f], mask[..., f], P) for f in range(2)]
        auto = np.stack([cls[f]["hot"] | cls[f]["spurious"] | cls[f]["dead"] for f in range(2)], -1)
        new = auto & ~flagged
        print(f"  pass {pas + 1}: {int(auto.sum())} auto-flagged cells ({int(new.sum())} new)")
        flagged |= auto
        if pas > 0 and new.sum() == 0:
            break

    flags = {c: np.stack([cls[f][c] for f in range(2)], -1) for c in ("hot", "spurious", "dead", "suspect")}
    flags["manual"] = manual & ~(flags["hot"] | flags["spurious"] | flags["dead"])
    flags["all"] = flagged
    gains = np.stack([metrics[f]["r50"] for f in range(2)], -1)

    # ---- report
    cells = []
    for f in range(2):
        for y, x in np.argwhere(flagged[..., f] | flags["suspect"][..., f]):
            m = metrics[f]
            classes = [c for c in CLASS_ORDER if flags[c][y, x, f]]
            if not classes:
                classes = ["suspect"]
            cells.append(dict(foot=FOOT[f], y=int(y), x=int(x), classes=classes,
                              manual=bool(manual[y, x, f]),
                              gain=None if np.isnan(m["r50"][y, x]) else round(float(m["r50"][y, x]), 3),
                              gain_frames=int(m["n_ok"][y, x]),
                              gain5=None if np.isnan(m["g5"][y, x]) else round(float(m["g5"][y, x]), 3),
                              deadfrac=None if np.isnan(m["deadfrac"][y, x]) else round(float(m["deadfrac"][y, x]), 3),
                              bracket_frames=int(m["n_br"][y, x]),
                              spurious_frames=int(m["spur_n"][y, x]),
                              spurious_frac=None if np.isnan(m["spurfrac"][y, x]) else round(float(m["spurfrac"][y, x]), 3),
                              spurious_mean_value=round(float(m["spur_mean"][y, x]), 1),
                              saturation_frac=round(float(m["satfrac"][y, x]), 4),
                              max_value=float(m["max_val"][y, x])))
    counts = {c: int(flags[c].sum()) for c in CLASS_ORDER + ["suspect"]}
    counts["all_repaired"] = int(flagged.sum())
    print("  flagged: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for c in cells:
        if "suspect" in c["classes"]:
            continue
        print(f"    {c['foot']:5s} ({c['y']:2d},{c['x']:2d}) {'+'.join(c['classes']):15s} gain={c['gain']} gain5={c['gain5']} "
              f"dead={c['deadfrac']} spur={c['spurious_frames']}/{c['spurious_frac']} sat={c['saturation_frac']}"
              + ("  [manual]" if c["manual"] else ""))

    np.save(out_dir / "faulty_mask.npy", flagged)
    np.savez(out_dir / "metrics.npz", mask=mask, **{f"{k}_{FOOT[f]}": v for f in range(2) for k, v in metrics[f].items() if not k.startswith("_")})
    plot_summary(subject, metrics, flags, mask, P, out_dir / "summary.png")

    # ---- repair every take
    take_reports = {}
    for om in takes:
        p = aligned_original(subject, om, P.work_dir, P.complete_dir, P.offsets_dir)
        clean, rstats = repair_take(p, flags, gains, P)
        rep, traces = cop_report(p, clean)
        for f in FOOT:
            rep[f].update(rstats[f])
        take_reports[om] = rep
        l, r = rep["left"], rep["right"]
        print(f"  {om:5s} T={p.shape[0]:5d}  CoP shift mean/p95 (sensels): L {l['cop_shift_mean']:.2f}/{l['cop_shift_p95']:.2f}  "
              f"R {r['cop_shift_mean']:.2f}/{r['cop_shift_p95']:.2f}   spurious-only frames L {l['frames_spurious_only']} R {r['frames_spurious_only']}   "
              f"load change L {l['load_change_pct']:+.1f}% R {r['load_change_pct']:+.1f}%   isolated zeroed L {l['isolated_zeroed']} R {r['isolated_zeroed']}")
        plot_cop(subject, om, traces, out_dir / f"{om}_cop.png")
        if not P.dry_run:
            dst = Path(P.complete_dir) / subject / om / P.out_name
            np.save(dst, clean)
        if P.video:
            save_video(p, clean, flags, out_dir / f"{om}_before_after.mp4", fps=50, max_frames=P.max_frames)

    with open(out_dir / "report.json", "w") as f:
        json.dump(dict(subject=subject, takes=takes, params=vars(P), counts=counts, cells=cells,
                       per_take=take_reports), f, indent=2, default=str)
    print(f"  report -> {out_dir}" + ("" if not P.dry_run else "   (dry run: no Pressure_clean.npy written)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help="process every subject in Complete/")
    ap.add_argument("--oms", nargs="+", default=None, help="restrict to these takes (numbers and/or names); NOTE this also narrows the detection pool")
    ap.add_argument("--complete_dir", default=COMPLETE)
    ap.add_argument("--work_dir", "--untrimmed_dir", dest="work_dir", default=WORK)
    ap.add_argument("--offsets_dir", default=OFFSETS)
    ap.add_argument("--output_dir", default=QC_PRESSURE)
    ap.add_argument("--manual_labels", default=FAULTY_SENSORS)
    ap.add_argument("--out_name", default="Pressure_clean.npy")
    ap.add_argument("--no_manual", action="store_true", help="ignore faulty_sensors.json")
    ap.add_argument("--dry_run", action="store_true", help="detect + report, do not write Pressure_clean.npy")
    ap.add_argument("--video", action="store_true", help="write side-by-side before/after videos")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--hot_mode", choices=["inpaint", "rescale"], default="inpaint")
    ap.add_argument("--keep_isolated", action="store_true", help="do not zero isolated active cells per frame")
    ap.add_argument("--passes", type=int, default=2)
    # thresholds
    ap.add_argument("--hot_gain", type=float, default=2.0)
    ap.add_argument("--suspect_gain", type=float, default=1.5)
    ap.add_argument("--hot_min_frames", type=int, default=100)
    ap.add_argument("--spur_frac", type=float, default=0.25)
    ap.add_argument("--spur_min", type=int, default=50)
    ap.add_argument("--dead_zero_frac", type=float, default=0.9)
    ap.add_argument("--dead_min_frames", type=int, default=200)
    ap.add_argument("--weak_gain", type=float, default=0.25)
    ap.add_argument("--weak_min_frames", type=int, default=300)
    ap.add_argument("--support", type=float, default=30.0, help="neighbour mean must exceed this to count as loaded")
    ap.add_argument("--bracket", type=float, default=30.0, help="direct neighbours must exceed this to bracket a dead cell")
    ap.add_argument("--sat_value", type=float, default=850.0)
    P = ap.parse_args()
    if P.all:
        subjects = sorted(d.name for d in Path(P.complete_dir).iterdir()
                          if d.is_dir() and take_names.take_dirs(d))
    elif P.subjects:
        subjects = P.subjects
    else:
        ap.error("give --subjects or --all")
    for s in subjects:
        process_subject(s, P)


if __name__ == "__main__":
    main()
