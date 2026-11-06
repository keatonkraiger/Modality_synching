
## Known issues and gotchas

- **`Complete/*/Pressure.npy` is not the recorded data for any subject.** The old
  stage D3 was run with `--save-cleaned` on every subject (Keaton included, whose
  labels were never written to `faulty_sensors.json`), so the hand-labelled cells in
  `Complete/` are already their 3×3 neighbour mean. The pristine arrays are
  `untrimmed/Pressure/<Subject>/OM<i>/Original_Pressure.npy`; stage D4 reads those
  and re-applies the `combine_all.py` offset cut itself.
- **The rows-14–17 band that reads zero on both feet of every subject is not a dead
  strip.** It is the toe-crease region, which is simply lightly loaded (max ~40–90
  units over a whole session, neighbours in rows 13/18 also low). Any dead-sensel
  detector that looks for "zero while bracketed above and below" will flag it; D4
  deliberately requires ≥5 loaded neighbours or direct H/V bracketing.
- **Edge cells at gain 1.5–2.0 are probably insole curl, not sensor faults.** The
  outermost loaded column on the lateral side (cols 1–2 / 17–18) often reads 1.5–1.7×
  its inner neighbours; D4 reports these as `suspect` but does not repair them by
  default (`--hot_gain` lowers the bar if you want them).
- **`combine_all.py`'s `CoM_inst` branch never fires.** It looks for
  `untrimmed/Model_output/<Subject>/<OM>/CoM_inst.npy`, but stage C writes
  instantaneous CoM straight to `Complete/`. The branch is vestigial; the working
  path is `instant_com.py --inst-save-root Complete`.
- **`convert_model_output.py` hardcodes `raw_data/offsets`** — it takes no
  `--offsets_dir` flag, unlike `combine_all.py`. If you keep offsets elsewhere,
  the two stages will disagree.
- **`faulty_sensors.json` keys are case-inconsistent** — Kyle uses `OM0`, Varad and
  Abhinav use `om0`. Repair unions across all keys so this is harmless today, but
  re-annotating a lowercase take through D1 will create a second, uppercase entry
  for the same OM rather than merging into the existing one.
- **`clean_pressure.py` has dead parameters.** `clean_pressure_data()` accepts
  `remove_isolated`, `remove_temporal` and `apply_median`, and the corresponding
  `remove_isolated_pixels` / `remove_temporal_outliers` /
  `apply_median_filter_selective` functions are defined — but none of them are
  called. Only the clip/threshold step runs. Enable them deliberately if you want
  them; don't assume they are already active.
- **`clean_pressure.py` assumes a 99 Hz recording.** The decimation is
  `combined[::int(99/target_fps)]`, i.e. hardcoded to halve. The inline comment
  claiming a 49 fps source contradicts the code. A genuinely 50 Hz export would be
  silently halved to 24 Hz.
- **`min_val=9` zeroes light contact.** Anything under 10 in the raw Tekscan units
  is discarded in step 1 and cannot be recovered without re-parsing the CSVs.
- **Step 0 is destructive.** `convert_all_fps.py` deletes and replaces the source
  video. Back up `raw_data/Video/` first.
- **`triangulate_oms.py`'s non-standard-resolution handling is half-applied.**
  `subject_broken_takes` (Spencer OM0, Abhinav OM11, and Varad OM1/3/8/9/10/12)
  marks takes recorded at 1279×720 instead of 1920×1080. For those, the intrinsics
  `K` are rescaled — but `scale_x` / `scale_y` are set to `0, 1` in *both* branches,
  so the 1D keypoints are never rescaled to match. Either the keypoints are already
  in 719p (in which case this is correct and the `scale_*` machinery is just dead)
  or they are not (in which case those takes' 2D is wrong). Verify the actual
  resolution of those videos before trusting their `BODY24_3D.npy`.
- **`OrdinaryMovements/` is maintained by hand.** No script in this repo produces
  or updates it; it is a copy of `Complete/` with the videos removed.

---

## Post-processing: pressure sensor repair

Some mats have individual dead or saturated cells that read high regardless of
load. Detection is by eye — the automatic detector was not reliable enough — so
this stage is *look, label, repair, look again*.

Faulty cells are stored **per subject** and repaired using the **union across all
of that subject's takes**, on the reasoning that a physically dead sensor is dead
for the whole session.

### D1. Render "before" videos

```bash
python post_processing/1_generate_pressure_viz.py \
    --subjects <Subject> [<Subject2> ...] \
    [--max_frames 1000] [--append raw]
```

Writes `post_processing_output/<Subject>/pressure_viz/OM<i>_pressure.mp4` from
`untrimmed/Pressure/`. Colour scale is capped at the 99th percentile so a single
stuck cell doesn't flatten the whole map. Watch these and note, for each take, a
frame number where a suspicious cell is clearly lit.

### D2. Label the faulty cells (interactive)

```bash
python post_processing/2_find_faulty_sensors.py \
    --subject <Subject> --om OM<i> --frame <frame_from_D1> [--vmax 200]
```

Shows that frame for both feet. **Left-click toggles a cell**; already-labelled
cells (the union across every OM of this subject) are outlined in red on open.
Press **`q`** to save and quit.

Appends to `post_processing_output/faulty_sensors.json`:

```json
{ "<Subject>": { "OM<i>": { "left": [[y, x], ...], "right": [[y, x], ...] } } }
```

New labels are *merged* with what's already there, never replaced — so you can
come back to a take repeatedly. To remove a mistaken label, edit the JSON by hand.

Needs a GUI display (see [Setup](#1-system-dependencies)).

### D3. Apply the repair

```bash
# dry run first: writes videos, does NOT touch the data
python post_processing/3_apply_sensor_fix.py --subject <Subject> --create-video

# once the videos look right:
python post_processing/3_apply_sensor_fix.py --subject <Subject> --save-cleaned --create-video
```

Builds the union mask over all of the subject's OMs and, for every
`Complete/<Subject>/OM<i>/Pressure.npy`, replaces each faulty cell with the mean
of its 3×3 spatial neighbours — **excluding other faulty cells**, so two adjacent
dead sensors don't launder each other's values. The correction is purely spatial;
no temporal smoothing is applied. Cleaned previews go to
`post_processing_output/<Subject>/cleaned_pressure_viz/`.

`--save-cleaned` **overwrites `Pressure.npy` in place with no backup.** The
neighbour-mean repair is idempotent (re-running it is a no-op), so a repeat run is
harmless — but see the warning on `--apply-global-smooth` below.

> **Avoid `--apply-global-smooth`.** It blurs *every* cell as `(value + neighbour_mean)/2`
> and, critically, treats `NaN` as missing data to be filled — but `NaN` here means
> "no sensor at this location". On a sample frame it invented pressure in 124 of
> 305 non-sensor cells, inflated the take's total load by ~6%, and cut the peak
> reading from 148 to 133. It is also not idempotent, so combined with
> `--save-cleaned` each run degrades the data further. Leave it off unless you are
> deliberately experimenting, and never with `--save-cleaned` on data you care about.

### D4. Automatic detection + repair → `Pressure_clean.npy` (preferred)

```bash
python post_processing/4_clean_pressure.py --subjects <Subject> [--dry_run] [--video --max_frames 1500]
python post_processing/4_clean_pressure.py --all
```

Supersedes D1–D3 for the repair itself (D2's hand labels are still merged in).
Reads the pristine `untrimmed/.../Original_Pressure.npy`, applies the same offset
cut as `combine_all.py`, and writes **`Complete/<Subject>/OM<i>/Pressure_clean.npy`**
next to the untouched `Pressure.npy`. Takes about 3 min per subject; detection
pools every take of the subject (same insoles all session), so `--oms` only
restricts which takes are written.

Three fault classes, thresholds calibrated on the hand labels and all CLI-tunable:

| class | test (pooled over the subject's takes) | default |
|---|---|---|
| `hot` | median of cell ÷ mean of its *active* neighbours, over frames with ≥2 active neighbours and neighbour mean > 30 | ≥ 2.0 (`--hot_gain`), ≥100 frames; population 99.5th pct is 1.5–1.9 |
| `spurious` | active while no valid neighbour is active | ≥25 % of its active frames and ≥50 frames (`--spur_frac`, `--spur_min`) |
| `dead` | zero in ≥90 % of frames where both direct H or V neighbours exceed 30, **or** median gain ≤0.25 with ≥5 active neighbours (≥300 frames) | `--dead_zero_frac`, `--weak_gain` |

Detection runs twice: cells flagged in pass 1 are dropped from every
neighbourhood in pass 2, so the second cell of a 2-wide dead strip, or the next
cell of a hot column, gets judged against good neighbours only.

Repair: flagged cells are filled by harmonic inpainting from the good cells around
them (iterated masked 3×3 mean over the faulty region only, so faulty cells never
launder each other and 2–3-wide strips fill smoothly); `--hot_mode rescale`
divides hot cells by their gain instead of replacing them. Then, per frame, any
active cell with no active valid neighbour is zeroed (`--keep_isolated` disables
this). `NaN` is preserved exactly and nothing temporal is applied, so the result
is idempotent and frame-aligned with `Pressure.npy`.

Outputs in `post_processing_output/<Subject>/pressure_clean/`:

- `summary.png` — gain / dead / isolated-activity maps and the repaired-cell map
- `OM<i>_cop.png` — per-foot CoP row/col and total load, before vs after
- `report.json` — every flagged cell with its metrics, plus per-take CoP shift
  (mean / p95 in sensel units), `frames_spurious_only` (frames where the original
  had load and the cleaned foot has none, i.e. foot-in-the-air ghosts) and load
  change; `faulty_mask.npy` `(60,21,2)`; `metrics.npz` for every cell
- `OM<i>_before_after.mp4` with `--video`

`--dry_run` writes everything except `Pressure_clean.npy`.

### Ordering

Stage D edits `Complete/` directly, not `untrimmed/`. **Re-running step 8
(`combine_all.py`) regenerates `Pressure.npy` from `untrimmed/` and silently
discards the D3 repair** (D4's `Pressure_clean.npy` is a separate file and
survives, but re-run D4 anyway if the offsets changed). `faulty_sensors.json` is
persistent, so D3 or D4 alone is enough.

Note also that D1 renders from `untrimmed/` (uncut) while D3 renders from
`Complete/` (offset-cut), so the "before" and "after" videos are not on the same
frame numbering.

---

## QC visualisations

### Comparison video: RGB + repaired vs unrepaired pressure

```bash
python viz/compare_pressure_video.py --subject Varad --om OM11 --old_source original
python viz/compare_pressure_video.py --subject Varad --om OM11 --start 2400 --max_frames 600 --view V2
```

Three panels hstacked at a common pixel height, each keeping its own aspect ratio:
one RGB view, `Pressure_clean.npy`, and the chosen "before" array, plus a shared
colour bar. Both pressure panels use **one** colour scale (99.5th percentile of the
repaired array) so they are directly comparable. A cyan ring marks each foot's
centre of pressure, repaired cells are outlined on the "before" panel, and the
footer prints the live per-foot CoP shift in sensels.

Frames are written one at a time as JPEGs to `<out>.frames/` and muxed with ffmpeg,
then deleted, so memory stays flat regardless of take length. `--keep_frames` keeps
them. Output defaults to
`post_processing_output/<Subject>/pressure_clean/<OM>_compare_<view>.mp4`.

**Choose the "before" array deliberately.** `--old_source complete` (the default)
compares against `Complete/.../Pressure.npy`, which for any hand-labelled cell is
already the old D3 output, not the recording — for Varad OM11 that comparison is
nearly blank because D3 had already fixed the two worst ghost sensels. Use
`--old_source original` for the true before/after against
`untrimmed/.../Original_Pressure.npy`.

```bash
python viz/construct_viz.py --name <Subject> [--oms OM1 OM3] [--use_inst_com] \
    [--skip_step pressure skeleton markers collage]
```

First re-verifies that all eight `.npy` files and both videos in the take have
matching frame counts, then renders into `manual_check/<Subject>/OM<i>/`:

- `pressure.mp4` — both feet
- `3D_joints.mp4` — BODY25 3D skeleton + CoM + floor projection
- `3D_markers.mp4` — the 39 Vicon markers + CoM
- `final.mp4` — all of the above plus both RGB views in one collage

`--use_inst_com` swaps `CoM.npy` for `CoM_inst.npy` and writes to
`*_inst_com.mp4` filenames, so you can render both and compare side by side.

Two smaller ad-hoc viewers exist for quick interactive looks; both have their
target path hardcoded near the top and are meant to be edited:

- `viz/viz_3d.py` — skeleton + CoM for one take
- `viz/viz_pressure.py` — pressure only
