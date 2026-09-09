
## Known issues and gotchas

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

### Ordering

Stage D edits `Complete/` directly, not `untrimmed/`. **Re-running step 8
(`combine_all.py`) regenerates `Pressure.npy` from `untrimmed/` and silently
discards the repair.** If you re-run the pipeline, re-run stage D afterwards —
`faulty_sensors.json` is persistent, so D3 alone is enough.

Note also that D1 renders from `untrimmed/` (uncut) while D3 renders from
`Complete/` (offset-cut), so the "before" and "after" videos are not on the same
frame numbering.

---

## QC visualisations

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
