# Ordinary Movements (OM) Dataset Construction

Turns raw multimodal captures — two RGB views, Tekscan foot-pressure mats, Vicon
mocap, and OpenPose 2D keypoints — into per-take, frame-aligned 50 fps bundles in
`Complete/`. Synchronisation between the pressure system and the video/mocap
systems is **manual**: a human scrubs the two streams side by side and records a
frame offset.

---

## Contents

- [Setup](#setup)
- [Data layout](#data-layout)
- [Pipeline](#pipeline)
  - [Stage A — build the untrimmed modalities](#stage-a--build-the-untrimmed-modalities)
  - [Stage B — synchronise and cut `Complete/`](#stage-b--synchronise-and-cut-complete)
  - [Stage C — instantaneous CoM](#stage-c--instantaneous-com)
- [Utilities](#utilities)

Building the dataset ends at Stage C. See **[claude.md](claude.md)** for what
comes after — the pressure sensor repair walkthrough, the QC visualisations, and
the pipeline's known issues and gotchas.

---

## Setup

### 1. System dependencies

`ffmpeg` (and `ffprobe`) must be on `PATH`. Every `.mp4` this repo writes goes
through matplotlib's ffmpeg writer, and fps probing/conversion shells out to
ffmpeg directly.

```bash
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version && ffprobe -version
```

Two steps are interactive GUIs (`syncher.py`, `post_processing/2_find_faulty_sensors.py`)
and need a working display. On WSL2 with WSLg this works out of the box; on a
plain remote box you need an X server and `export DISPLAY=...`. PyQt5 may also
want the usual X libs:

```bash
sudo apt install -y libxcb-xinerama0 libgl1 libglib2.0-0
```

### 2. Virtual environment

```bash
cd /mnt/e/Data/OMs/Modality_synching

python3 -m venv .venv
source .venv/bin/activate            # zsh/bash;  .venv\Scripts\activate on Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`.venv/` is already covered by `.gitignore`.

Verify:

```bash
python -c "import numpy, scipy, matplotlib, cv2, pandas, tqdm; print('core OK')"
python -c "import PyQt5; print('GUI OK')"          # only needed for the two GUI steps
```

To leave the environment: `deactivate`.

### 3. OpenPose (external)

OpenPose is **not** installed by `requirements.txt`. It is a separate C++/CUDA
build used once, in [step 4](#4-run-openpose), to produce per-frame JSON keypoints.
Install it from the upstream project and make sure its BODY_25 model is available.
Everything downstream only reads the JSON files it emits, so if you already have
`raw_data/openpose_output/` populated you can skip installing it entirely.

---

## Data layout

Paths below are the **defaults the scripts actually use**. Directories not tracked
in git (they hold the data itself) are marked *(data)*.

```
raw_data/                                    (data) — untouched captures
├── Video/<Subject>/OM<i>_V[1|2].mp4          two synchronised RGB views, 50 fps
│   └── original_fps/OM<i>_V[1|2].mp4         pre-conversion source, only for
│                                             takes that were not already 50 fps
├── Pressure/<Subject>/OM<i>_[L|R].csv        Tekscan exports, one CSV per foot
├── openpose_output/<Subject>/OM<i>/OM<i>_V[1|2]_jsons/*.json
├── XCPs/<Subject>/OM<i>.xcp                  Vicon camera calibration per take
├── Model_output/<Subject>/OM<i>.json         Vicon model output + markers export
└── offsets/<Subject>/OM<i>.json              manual sync offsets (from syncher.py)

untrimmed/                                   (data) — per-modality, not yet cut
├── Pressure/<Subject>/OM<i>/Original_Pressure.npy   (N,60,21,2) @50 Hz
│                            Original_Pressure.mp4   preview used for syncing
├── BODY25/<Subject>/OM<i>/BODY25_V[1|2].npy   (N,25,3)  x,y,conf
│                          BODY25_3D.npy       (N,25,4)  x,y,z,conf
└── Model_output/<Subject>/OM<i>/MOCAP_3D.npy  (N,17,4)
                                MOCAP_MRK.npy  (N,39,4)
                                CoM.npy        (N,4)
                                CoM_floor.npy  (N,4)

Model_output/<Subject>/OM<i>.npy             (data) — full Vicon structure,
                                             already resampled + offset-cut.
                                             Input to the instantaneous-CoM code.

Complete/<Subject>/OM<i>/                    (data) — THE DELIVERABLE, 50 fps,
├── Video_V1.mp4, Video_V2.mp4                all streams frame-for-frame aligned
├── Pressure.npy            (N,60,21,2)   as recorded (after clip/threshold in step 2)
├── Pressure_clean.npy      (N,60,21,2)   faulty sensels repaired (stage D4); same NaN mask
├── BODY25_V1.npy, BODY25_V2.npy, BODY25_3D.npy
├── MOCAP_3D.npy, MOCAP_MRK.npy
├── CoM.npy, CoM_floor.npy                    Vicon's own CoM
├── CoM_inst.npy, CoM_floor_inst.npy          instantaneous CoM (Stage C)
└── manifest.json                             which streams this take has
Complete/om_sub_info.json                     per-subject mass / height / gender / age

post_processing_output/                      (data) — QC videos + annotations
├── faulty_sensors.json                       hand-labelled dead pressure sensors
├── <Subject>/pressure_viz/ , cleaned_pressure_viz/
└── <Subject>/pressure_clean/                 stage D4 report: summary.png, OM<i>_cop.png,
                                             report.json, faulty_mask.npy, metrics.npz

manual_check/<Subject>/OM<i>/                (data) — QC collage videos
OrdinaryMovements/                           (data) — release mirror of Complete/
                                             minus the videos. Copied by hand; no
                                             script in this repo produces it.
```

### Array conventions

- **Pressure** `(N, 60, 21, 2)` — last axis is `[left, right]` foot. `NaN` means
  *there is no sensor at that cell* (the mat outline) and is **static across all
  frames of a take**; `0` means a real sensor reading nothing. Do not conflate
  the two — filling `NaN` invents pressure outside the mat.
- **Poses / markers / CoM** — last column is a validity/confidence flag. `1.0`
  valid, `0.0` invalid (and XYZ is then meaningless, usually zeroed).
- **Units** — millimetres for all 3D data, in the Vicon world frame.
- **Rate** — everything in `Complete/` is 50 fps.

### Partial takes

The six mocap-derived arrays (`MOCAP_3D`, `MOCAP_MRK`, `CoM`, `CoM_floor`,
`CoM_inst`, `CoM_floor_inst`) are **optional per take**. A take that was labelled
in Nexus but never run through Plug-in-Gait has markers and nothing else; a take
with no Vicon export at all has none of them. Neither is treated as an error —
the video / pressure / OpenPose bundle is still built and synced, so every take
can be synchronised even when the mocap side is incomplete.

Each take directory carries a `manifest.json` saying exactly what it got:

```json
{
  "subject": "Subject1", "om": "OM1", "frames": 4732, "fps": 50, "offset": -88,
  "streams": ["Video_V1.mp4", "...", "Pressure.npy", "MOCAP_MRK.npy"],
  "vicon_complete": false,
  "missing_vicon": ["MOCAP_3D.npy", "CoM.npy", "CoM_floor.npy",
                    "CoM_inst.npy", "CoM_floor_inst.npy"]
}
```

Read `vicon_complete` rather than probing for files. Stages that genuinely need
mocap — the instantaneous CoM (C1/C2) and the skeleton/marker visualisations —
skip a partial take with a message instead of failing the whole run.

---

## Pipeline

```
                    raw_data/Video ──1──► 50 fps videos
                                            │
                        ┌───────────────────┼─────────────────┐
                        │                   │                 │
raw_data/Pressure ──2──► untrimmed/Pressure │        4  OpenPose (external)
                        │                   │                 │
                        └──3── syncher.py ──┘                 5  format_OP_jsons
                               (manual offsets)               │
raw_data/XCPs ────────────────────────────────6── triangulate_oms
raw_data/Model_output ────────────────────────7── get_model_output
                                                              │
                                          8  combine_all ──► Complete/
                                                              │
                                    C1 convert_model_output ──┤
                                    C2 instant_com ───────────┘  (+CoM_inst)
```

Repair and QC of the finished dataset are documented separately in
[claude.md](claude.md).

---

### Stage A — build the untrimmed modalities

#### 1. Normalise all videos to 50 fps

```bash
python scripts/convert_all_fps.py --root_dir raw_data/Video/<Subject>
```

Walks the tree for `OM<i>_V[1|2].mp4` and re-encodes anything that isn't already
50 fps with `-filter:v fps=50`. Videos already at 50 fps are left untouched.

The re-encoded file takes over the original filename, so everything downstream
just sees 50 fps videos. Before that swap happens the **untouched source is moved
to `<same dir>/original_fps/<same filename>`**, so the data at its original frame
rate is always recoverable. A `conversion_log.json` at the root of `--root_dir`
records each take's original fps, new fps, and the path of its preserved copy.

The step is safe to re-run: `original_fps/` is pruned from the directory walk, and
a take whose backup already exists is skipped rather than converted a second time
(which would otherwise overwrite the only remaining original).

#### 2. Parse and clean the pressure CSVs

```bash
python scripts/clean_pressure.py \
    --data_dir raw_data/Pressure/<Subject> \
    --save_dir untrimmed/Pressure/<Subject>
```

For each `OM<i>_L.csv` / `OM<i>_R.csv` pair it:

1. Parses the Tekscan ASCII export (31 header lines, then repeating `Frame` blocks
   of 60 rows × 21 columns; `B` cells become `NaN`).
2. Truncates both feet to the shorter of the two.
3. Zeroes readings below `min_val=10`, clips to `max_val=862`, and zeroes anything
   above `impossible_threshold=1500`.
4. Stacks to `(N, 60, 21, 2)` and **decimates 100 Hz → 50 Hz** by taking every
   other frame.
5. Writes `Original_Pressure.npy` and `Original_Pressure.mp4` per take.

The `.mp4` is what you scrub against the RGB video in the next step.

#### 3. Determine the sync offset (manual)

```bash
python syncher.py \
    --rgb      raw_data/Video/<Subject>/OM<i>_V1.mp4 \
    --pressure untrimmed/Pressure/<Subject>/OM<i>/Original_Pressure.mp4
```

A PyQt window shows both streams side by side. Scrub, set the **offset (frames)**
slider until an unambiguous event (a heel strike, a stomp) lines up, press
*Test Offset Playback* to confirm, then *Save Sync Config*.

Save to `raw_data/offsets/<Subject>/OM<i>.json`. The file looks like:

```json
{
  "rgb_video_path": "...\/OM2_V1.mp4",
  "pressure_video_path": "...\/Original_Pressure.mp4",
  "offset": -103,
  "rgb_video_fps": 50.0,
  "pressure_fps": 50.0
}
```

**Sign convention.** `offset` is *pressure start − video start*, in 50 fps frames:

- `offset < 0` — video/mocap started **first**. The first `abs(offset)` frames of
  video/mocap/OpenPose are dropped; pressure is used from frame 0. This is the
  common case.
- `offset > 0` — pressure started first. The first `offset` pressure frames are
  dropped; video/mocap is used from frame 0.

Only the `offset` key is consumed downstream. The fps keys are informational —
the resampling decision in step 8 is made from array lengths, not from fps
metadata. (An `original_video_fps` key is also read if present, but it is only
printed; it does not change any behaviour.)

#### 4. Run OpenPose

Run each 50 fps video through OpenPose with the BODY_25 model, writing per-frame
JSON to:

```
raw_data/openpose_output/<Subject>/OM<i>/OM<i>_V[1|2]_jsons/*.json
```

If your OpenPose invocation produced `OM<i>/json/Video_V[1|2]/` instead, `move.py`
restructures it into the layout above — edit the `base_dir` at the bottom of the
file and run `python move.py`.

#### 5. OpenPose JSON → `.npy`

```bash
python scripts/format_OP_jsons.py --name <Subject> [--specific_oms 1 3]
```

Collapses each `_jsons` directory into `untrimmed/BODY25/<Subject>/OM<i>/BODY25_V[1|2].npy`
of shape `(N, 25, 3)`. Only the **first detected person** is kept; frames with no
detection become zeros.

It cross-checks the JSON count against the video frame count: a difference of 1 is
tolerated (both are truncated to the shorter), anything larger raises and aborts —
that almost always means the OpenPose run and the 50 fps conversion disagree.

#### 6. Triangulate to 3D

```bash
python scripts/triangulate_oms.py --subjects <Subject> [--specific_oms OM1 OM3] [--animate] [--dry_run]
```

Parses the per-take `.xcp` for both Vue cameras (`2111392` → Camera 1,
`2111394` → Camera 2), builds `K` and `P`, undistorts the 2D keypoints with the
Vicon radial model, and triangulates to `BODY25_3D.npy` `(N, 25, 4)`.

- `--animate` renders a 3D preview instead of just writing files.
- `--dry_run` computes without saving.
- **Do not pass `--no_world_fix`** for Keaton, Kyle, Varad, Spencer, or Abhinav —
  their captures need the world-frame correction to land in the same frame as the
  Vicon data.

#### 7. Extract the Vicon model output

```bash
python scripts/get_model_output.py --person <Subject> [--specific_oms 1 3]
```

Reads `raw_data/Model_output/<Subject>/OM<i>.json` and writes, per take:

| File | Shape | Contents |
|---|---|---|
| `MOCAP_3D.npy` | `(N, 17, 4)` | joint centres in a fixed order (R/L shoulder-elbow-wrist, R/L hip-knee-ankle, pelvis, waist, neck, clavicle, thorax) |
| `MOCAP_MRK.npy` | `(N, 39, 4)` | the 39 Plug-in-Gait markers in a fixed order |
| `CoM.npy` | `(N, 4)` | Vicon `CentreOfMass` |
| `CoM_floor.npy` | `(N, 4)` | Vicon `CentreOfMassFloor` |

Missing markers are filled with zeros and reported. A `Rz(-90°)` rotation
(`X,Y,Z → Y,−X,Z`) is applied by default to bring Vicon into the shared world
frame — again, **do not pass `--no_world_fix`** for the five subjects above.

---

### Stage B — synchronise and cut `Complete/`

#### 8. Combine every modality

```bash
python scripts/combine_all.py \
    --name <Subject> \
    --offsets_dir raw_data/offsets \
    [--specific_oms 1 3]
```

For each take this:

1. Asserts the five video/OpenPose lengths (V1, V2, BODY25_V1, BODY25_V2,
   BODY25_3D) agree to within 1 frame and takes the minimum as `target_len`.
   A larger spread raises — it means an upstream step drifted.
2. Decides whether the Vicon streams need resampling **by comparing lengths**, not
   by trusting fps metadata: as-is, `//2` (100 Hz capture), or `×2` (25 Hz capture,
   linear interpolation). If none of the three lands within 1 frame of the target
   it raises with a diagnostic.
3. Applies the offset from step 3 using the sign convention above, and cuts every
   stream to the common overlap.
4. Writes the trimmed videos (re-encoded frame by frame with OpenCV `mp4v`) and
   all `.npy` arrays into `Complete/<Subject>/OM<i>/`.

At this point every file in the take directory has identical frame count and the
dataset is usable.

---

### Stage C — instantaneous CoM

`CoM.npy` from stage A is Vicon's own centre-of-mass, which uses a session-averaged
thorax segment. The *instantaneous* CoM recomputes the thorax contribution
per-frame from C7 and the L5 top, and is what you want for balance/CoP analysis.

This needs the full Vicon segment structure (segment rotation-translation-scale
vectors, which `get_model_output.py` discards), so there is a separate extraction.

#### C1. Re-extract the full Vicon structure, already synced

```bash
python scripts/convert_model_output.py --persons <Subject> [--specific_oms 1 3]
```

Re-reads the raw Vicon JSON keeping **everything** (params, segments, all model
outputs with their component types, all markers), applies the same
resample-then-offset-slice logic as step 8, and writes one dict-valued
`Model_output/<Subject>/OM<i>.npy` per take.

It asserts its `CentreOfMass` matches `Complete/<Subject>/OM<i>/CoM.npy` — so
**step 8 must have run first**, and this doubles as a check that the two
independent sync implementations agree. If that file is absent the check is
skipped with a warning rather than failing.

Takes with no `CentreOfMass` in their model output are skipped with an
explanation; they have no segment poses for stage C2 to use.

The `Rz(-90°)` fix is applied only to outputs whose `componentNames` are `X,Y,Z`
*and* whose `componentTypes` are all spatial, plus the two CoM channels — segment
rotation vectors are correctly left alone.

#### C2. Compute the instantaneous CoM

```bash
python post_processing/instant_com.py \
    --model-root     Model_output \
    --gt-root        Complete \
    --inst-save-root Complete \
    --save-dir       om_comparison \
    [--subjects <Subject>] [--oms OM1 OM2]
```

Sums 15 segment contributions using standard anthropometric mass fractions and
proximal/distal ratios (pelvis 0.142, femur 0.100, tibia 0.0465, foot 0.0145,
humerus 0.028, radius 0.016, hand 0.006, thorax 0.355, head 0.081). Segment poses
come from the Vicon `<SEG>` rotation/translation/scale vectors via `bone_segment()`;
the foot uses the TOE↔HEE distance (minus the marker radius) as its scale. If any
input for a frame is invalid, that frame is written as `[0,0,0,0]`.

Outputs, per take:

- `Complete/<Subject>/OM<i>/CoM_inst.npy` `(N,4)`
- `Complete/<Subject>/OM<i>/CoM_floor_inst.npy` — same with `z` forced to 0
- `om_comparison/<Subject>/data/OM<i>_com_data.npy` — a dict with the instantaneous
  CoM, Vicon's CoM, the `Complete/` CoM, and mean/std/max Euclidean error between them
- `om_comparison/summary.npy` — all of the above stats keyed by subject/OM

Because `Model_output/` is already offset-cut, `CoM_inst.npy` is already aligned
and drops straight into `Complete/`. Set `--gt-root Complete` (the built-in
default points at an old absolute path that no longer exists).

`ViconCoMCalculator` in the same file reimplements Vicon's *session-averaged*
thorax approach; it exists to validate the port against Vicon's own numbers and is
not part of the normal run.

---


---

## Utilities

| Script | Purpose |
|---|---|
| `count.py` | Counts frames per OM activity across `Complete/` and emits `om_table.tex`. Excludes `Subject1` and `OM12`. Also holds the OM index → activity-name mapping (OM1 Circular Walking … OM11 Full-Body Stretches). |
| `move.py` | One-off restructure of OpenPose output from `OM<i>/json/Video_V*/` to `OM<i>/OM<i>_V*_jsons/`. Edit `base_dir` at the bottom before running. |
| `scripts/remove_bad_pixels.py` | **Superseded** by `post_processing/2_*` and `3_*`. It was the automatic faulty-sensor detector; it never persists corrected arrays, hardcodes per-subject thresholds for Keaton only (and raises `KeyError` for anyone else), and references a `sensor_configs.json` that does not exist. Kept for reference only — do not use. |

---
