"""
Single source of truth for where the pipeline's data lives.

Layout (relative to DATA_ROOT, which defaults to this repo's directory and can
be overridden with the OM_DATA_ROOT environment variable):

    raw_data/   <-> OneDrive   inputs: recordings + every hand-made file.
                               The pipeline only ever reads from here
                               (exception: step 0 converts videos in place).
    Complete/   <-> OneDrive   the release.
    work/       local only     regenerable intermediates (old `untrimmed/`).
    qc/         local only     QC videos, reports, comparisons.

Anything outside raw_data/ and Complete/ can be deleted and rebuilt.
Every script still accepts its old CLI flags; these are only the defaults.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("OM_DATA_ROOT", REPO_ROOT)).resolve()

# --- raw_data/: synced with OneDrive -------------------------------------
RAW = DATA_ROOT / "raw_data"
RAW_VIDEO = RAW / "Video"                    # <S>/OM<i>_V{1,2}.mp4
RAW_PRESSURE = RAW / "Pressure"              # <S>/OM<i>_{L,R}.csv
RAW_XCPS = RAW / "XCPs"                      # <S>/OM<i>.xcp
RAW_MODEL_OUTPUT = RAW / "Model_output"      # <S>/OM<i>.json   (Vicon export)
RAW_SMPL = RAW / "SMPL"
RAW_MOCAP_MRK = RAW / "MOCAP_MRK"            # <S>/OM<i>.npy  uncropped markers, native Vicon rate (export_mocap_mrk.py)
RAW_OPENPOSE = RAW / "openpose"              # <S>/OM<i>.zip    (zipped JSONs)
OFFSETS = RAW / "offsets"                    # <S>/OM<i>.json   hand: syncher.py
FAULTY_SENSORS = RAW / "faulty_sensors.json"  # hand: post_processing/2_*
SUBJECT_INFO = RAW / "subject_info.json"      # hand: mass/height/gender/age

# --- Complete/: synced with OneDrive -------------------------------------
COMPLETE = DATA_ROOT / "Complete"
COMPLETE_SUBJECT_INFO_NAME = "om_sub_info.json"  # copy of SUBJECT_INFO shipped in Complete/

# --- work/: local, regenerable ------------------------------------------
WORK = DATA_ROOT / "work"                    # BODY25/ Pressure/ Model_output/ live directly under it
WORK_OPENPOSE = WORK / "openpose"            # <S>/OM<i>/OM<i>_V{1,2}_jsons/  (unzipped)
WORK_BODY25 = WORK / "BODY25"
WORK_PRESSURE = WORK / "Pressure"            # <S>/OM<i>/Original_Pressure.npy
WORK_VICON = WORK / "Model_output"           # <S>/OM<i>/MOCAP_*.npy, CoM*.npy
VICON_FULL = WORK / "Model_output_full"      # <S>/OM<i>.npy  full synced Vicon dict (stage C1)

# --- qc/: local ----------------------------------------------------------
QC = DATA_ROOT / "qc"
QC_PRESSURE = QC / "pressure"                # <S>/pressure_viz, pressure_clean, ...
MANUAL_CHECK = QC / "manual_check"           # <S>/OM<i>/*.mp4
COM_COMPARISON = QC / "com_comparison"       # instant_com.py comparison output

# --- OneDrive ------------------------------------------------------------
REMOTE = os.environ.get("OM_REMOTE", "onedrive:Datasets/OrdinaryMovements")
SYNCED = {"raw_data": RAW, "Complete": COMPLETE}

