import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import RAW, WORK, COMPLETE, OFFSETS, SUBJECT_INFO, COMPLETE_SUBJECT_INFO_NAME  # noqa: E402

def upsample_linear(data):
    """Upsample data by 2x using linear interpolation.
    Works for both (N, J, D) and (N, D) arrays."""
    if data.ndim == 3:
        N, J, D = data.shape
        upsampled = np.empty((2*N, J, D), dtype=data.dtype)
    elif data.ndim == 2:
        N, D = data.shape
        upsampled = np.empty((2*N, D), dtype=data.dtype)
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    
    # copy originals at even indices
    upsampled[0::2] = data
    # interpolate at odd indices
    upsampled[1:-1:2] = (data[:-1] + data[1:]) / 2
    # last frame duplication
    upsampled[-1] = data[-1]
    
    return upsampled

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def trim_video(input_path, output_path, start, end):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, end):
        ret, frame = cap.read()
        if not ret: 
            break
        out.write(frame)
    cap.release()
    out.release()

def process_subject(raw_data_dir, work_dir, name, save_root, offsets_dir, specific_oms=None):
    """
    Combine all multimodal data for a subject.
    
    Args:
        raw_data_dir: Directory containing raw Video files
        work_dir: Directory containing BODY25, Pressure, Model_output
        name: Subject name
        save_root: Output directory for combined data
        offsets_dir: Directory containing offset JSON files
    """
    # Input directories
    video_dir = Path(raw_data_dir) / "Video" / name
    body25_dir = Path(work_dir) / "BODY25" / name
    pressure_dir = Path(work_dir) / "Pressure" / name
    vicon_data_dir = Path(work_dir) / "Model_output" / name
    
    # Get all OM directories from BODY25 (since that's where we know processed data exists)
    om_dirs = sorted([d for d in body25_dir.iterdir() if d.is_dir() and d.name.startswith("OM")])
    
    for om_dir in om_dirs:
        om_idx = om_dir.name
        om_idx_int = int(om_idx.replace("OM", ""))
        if specific_oms is not None and om_idx_int not in specific_oms:
            print(f"Skipping {om_idx} as it is not in the specified list")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {name}/{om_idx}")
        print('='*60)
        
        save_dir = Path(save_root) / name / om_idx
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Load offset config ---
        offset_file = Path(offsets_dir) / name / f"{om_idx}.json"
        if not offset_file.exists():
            # Try alternative naming
            offset_file_alt = Path(offsets_dir) / name / f"{om_idx}_cfg.json"
            if not offset_file_alt.exists():
                raise FileNotFoundError(f"Offset file not found: {offset_file} or {offset_file_alt}")
            # Rename to standard format
            offset_file.parent.mkdir(parents=True, exist_ok=True)
            offset_file_alt.rename(offset_file)
        
        with open(offset_file, "r") as f:
            config = json.load(f)
        
        offset = config["offset"]
        original_video_fps = config.get("original_video_fps", 50.0)
        
        print(f"  Offset: {offset}")
        print(f"  Original video fps: {original_video_fps}")
        
        # --- Load video files ---
        v1 = video_dir / f"{om_idx}_V1.mp4"
        v2 = video_dir / f"{om_idx}_V2.mp4"
        
        if not v1.exists():
            raise FileNotFoundError(f"Video not found: {v1}")
        if not v2.exists():
            raise FileNotFoundError(f"Video not found: {v2}")
        
        # --- Load BODY25 arrays ---
        b1 = np.load(body25_dir / om_idx / "BODY25_V1.npy")
        b2 = np.load(body25_dir / om_idx / "BODY25_V2.npy")
        b3d = np.load(body25_dir / om_idx / "BODY25_3D.npy")
        
        # --- Get video frame counts ---
        v1_len, fps = get_video_info(v1)
        v2_len, _ = get_video_info(v2)
        
        print(f"  V1 video length: {v1_len}")
        print(f"  V2 video length: {v2_len}")
        print(f"  BODY25_V1 length: {b1.shape[0]}")
        print(f"  BODY25_V2 length: {b2.shape[0]}")
        print(f"  BODY25_3D length: {b3d.shape[0]}")
        
        # --- Check if all lengths are within ±1 of each other ---
        all_video_body_lengths = [v1_len, v2_len, b1.shape[0], b2.shape[0], b3d.shape[0]]
        max_len = max(all_video_body_lengths)
        min_len = min(all_video_body_lengths)
        
        if max_len - min_len <= 1:
            # All within ±1, use the minimum
            target_len = min_len
            print(f"  → All video/BODY25 lengths within ±1 frame")
            print(f"  → Using minimum length: {target_len}")
        elif max_len - min_len > 1:
            # Differences too large
            raise ValueError(
                f"Video/BODY25 lengths differ by more than 1 frame:\n"
                f"  V1 video: {v1_len}\n"
                f"  V2 video: {v2_len}\n"
                f"  BODY25_V1: {b1.shape[0]}\n"
                f"  BODY25_V2: {b2.shape[0]}\n"
                f"  BODY25_3D: {b3d.shape[0]}\n"
                f"  Max difference: {max_len - min_len}"
            )
        
        print(f"  Target length (for Vicon): {target_len}")
        
        # --- Load Vicon files (every one of them optional) ---
        # A take that was labelled but never run through Plug-in-Gait has markers
        # and nothing else; a take with no Vicon export at all has none of them.
        # Neither case is fatal -- the video / pressure / OpenPose bundle is still
        # worth building, so students can sync every take even when the mocap
        # side is incomplete.
        VICON_FILES = ["MOCAP_3D.npy", "MOCAP_MRK.npy", "CoM.npy", "CoM_floor.npy",
                       "CoM_inst.npy", "CoM_floor_inst.npy"]
        vicon = {}
        for fname in VICON_FILES:
            fpath = vicon_data_dir / om_idx / fname
            if fpath.exists():
                vicon[fname] = np.load(fpath)

        if vicon:
            lengths = {f: a.shape[0] for f, a in vicon.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"Vicon arrays have differing lengths for {name}/{om_idx}: {lengths}"
                )
            vicon_len = next(iter(lengths.values()))
            print(f"  Vicon streams present ({len(vicon)}): {', '.join(sorted(vicon))}")
            print(f"  Vicon raw length: {vicon_len}")
            absent = [f for f in VICON_FILES if f not in vicon]
            if absent:
                print(f"  [!] Vicon streams absent: {', '.join(absent)}")
        else:
            vicon_len = None
            print(f"  [!] No Vicon data found in {vicon_data_dir / om_idx}")
            print(f"  [!] Building video / pressure / OpenPose only for this take.")

        # --- Intelligently determine if Vicon needs resampling ---
        if vicon_len is None:
            others_available = target_len
            print(f"  Video/OpenPose available frames: {others_available}")
        else:
            needs_resampling = False
            resample_operation = None
            # Check if Vicon already matches target (within ±1 frame)
            if abs(vicon_len - target_len) <= 1:
                print(f"  ✓ Vicon length already matches target (±1 frame), no resampling needed")
                needs_resampling = False

            # Check if downsampling by 2 would match target (use integer division)
            elif abs((vicon_len // 2) - target_len) <= 1:
                print(f"  → Vicon needs downsampling by 2x: {vicon_len} → ~{target_len}")
                needs_resampling = True
                resample_operation = "downsample"

            # Check if upsampling by 2 would match target
            elif abs(vicon_len * 2 - target_len) <= 1:
                print(f"  → Vicon needs upsampling by 2x: {vicon_len} → ~{target_len}")
                needs_resampling = True
                resample_operation = "upsample"

            else:
                # Cannot determine resampling strategy
                raise ValueError(
                    f"Cannot determine Vicon resampling strategy:\n"
                    f"  Vicon length: {vicon_len}\n"
                    f"  Target length (Video/BODY25): {target_len}\n"
                    f"  Vicon as-is: diff = {abs(vicon_len - target_len)} (needs ≤1)\n"
                    f"  Vicon/2: {vicon_len//2}, diff = {abs((vicon_len//2) - target_len)} (needs ≤1)\n"
                    f"  Vicon*2: {vicon_len*2}, diff = {abs(vicon_len*2 - target_len)} (needs ≤1)\n"
                    f"  Original video fps: {original_video_fps}"
                )

            # --- Apply resampling if needed ---
            if needs_resampling:
                if resample_operation == "downsample":
                    print(f"  Downsampling Vicon data by 2x...")
                    vicon = {f: a[::2][:target_len] for f, a in vicon.items()}
                elif resample_operation == "upsample":
                    print(f"  Upsampling Vicon data by 2x...")
                    vicon = {f: upsample_linear(a) for f, a in vicon.items()}

                print(f"  Vicon resampled length: {next(iter(vicon.values())).shape[0]}")

            # --- Verify all Vicon arrays still match after resampling ---
            lengths = {f: a.shape[0] for f, a in vicon.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"After resampling, Vicon arrays have differing lengths: {lengths}"
                )

            # --- Final validation: Vicon should now match target within ±1 ---
            vicon_final_len = next(iter(lengths.values()))
            if abs(vicon_final_len - target_len) > 1:
                raise ValueError(
                    f"After resampling, Vicon length ({vicon_final_len}) still doesn't match "
                    f"target length ({target_len}) within ±1 frame"
                )

            # --- Use minimum of all lengths (target_len and vicon_final_len) ---
            others_available = min(target_len, vicon_final_len)

            print(f"  ✓ All data lengths match (±1 frame)")
            print(f"  Video/Mocap/OpenPose available frames: {others_available}")


        # --- Load Pressure file ---
        pressure = np.load(pressure_dir / om_idx / "Original_Pressure.npy")
        print(f"  Pressure length: {pressure.shape[0]}")
        
        # --- Apply offset trimming ---
        # Offset represents how many frames video/mocap started BEFORE pressure
        # Most common case: offset is negative (e.g., -50)
        #   - This means video/mocap started 50 frames before pressure recording began
        #   - We skip the first 50 frames of video/mocap to align with pressure start
        #   - We use all pressure frames starting from frame 0
        
        if offset < 0:
            # Video/mocap started BEFORE pressure (typical case)
            # Skip abs(offset) frames from beginning of video/mocap
            # Use all of pressure starting from frame 0
            others_start = abs(offset)
            pressure_start = 0
            print(f"  → Video/mocap started {abs(offset)} frames before pressure")
            print(f"  → Skipping first {others_start} frames of video/mocap")
            
        else:
            # Pressure started BEFORE video/mocap (rare case)
            # Skip offset frames from beginning of pressure
            # Use all of video/mocap starting from frame 0
            pressure_start = offset
            others_start = 0
            print(f"  → Pressure started {offset} frames before video/mocap")
            print(f"  → Skipping first {pressure_start} frames of pressure")
        
        # Calculate how many frames are available after applying offset
        others_remaining = others_available - others_start
        pressure_remaining = pressure.shape[0] - pressure_start
        
        # Final length is limited by whichever runs out first
        final_len = min(others_remaining, pressure_remaining)
        
        # Calculate end indices
        others_end = others_start + final_len
        pressure_end = pressure_start + final_len
       
        print(f"  Video/Mocap/OpenPose trim: [{others_start}:{others_end}] ({final_len} frames)")
        print(f"  Pressure trim: [{pressure_start}:{pressure_end}] ({final_len} frames)")
        print(f"  Final synced length: {final_len}")
        
        # --- Trim and save data ---
        print(f"  Saving data...")
        
        # Videos - trim to others_end (which is ≤ target_len)
        trim_video(v1, save_dir / "Video_V1.mp4", others_start, others_end)
        trim_video(v2, save_dir / "Video_V2.mp4", others_start, others_end)
        
        # BODY25 - trim to others_end (which is ≤ target_len)
        np.save(save_dir / "BODY25_V1.npy", b1[others_start:others_end])
        np.save(save_dir / "BODY25_V2.npy", b2[others_start:others_end])
        np.save(save_dir / "BODY25_3D.npy", b3d[others_start:others_end])
        
        # Pressure
        np.save(save_dir / "Pressure.npy", pressure[pressure_start:pressure_end])
        
        # Vicon (already resampled) - only whatever this take actually has
        for fname, arr in sorted(vicon.items()):
            np.save(save_dir / fname, arr[others_start:others_end])

        # A stale file from an earlier run would silently look like real data, so
        # drop any Vicon stream this take no longer has.
        for fname in VICON_FILES:
            if fname not in vicon and (save_dir / fname).exists():
                (save_dir / fname).unlink()
                print(f"  Removed stale {fname} (not present for this take)")

        # --- Manifest: what this take actually contains ---
        # Downstream code and students consuming Complete/ should not have to
        # guess which streams exist for a given take.
        streams = ["Video_V1.mp4", "Video_V2.mp4", "BODY25_V1.npy",
                   "BODY25_V2.npy", "BODY25_3D.npy", "Pressure.npy"] + sorted(vicon)
        manifest = {
            "subject": name,
            "om": om_idx,
            "frames": int(final_len),
            "fps": 50,
            "offset": offset,
            "streams": streams,
            "vicon_complete": all(f in vicon for f in
                                  ["MOCAP_3D.npy", "MOCAP_MRK.npy", "CoM.npy", "CoM_floor.npy"]),
            "missing_vicon": [f for f in VICON_FILES if f not in vicon],
        }
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  ✓ Successfully saved {final_len} frames to {save_dir}")
        if not manifest["vicon_complete"]:
            print(f"  [!] PARTIAL take: no {', '.join(manifest['missing_vicon'])}")

def main():
    parser = argparse.ArgumentParser(description="Trim and align multimodal motion capture data")
    parser.add_argument("--raw_data_dir", default=RAW, 
                        help="Directory containing raw Video files")
    parser.add_argument("--work_dir", "--untrimmed_dir", dest="work_dir", default=WORK,
                        help="Directory containing BODY25/Pressure/Model_output")
    parser.add_argument("--name", required=True, 
                        help="Subject name (e.g., 'Kyle', 'Keaton')")
    parser.add_argument("--save_root", default=COMPLETE, 
                        help="Output directory for combined synced data")
    parser.add_argument("--offsets_dir", default=OFFSETS, 
                        help="Directory containing offset JSON files")
    parser.add_argument('--specific_oms', nargs='+', default=None, type=int, 
                        help="Process only specific OM directories (e.g., OM1 OM3)")
    args = parser.parse_args()
    
    process_subject(args.raw_data_dir, args.work_dir, args.name, 
                   args.save_root, args.offsets_dir, specific_oms=args.specific_oms)

    # Ship the hand-maintained subject metadata with the release.
    if Path(SUBJECT_INFO).exists():
        dst = Path(args.save_root) / COMPLETE_SUBJECT_INFO_NAME
        shutil.copyfile(SUBJECT_INFO, dst)
        print(f"Copied {SUBJECT_INFO} -> {dst}")

if __name__ == "__main__":
    main()
