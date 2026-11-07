import os
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_paths import RAW, WORK, RAW_OPENPOSE, WORK_OPENPOSE  # noqa: E402
from openpose_zip import unpack_subject  # noqa: E402
import takes  # noqa: E402

def count_video_frames(video_path: str) -> int:
    """Return number of frames in a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def load_openpose_json(json_path: str) -> np.ndarray:
    """Load one OpenPose JSON file and return (25, 3) array (x, y, conf)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    people = data.get("people", [])
    if len(people) == 0:  # no person detected
        return np.zeros((25, 3), dtype=np.float32)
    # Only take the first person
    keypoints = people[0]["pose_keypoints_2d"]
    arr = np.array(keypoints, dtype=np.float32).reshape((-1, 3))  # (25, 3)
    return arr

def process_subject(raw_data_dir: str, name: str, output_dir: str, specific_oms=None,
                    openpose_root=WORK_OPENPOSE, openpose_zips=RAW_OPENPOSE):
    """
    Process one subject's OpenPose data.
    
    Args:
        raw_data_dir: Root directory containing raw data
        name: Subject name (e.g., 'Kyle', 'Keaton')
        output_dir: Output directory for BODY25 .npy files
        specific_oms: Only these takes: numbers and/or names (e.g. [1, 3, 'Pick-up'])
        openpose_root: Unzipped OpenPose JSONs, <root>/<name>/OM<i>/OM<i>_V*_jsons/
        openpose_zips: Per-take zips; any take missing from openpose_root is extracted from here
    """
    # Paths
    unpack_subject(name, openpose_zips, openpose_root, oms=specific_oms)
    openpose_dir = Path(openpose_root) / name
    video_dir = Path(raw_data_dir) / "Video" / name
    
    if not openpose_dir.exists():
        print(f"Warning: OpenPose directory not found: {openpose_dir}")
        return
    
    if not video_dir.exists():
        print(f"Warning: Video directory not found: {video_dir}")
        return
    
    selection = takes.parse_selection(specific_oms)
    for om_dir in takes.take_dirs(openpose_dir):
        om_idx = om_dir.name  # e.g., "OM1"
        print(f"Processing {name}/{om_idx}")
        
        # Find all *_jsons directories in this OM folder
        json_subdirs = sorted([d for d in om_dir.iterdir() if d.is_dir() and d.name.endswith("_jsons")])
      
        if not takes.selected(om_idx, selection):
            print(f"  Skipping {om_idx} as it's not in the specified list")
            continue
        
        if not json_subdirs:
            print(f"  Warning: No *_jsons directories found in {om_dir}")
            continue
        
        # Create output directory for this OM
        om_output_dir = Path(output_dir) / "BODY25" / name / om_idx
        om_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each view (V1, V2)
        for json_subdir in json_subdirs:
            # Extract view name (e.g., "OM1_V1_jsons" -> "V1")
            subdir_name = json_subdir.name
            
            # Determine version (V1 or V2)
            if subdir_name.upper().endswith("_V1_JSONS"):
                version = "V1"
                video_name = f"{om_idx}_V1.mp4"
            elif subdir_name.upper().endswith("_V2_JSONS"):
                version = "V2"
                video_name = f"{om_idx}_V2.mp4"
            else:
                print(f"  Warning: Cannot determine view version from {subdir_name}")
                continue
            
            # Find corresponding video
            video_path = video_dir / video_name
            if not video_path.exists():
                print(f"  Warning: Video not found: {video_path}")
                continue
            
            # Count frames in video
            video_frame_count = count_video_frames(str(video_path))
            
            # Collect JSON files
            json_files = sorted([f for f in json_subdir.iterdir() if f.suffix == ".json"])
            json_frame_count = len(json_files)
            
            # Check frame count mismatch
            frame_diff = abs(video_frame_count - json_frame_count)
            
            if frame_diff > 1:
                # Significant mismatch - raise error
                error_msg = (
                    f"ERROR: Frame count mismatch for {name}/{om_idx}/{version}:\n"
                    f"  Video has {video_frame_count} frames\n"
                    f"  Found {json_frame_count} JSON files\n"
                    f"  Difference: {frame_diff} frames (exceeds threshold of 1)"
                )
                print(f"  {error_msg}")
                raise ValueError(error_msg)
            
            elif frame_diff == 1:
                # Off by one - use shorter length
                num_frames = min(video_frame_count, json_frame_count)
                print(f"  Note: Frame count off by 1 for {version}:")
                print(f"    Video: {video_frame_count}, JSON: {json_frame_count}")
                print(f"    Using {num_frames} frames (shorter of the two)")
                json_files = json_files[:num_frames]
            
            else:
                # Perfect match
                num_frames = video_frame_count
                print(f"  Frame counts match: {num_frames} frames")
            
            # Process JSONs → NPY
            all_keypoints = np.zeros((num_frames, 25, 3), dtype=np.float32)
            for i, json_file in enumerate(json_files):
                all_keypoints[i] = load_openpose_json(str(json_file))
            
            # Save NPY
            npy_name = f"BODY25_{version}.npy"
            npy_path = om_output_dir / npy_name
            np.save(npy_path, all_keypoints)
            print(f"  ✓ Saved {npy_path} with shape {all_keypoints.shape}")

def main():
    parser = argparse.ArgumentParser(description="Convert OpenPose JSONs to BODY25 .npy files")
    parser.add_argument("--raw_data_dir", type=str, default=RAW,
                        help="Root directory containing raw data (with Video/ subdir)")
    parser.add_argument("--openpose_dir", type=str, default=WORK_OPENPOSE,
                        help="Unzipped OpenPose JSONs (<dir>/<name>/OM<i>/OM<i>_V*_jsons/)")
    parser.add_argument("--openpose_zips", type=str, default=RAW_OPENPOSE,
                        help="Per-take OpenPose zips, extracted into --openpose_dir when missing")
    parser.add_argument("--name", type=str, required=True,
                        help="Subject name (e.g., 'Kyle', 'Keaton')")
    parser.add_argument("--output_dir", type=str, default=WORK,
                        help="Output directory (will create BODY25/name/OM<idx>/ structure)")
    parser.add_argument('--specific_oms', nargs='+', default=None,
                        help="Only these takes: numbers and/or names (e.g. 1 3 Pick-up)")
    args = parser.parse_args()
    
    try:
        process_subject(args.raw_data_dir, args.name, args.output_dir, specific_oms=args.specific_oms,
                        openpose_root=args.openpose_dir, openpose_zips=args.openpose_zips)
        print("\n✓ Processing complete!")
    except ValueError as e:
        print(f"\n✗ Processing failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()