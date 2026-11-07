import os
import re
import subprocess
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import RAW_VIDEO  # noqa: E402

parser = argparse.ArgumentParser(description="Convert all videos to 50 fps if needed.")
parser.add_argument('--subject', type=str, default=None, help="Shorthand for --root_dir raw_data/Video/<S>")
parser.add_argument('--root_dir', type=str, default=None)

args = parser.parse_args()
root_dir = args.root_dir or (os.path.join(RAW_VIDEO, args.subject) if args.subject else None)
if root_dir is None:
    parser.error("give --subject or --root_dir")
log_file = os.path.join(root_dir, "conversion_log.json")

# Converting to 50 fps is destructive: the re-encoded file takes the original's
# name. Before that happens the untouched source is moved into this subdirectory,
# keeping its filename, so the data at the original frame rate is never lost.
ORIGINAL_FPS_DIRNAME = "original_fps"

# <take>_V1.mp4 / <take>_V2.mp4 for any take name (OM3, OM5_T2, Pick-up, ...)
pattern = re.compile(r"^(.+)_V[12]\.mp4$", re.IGNORECASE)

def get_fps(video_path):
    """Return the fps of a video using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        num, den = result.stdout.strip().split('/')
        return float(num) / float(den)
    except Exception as e:
        print(f"Error reading fps for {video_path}: {e}")
        return None

def process_video(video_path, out_path):
    """Convert video to 50 fps if needed. Returns (converted, original_fps)"""
    fps = get_fps(video_path)
    if fps is None:
        return False, None
   
    if 'pressure' in video_path.lower():
        print(f"Skipping {video_path} (pressure video)")
        return False, None
      
    if abs(fps - 50) < 0.01:  # already 50 fps
        print(f"Skipping {video_path} (already 50 fps)")
        return False, None
   
    print(f"Converting {video_path} ({fps:.2f} fps → 50 fps)")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-filter:v", "fps=50",
            "-c:a", "copy",  # keep audio unchanged
            out_path
        ],
        check=True
    )
    return True, fps

import shutil

# Dictionary to store conversion log
conversion_log = {}

for subdir, dirs, files in os.walk(root_dir):
    # Never descend into the backups we make below, or a second run would try to
    # "convert" the preserved originals and destroy them.
    dirs[:] = [d for d in dirs if d != ORIGINAL_FPS_DIRNAME]

    for file in files:
        if pattern.match(file):
            in_path = os.path.join(subdir, file)
            backup_dir = os.path.join(subdir, ORIGINAL_FPS_DIRNAME)
            backup_path = os.path.join(backup_dir, file)

            # An existing backup means a previous run already converted this take.
            # Bail out rather than overwrite it -- that copy is the only remaining
            # source at the original frame rate.
            if os.path.exists(backup_path):
                print(f"Skipping {in_path}: original already preserved at {backup_path}")
                continue

            out_name = file.split('_')[1]  # e.g. V1.mp4
            out_path = os.path.join(subdir, out_name)
            processed, original_fps = process_video(in_path, out_path)
          
            if not processed:
                print(f"Skipped {in_path}")
                continue
            else:
                print(f"Processed {in_path}")
                # Extract subject name and OM index for logging
                # Assumes structure like: root_dir/<subject_name>/...
                rel_path = os.path.relpath(in_path, root_dir)
                path_parts = Path(rel_path).parts
                
                # Extract the take name from the filename (e.g., OM1_V1.mp4 -> OM1)
                om_match = pattern.match(file)
                if om_match:
                    om_idx = om_match.group(1)
                    
                    # Create a relative path identifier for the log
                    log_key = str(Path(*path_parts[:-1]) / file)  # path relative to root_dir
                    
                    conversion_log[log_key] = {
                        "original_fps": round(original_fps, 2),
                        "new_fps": 50.0,
                        "om_index": om_idx,
                        "full_path": in_path,
                        "original_backup": backup_path
                    }
                
                # Preserve the untouched source at its original fps, then let the
                # re-encoded file take over the original name.
                os.makedirs(backup_dir, exist_ok=True)
                shutil.move(in_path, backup_path)
                print(f"  Original ({original_fps:.2f} fps) preserved at {backup_path}")
                shutil.move(out_path, os.path.join(subdir, file))

# Save conversion log to JSON file
if conversion_log:
    with open(log_file, 'w') as f:
        json.dump(conversion_log, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Conversion log saved to: {log_file}")
    print(f"Total videos converted: {len(conversion_log)}")
    print('='*60)
    
    # Print summary
    print("\nSummary of converted videos:")
    for video_path, info in conversion_log.items():
        print(f"  {info['om_index']}: {info['original_fps']} fps → {info['new_fps']} fps")
    print(f"\nOriginals preserved in '{ORIGINAL_FPS_DIRNAME}/' next to each converted video.")
else:
    print("\nNo videos were converted.")