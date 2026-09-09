import shutil
from pathlib import Path

def reformat_openpose_dirs(base_dir):
    """
    In-place restructure:
      From:
        base_dir/OM10/json/Video_V1/*.json
        base_dir/OM10/json/Video_V2/*.json
      To:
        base_dir/OM10/OM10_V1_jsons/*.json
        base_dir/OM10/OM10_V2_jsons/*.json
    Deletes the old 'json/' directory afterwards.
    """
    base_dir = Path(base_dir)
    print(f"\n=== Starting in-place reformat under '{base_dir}' ===\n")

    for om_dir in sorted(base_dir.glob("OM*")):
        om_name = om_dir.name
        json_root = om_dir / "json"

        if not json_root.exists():
            print(f"⚠️  Skipping {om_name}: no 'json/' folder found.\n")
            continue

        print(f"📂 Processing {om_name}")
        video_dirs = sorted(json_root.glob("Video_*"))
        if not video_dirs:
            print(f"  ⚠️  No Video_* subdirs found in {json_root}\n")
            continue

        for video_dir in video_dirs:
            suffix = video_dir.name.split("_")[1]  # e.g., V1
            new_dir = om_dir / f"{om_name}_{suffix}_jsons"
            new_dir.mkdir(parents=True, exist_ok=True)

            json_files = list(video_dir.glob("*.json"))
            print(f"  ▶ Moving {len(json_files)} files from {video_dir.name} → {new_dir.name}")

            for f in json_files:
                shutil.move(str(f), new_dir / f.name)

        # After all moves, delete old json tree
        try:
            shutil.rmtree(json_root)
            print(f"  🗑️  Deleted old directory: {json_root}\n")
        except Exception as e:
            print(f"  ⚠️  Could not delete {json_root}: {e}\n")

    print("✅ Done! All OpenPose JSON directories reformatted.\n")

if __name__ == "__main__":
    base_dir = "/mnt/e/Data/OMs/construct_complete/raw_data/openpose_output/Spencer"
    reformat_openpose_dirs(base_dir)
