import os
import shutil

# Path to the folder containing behavior subfolders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script lives
RAW_DIR = os.path.join(SCRIPT_DIR, "caa_datasets", "raw")

if not os.path.exists(RAW_DIR):
    raise FileNotFoundError(f"Cannot find the raw folder at {RAW_DIR}")

# Move each dataset.json up and rename it to <behavior>.json
for root, dirs, files in os.walk(RAW_DIR):
    for file in files:
        if file == "dataset.json":
            behavior = os.path.basename(root)
            src = os.path.join(root, file)
            dst = os.path.join(RAW_DIR, f"{behavior}.json")
            print(f"Moving {src} -> {dst}")
            shutil.move(src, dst)

# Optional: remove empty subfolders
for subfolder in os.listdir(RAW_DIR):
    path = os.path.join(RAW_DIR, subfolder)
    if os.path.isdir(path):
        try:
            os.rmdir(path)
            print(f"Removed empty folder {path}")
        except OSError:
            pass  # folder not empty, ignore

