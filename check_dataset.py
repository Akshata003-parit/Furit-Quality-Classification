
from pathlib import Path

# Dataset folder location
dataset_path = Path("dataset")

# Check whether dataset folder exists
if not dataset_path.exists():
    print("Dataset folder not found!")
else:
    print("DATASET INSPECTION")
    print("------------------")

    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    total_images = 0

    # Check every subfolder
    for folder in sorted(dataset_path.rglob("*")):

        if folder.is_dir():

            images = [
                file for file in folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in image_extensions
            ]

            if images:
                count = len(images)
                total_images += count

                print(folder, ":", count, "images")

    print("------------------")
    print("Total images:", total_images)
