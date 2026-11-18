
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Location of training dataset
dataset_path = Path("dataset/train")

# Six fruit categories
categories = [
    "freshapples",
    "rottenapples",
    "freshbanana",
    "rottenbanana",
    "freshoranges",
    "rottenoranges"
]

# Create a figure to display images
plt.figure(figsize=(12, 8))

for i, category in enumerate(categories):

    folder = dataset_path / category

    # Find image files
    images = []

    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        images.extend(folder.glob(extension))

    if not images:
        print("No images found in:", folder)
        continue

    # Read the first image
    image_path = images[0]
    image = cv2.imread(str(image_path))

    if image is None:
        print("Unable to read:", image_path)
        continue

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(category)
    plt.axis("off")

    print(category, ":", image.shape)

plt.tight_layout()
plt.show()
