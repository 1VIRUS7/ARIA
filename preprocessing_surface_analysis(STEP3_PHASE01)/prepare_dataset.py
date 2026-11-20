
from torchvision.datasets import OxfordIIITPet
from pathlib import Path
from collections import defaultdict
from PIL import Image
import random
import csv

# Configuration
ROOT = Path("dataset")
OUTPUT = ROOT / "clean"
NUM_IMAGES = 50
SEED = 42

random.seed(SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)

print("[+] Downloading Oxford-IIIT Pet Dataset...")

dataset = OxfordIIITPet(
    root=str(ROOT),
    split="trainval",
    target_types="category",
    download=True
)

# Group images by category
groups = defaultdict(list)

for index in range(len(dataset)):
    image, label = dataset[index]
    groups[int(label)].append(index)

# Shuffle each category
for indices in groups.values():
    random.shuffle(indices)

# Balanced round-robin selection
selected = []

while len(selected) < NUM_IMAGES:
    added = False

    for label in sorted(groups):
        if groups[label]:
            selected.append((groups[label].pop(), label))
            added = True

            if len(selected) == NUM_IMAGES:
                break

    if not added:
        break

# Save images and metadata
metadata_path = ROOT / "metadata.csv"

with open(metadata_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "dataset_label"])

    for number, (index, label) in enumerate(selected, start=1):
        image, _ = dataset[index]

        filename = f"image_{number:03d}.jpg"
        image.convert("RGB").save(OUTPUT / filename, quality=95)

        writer.writerow([filename, label])

        print(f"[+] Saved {filename} | Category ID: {label}")

print("\n================================")
print(f"Dataset preparation completed")
print(f"Total images: {len(selected)}")
print(f"Output: {OUTPUT}")
print(f"Metadata: {metadata_path}")
print("================================")
