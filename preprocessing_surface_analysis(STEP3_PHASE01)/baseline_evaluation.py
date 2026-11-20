
import csv
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_DIR = Path("dataset/clean")
METADATA_FILE = Path("dataset/metadata.csv")

OUTPUT_DIR = Path("comparison_results")
PLOTS_DIR = OUTPUT_DIR / "baseline_plots"

OUTPUT_CSV = OUTPUT_DIR / "baseline_predictions.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 60)
print("RESNET18 BASELINE EVALUATION")
print("=" * 60)

print(f"[+] Device: {DEVICE}")
print("[+] Loading ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)
model.to(DEVICE)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

print("[+] Model loaded successfully.")


# =====================================================
# LOAD DATASET METADATA
# =====================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = list(csv.DictReader(file))

if not metadata:
    raise ValueError("Dataset metadata is empty.")

print(f"[+] Total images: {len(metadata)}")


# =====================================================
# BASELINE EVALUATION
# =====================================================

results = []

print("\n[+] Starting baseline evaluation...\n")

for index, item in enumerate(metadata, start=1):

    filename = item["image"]
    dataset_label = item["dataset_label"]

    image_path = DATASET_DIR / filename

    try:

        image = Image.open(image_path).convert("RGB")

        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():

            logits = model(input_tensor)

            probabilities = torch.softmax(logits, dim=1)

            confidence, predicted = probabilities.max(dim=1)

            top5_prob, top5_indices = probabilities.topk(5, dim=1)

        predicted_id = predicted.item()
        confidence_value = confidence.item()

        top5 = []

        for class_id, probability in zip(
            top5_indices[0].tolist(),
            top5_prob[0].tolist()
        ):

            top5.append({
                "class_id": class_id,
                "class_name": categories[class_id],
                "confidence": probability
            })

        top5_string = " | ".join(
            f"{item['class_name']} ({item['confidence']:.6f})"
            for item in top5
        )

        result = {
            "image": filename,
            "dataset_label": dataset_label,
            "predicted_class_id": predicted_id,
            "predicted_class": categories[predicted_id],
            "confidence": confidence_value,
            "top5": top5_string,
            "top5_data": top5,
            "status": "SUCCESS"
        }

        results.append(result)

        print(
            f"[{index:02d}/{len(metadata)}] "
            f"{filename:<18} "
            f"Prediction: {categories[predicted_id]:<20} "
            f"Confidence: {confidence_value:.4f}"
        )

    except Exception as error:

        print(f"[!] Error processing {filename}: {error}")

        results.append({
            "image": filename,
            "dataset_label": dataset_label,
            "predicted_class_id": "",
            "predicted_class": "",
            "confidence": "",
            "top5": "",
            "top5_data": [],
            "status": f"ERROR: {error}"
        })


# =====================================================
# SAVE CSV
# =====================================================

fieldnames = [
    "image",
    "dataset_label",
    "predicted_class_id",
    "predicted_class",
    "confidence",
    "top5",
    "status"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in results:

        csv_result = {
            key: result[key]
            for key in fieldnames
        }

        writer.writerow(csv_result)

print(f"\n[+] CSV saved: {OUTPUT_CSV}")


# =====================================================
# PREPARE VISUALIZATION DATA
# =====================================================

valid_results = [
    result for result in results
    if result["status"] == "SUCCESS"
]

if not valid_results:
    raise RuntimeError("No valid predictions available for plotting.")

image_names = [
    result["image"]
    for result in valid_results
]

predictions = [
    result["predicted_class"]
    for result in valid_results
]

confidences = np.array([
    float(result["confidence"])
    for result in valid_results
])

prediction_counts = Counter(predictions)

prediction_labels = list(prediction_counts.keys())
prediction_values = list(prediction_counts.values())

top5_array = np.array([
    [
        item["confidence"]
        for item in result["top5_data"]
    ]
    for result in valid_results
])

image_indices = np.arange(1, len(valid_results) + 1)


# =====================================================
# PLOT 1: PREDICTION DISTRIBUTION
# =====================================================

plt.figure(figsize=(14, 7))

plt.bar(
    prediction_labels,
    prediction_values,
    edgecolor="black"
)

plt.title(
    "ResNet18 Prediction Distribution",
    fontsize=15
)

plt.xlabel("Predicted ImageNet Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=70, ha="right")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "prediction_distribution.png",
    dpi=300
)

plt.close()


# =====================================================
# PLOT 2: CONFIDENCE PER IMAGE
# =====================================================

plt.figure(figsize=(15, 6))

plt.bar(
    image_names,
    confidences,
    edgecolor="black"
)

plt.title(
    "ResNet18 Confidence per Image",
    fontsize=15
)

plt.xlabel("Image")
plt.ylabel("Confidence")

plt.ylim(0, 1)

plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "confidence_per_image.png",
    dpi=300
)

plt.close()


# =====================================================
# PLOT 3: CONFIDENCE HISTOGRAM
# =====================================================

plt.figure(figsize=(9, 6))

plt.hist(
    confidences,
    bins=10,
    edgecolor="black"
)

plt.title(
    "Confidence Distribution",
    fontsize=15
)

plt.xlabel("Confidence")
plt.ylabel("Number of Images")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "confidence_histogram.png",
    dpi=300
)

plt.close()


# =====================================================
# PLOT 4: TOP-5 CONFIDENCE
# =====================================================

plt.figure(figsize=(13, 6))

for rank in range(5):

    plt.plot(
        image_indices,
        top5_array[:, rank],
        marker="o",
        markersize=3,
        label=f"Top-{rank + 1}"
    )

plt.title(
    "Top-5 Prediction Confidence",
    fontsize=15
)

plt.xlabel("Image Index")
plt.ylabel("Confidence")

plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "top5_confidence.png",
    dpi=300
)

plt.close()


# =====================================================
# PLOT 5: BASELINE DASHBOARD
# =====================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(17, 12)
)

# Prediction distribution

axes[0, 0].bar(
    prediction_labels,
    prediction_values,
    edgecolor="black"
)

axes[0, 0].set_title("Prediction Distribution")
axes[0, 0].set_ylabel("Number of Images")

axes[0, 0].tick_params(
    axis="x",
    rotation=65
)

# Confidence histogram

axes[0, 1].hist(
    confidences,
    bins=10,
    edgecolor="black"
)

axes[0, 1].set_title("Confidence Distribution")
axes[0, 1].set_xlabel("Confidence")
axes[0, 1].set_ylabel("Number of Images")

# Confidence per image

axes[1, 0].bar(
    image_indices,
    confidences,
    edgecolor="black"
)

axes[1, 0].set_title("Confidence per Image")
axes[1, 0].set_xlabel("Image Index")
axes[1, 0].set_ylabel("Confidence")
axes[1, 0].set_ylim(0, 1)

# Top-5 confidence

for rank in range(5):

    axes[1, 1].plot(
        image_indices,
        top5_array[:, rank],
        marker="o",
        markersize=3,
        label=f"Top-{rank + 1}"
    )

axes[1, 1].set_title("Top-5 Confidence")
axes[1, 1].set_xlabel("Image Index")
axes[1, 1].set_ylabel("Confidence")

axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()

fig.suptitle(
    "ResNet18 Baseline Evaluation Dashboard",
    fontsize=18
)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "baseline_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================================
# FINAL SUMMARY
# =====================================================

successful = len(valid_results)
failed = len(results) - successful

print("\n" + "=" * 60)
print("BASELINE EVALUATION SUMMARY")
print("=" * 60)

print(f"Total images:       {len(metadata)}")
print(f"Successful:         {successful}")
print(f"Failed:             {failed}")

print(f"Average confidence: {np.mean(confidences):.4f}")
print(f"Minimum confidence: {np.min(confidences):.4f}")
print(f"Maximum confidence: {np.max(confidences):.4f}")

print("\nGenerated files:")

print(f"[+] {OUTPUT_CSV}")
print(f"[+] {PLOTS_DIR / 'prediction_distribution.png'}")
print(f"[+] {PLOTS_DIR / 'confidence_per_image.png'}")
print(f"[+] {PLOTS_DIR / 'confidence_histogram.png'}")
print(f"[+] {PLOTS_DIR / 'top5_confidence.png'}")
print(f"[+] {PLOTS_DIR / 'baseline_dashboard.png'}")

print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETED")
print("=" * 60)
