
import csv
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.models import resnet18, ResNet18_Weights


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent

CLEAN_DIR = ROOT / "dataset" / "clean"
FGSM_DIR = ROOT / "dataset" / "adversarial" / "FGSM"
EOT_DIR = ROOT / "dataset" / "adversarial" / "EOT-PGD"

BASELINE_CSV = ROOT / "comparison_results" / "baseline_predictions.csv"

OUTPUT_DIR = ROOT / "comparison_results" / "preprocessing_evaluation"
PLOTS_DIR = OUTPUT_DIR / "plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTACK_DIRS = {
    "FGSM": FGSM_DIR,
    "EOT-PGD": EOT_DIR,
}


# =========================================================
# MODEL
# =========================================================

def load_model():
    print("[+] Loading ResNet18...")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(DEVICE)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    print(f"[+] Device: {DEVICE}")

    return model, preprocess, categories


@torch.no_grad()
def predict(image, model, preprocess, categories):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)

    confidence, prediction = probabilities.max(dim=1)

    class_id = int(prediction.item())

    return {
        "class_id": class_id,
        "class_name": categories[class_id],
        "confidence": float(confidence.item()),
    }


# =========================================================
# BASELINE PSEUDO-LABELS
# =========================================================

def load_baseline_labels():
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {BASELINE_CSV}")

    with open(BASELINE_CSV, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Baseline CSV is empty.")

    print("[+] Baseline CSV columns:", list(rows[0].keys()))

    def find_column(candidates):
        for candidate in candidates:
            if candidate in rows[0]:
                return candidate
        return None

    image_col = find_column(["image", "filename", "file"])
    class_col = find_column([
        "predicted_class_id",
        "prediction_id",
        "predicted_id",
        "class_id",
        "prediction"
    ])

    if not image_col or not class_col:
        raise ValueError(
            "Could not identify image/prediction columns in baseline CSV. "
            "Check the printed column names and update the aliases."
        )

    labels = {}

    for row in rows:
        image_name = Path(row[image_col]).name
        labels[Path(image_name).stem] = int(row[class_col])

    return labels


# =========================================================
# PREPROCESSING TRANSFORMATIONS
# =========================================================

def jpeg_transform(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def realistic_resize(image, short_side=1000):
    width, height = image.size

    scale = short_side / min(width, height)

    new_size = (
        round(width * scale),
        round(height * scale)
    )

    return image.resize(new_size, Image.Resampling.BICUBIC)


def realistic_center_crop(image, size=200):
    image = realistic_resize(image, short_side=256)

    width, height = image.size

    left = (width - size) // 2
    top = (height - size) // 2

    return image.crop((
        left,
        top,
        left + size,
        top + size
    ))


def platform_recompression(
    image,
    max_side=384,
    first_quality=90,
    platform_quality=75
):
    image = jpeg_transform(image, first_quality)

    width, height = image.size

    scale = min(1.0, max_side / max(width, height))

    if scale < 1:
        image = image.resize(
            (
                round(width * scale),
                round(height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    return jpeg_transform(image, platform_quality)


TRANSFORMATIONS = {
    "raw": lambda image: image.copy(),

    "resize_short_1000": lambda image:
        realistic_resize(image, short_side=1000),

    "center_crop_200": lambda image:
        realistic_center_crop(image, size=200),

    "jpeg_q75": lambda image:
        jpeg_transform(image, quality=75),

    "jpeg_q50": lambda image:
        jpeg_transform(image, quality=50),

    "platform_q90_q75": lambda image:
        platform_recompression(
            image,
            max_side=384,
            first_quality=90,
            platform_quality=75
        ),

    "platform_q85_q60": lambda image:
        platform_recompression(
            image,
            max_side=384,
            first_quality=85,
            platform_quality=60
        ),
}


# =========================================================
# PERTURBATION METRICS
# =========================================================

def calculate_metrics(clean_image, adversarial_image):
    clean = np.asarray(clean_image, dtype=np.float32) / 255.0
    adversarial = np.asarray(adversarial_image, dtype=np.float32) / 255.0

    if clean.shape != adversarial.shape:
        raise ValueError(
            f"Image shape mismatch: {clean.shape} vs {adversarial.shape}"
        )

    delta = np.abs(adversarial - clean)

    return {
        "mae": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "max_delta": float(np.max(delta)),
    }


# =========================================================
# CSV OUTPUT
# =========================================================

def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, preprocess, categories, baseline_labels):
    detailed_rows = []

    # Cache clean transformed images and their predictions.
    clean_cache = {}

    clean_drift_stats = defaultdict(list)

    print("\n[+] Evaluating clean images...")

    for clean_path in sorted(CLEAN_DIR.glob("*")):
        if not clean_path.is_file():
            continue

        stem = clean_path.stem

        if stem not in baseline_labels:
            print(f"[!] Missing baseline label for {clean_path.name}")
            continue

        original_label = baseline_labels[stem]

        with Image.open(clean_path) as source:
            clean_original = source.convert("RGB")

        for transform_name, transform_function in TRANSFORMATIONS.items():
            transformed_clean = transform_function(clean_original)

            clean_prediction = predict(
                transformed_clean,
                model,
                preprocess,
                categories
            )

            clean_drift = (
                clean_prediction["class_id"] != original_label
            )

            clean_cache[(stem, transform_name)] = (
                transformed_clean,
                clean_prediction
            )

            clean_drift_stats[transform_name].append(clean_drift)

    print("[+] Evaluating adversarial images...")

    total_cases = len(clean_cache) * len(ATTACK_DIRS)

    for index, ((stem, transform_name), cache_item) in enumerate(
        clean_cache.items(), start=1
    ):
        transformed_clean, clean_prediction = cache_item

        original_label = baseline_labels[stem]

        clean_path = next(CLEAN_DIR.glob(stem + ".*"))

        for attack_name, attack_dir in ATTACK_DIRS.items():
            adversarial_path = attack_dir / f"{stem}.png"

            if not adversarial_path.exists():
                print(f"[!] Missing: {adversarial_path}")
                continue

            with Image.open(adversarial_path) as source:
                adversarial_original = source.convert("RGB")

            adversarial_transformed = TRANSFORMATIONS[
                transform_name
            ](adversarial_original)

            adversarial_prediction = predict(
                adversarial_transformed,
                model,
                preprocess,
                categories
            )

            conditional_success = (
                adversarial_prediction["class_id"]
                != clean_prediction["class_id"]
            )

            fixed_reference_success = (
                adversarial_prediction["class_id"] != original_label
            )

            metrics = calculate_metrics(
                transformed_clean,
                adversarial_transformed
            )

            detailed_rows.append({
                "image": clean_path.name,
                "attack": attack_name,
                "transformation": transform_name,

                "original_pseudo_label_id": original_label,

                "clean_prediction_id": clean_prediction["class_id"],
                "clean_prediction_name": clean_prediction["class_name"],
                "clean_confidence": clean_prediction["confidence"],

                "adversarial_prediction_id": adversarial_prediction["class_id"],
                "adversarial_prediction_name": adversarial_prediction["class_name"],
                "adversarial_confidence": adversarial_prediction["confidence"],

                "clean_prediction_drift": (
                    clean_prediction["class_id"] != original_label
                ),

                "conditional_attack_success": conditional_success,
                "fixed_reference_success": fixed_reference_success,

                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "max_delta": metrics["max_delta"],

                "status": "success",
            })

        if index % 10 == 0:
            print(
                f"    Processed {index}/{len(clean_cache)} "
                f"image-transformation pairs..."
            )

    return detailed_rows, clean_drift_stats


# =========================================================
# SUMMARY
# =========================================================

def create_summary(detailed_rows, clean_drift_stats):
    grouped = defaultdict(list)

    for row in detailed_rows:
        key = (row["attack"], row["transformation"])
        grouped[key].append(row)

    summary_rows = []

    for (attack, transform), rows in sorted(grouped.items()):
        count = len(rows)

        conditional_successes = sum(
            row["conditional_attack_success"] for row in rows
        )

        fixed_valid = [
            row for row in rows
            if not row["clean_prediction_drift"]
        ]

        fixed_successes = sum(
            row["fixed_reference_success"] for row in fixed_valid
        )

        drift_values = clean_drift_stats[transform]

        summary_rows.append({
            "attack": attack,
            "transformation": transform,
            "images_evaluated": count,

            "clean_drift_count": sum(drift_values),
            "clean_drift_rate_percent": (
                100 * np.mean(drift_values) if drift_values else 0
            ),

            "conditional_success_count": conditional_successes,
            "conditional_asr_percent": (
                100 * conditional_successes / count if count else 0
            ),

            "fixed_reference_valid_count": len(fixed_valid),
            "fixed_reference_success_count": fixed_successes,
            "fixed_reference_asr_percent": (
                100 * fixed_successes / len(fixed_valid)
                if fixed_valid else 0
            ),

            "average_mae": float(np.mean([r["mae"] for r in rows])),
            "average_rmse": float(np.mean([r["rmse"] for r in rows])),
            "average_max_delta": float(
                np.mean([r["max_delta"] for r in rows])
            ),
        })

    return summary_rows


# =========================================================
# PLOTS
# =========================================================

def make_plots(summary_rows, clean_drift_stats):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    transformations = list(TRANSFORMATIONS.keys())
    x = np.arange(len(transformations))
    width = 0.35

    def get_values(attack, field):
        lookup = {
            row["transformation"]: row[field]
            for row in summary_rows
            if row["attack"] == attack
        }

        return [lookup.get(name, 0) for name in transformations]

    fgsm_asr = get_values("FGSM", "conditional_asr_percent")
    eot_asr = get_values("EOT-PGD", "conditional_asr_percent")

    fgsm_mae = get_values("FGSM", "average_mae")
    eot_mae = get_values("EOT-PGD", "average_mae")

    fgsm_rmse = get_values("FGSM", "average_rmse")
    eot_rmse = get_values("EOT-PGD", "average_rmse")

    drift = [
        100 * np.mean(clean_drift_stats[name])
        if clean_drift_stats[name] else 0
        for name in transformations
    ]

    def grouped_bar(values1, values2, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.bar(x - width / 2, values1, width, label="FGSM")
        ax.bar(x + width / 2, values2, width, label="EOT-PGD")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(transformations, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)

    grouped_bar(
        fgsm_asr,
        eot_asr,
        "Conditional Attack Success Rate",
        "ASR (%)",
        "asr_per_transformation.png"
    )

    grouped_bar(
        fgsm_mae,
        eot_mae,
        "Average Perturbation MAE",
        "MAE",
        "mae_comparison.png"
    )

    grouped_bar(
        fgsm_rmse,
        eot_rmse,
        "Average Perturbation RMSE",
        "RMSE",
        "rmse_comparison.png"
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x, drift)
    ax.set_title("Clean Prediction Drift")
    ax.set_ylabel("Drift Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(transformations, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "clean_prediction_drift.png", dpi=150)
    plt.close(fig)

    # Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(x - width / 2, fgsm_asr, width, label="FGSM")
    axes[0, 0].bar(x + width / 2, eot_asr, width, label="EOT-PGD")
    axes[0, 0].set_title("Conditional ASR (%)")
    axes[0, 0].legend()

    axes[0, 1].bar(x, drift)
    axes[0, 1].set_title("Clean Prediction Drift (%)")

    axes[1, 0].bar(x - width / 2, fgsm_mae, width, label="FGSM")
    axes[1, 0].bar(x + width / 2, eot_mae, width, label="EOT-PGD")
    axes[1, 0].set_title("Average MAE")
    axes[1, 0].legend()

    axes[1, 1].bar(x - width / 2, fgsm_rmse, width, label="FGSM")
    axes[1, 1].bar(x + width / 2, eot_rmse, width, label="EOT-PGD")
    axes[1, 1].set_title("Average RMSE")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(transformations, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Preprocessing Robustness Evaluation", fontsize=16)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "preprocessing_dashboard.png", dpi=160)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess, categories = load_model()
    baseline_labels = load_baseline_labels()

    detailed_rows, clean_drift_stats = evaluate(
        model,
        preprocess,
        categories,
        baseline_labels
    )

    summary_rows = create_summary(
        detailed_rows,
        clean_drift_stats
    )

    save_csv(
        OUTPUT_DIR / "detailed_results.csv",
        detailed_rows
    )

    save_csv(
        OUTPUT_DIR / "summary.csv",
        summary_rows
    )

    make_plots(summary_rows, clean_drift_stats)

    print("\n" + "=" * 55)
    print("PREPROCESSING EVALUATION COMPLETE")
    print("=" * 55)

    print(f"Detailed cases: {len(detailed_rows)}")
    print(f"Summary rows:   {len(summary_rows)}")
    print(f"Output folder:  {OUTPUT_DIR}")

    print("\nSUMMARY:")

    for row in summary_rows:
        print(
            f"{row['attack']:8} | "
            f"{row['transformation']:20} | "
            f"Conditional ASR: "
            f"{row['conditional_asr_percent']:6.2f}% | "
            f"Clean Drift: "
            f"{row['clean_drift_rate_percent']:6.2f}%"
        )


if __name__ == "__main__":
    main()
