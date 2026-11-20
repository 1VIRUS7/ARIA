
import csv
import io
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import resnet18, ResNet18_Weights


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = Path("dataset/clean")
BASELINE_CSV = Path("comparison_results/baseline_predictions.csv")

OUTPUT_DIR = Path("dataset/adversarial")
FGSM_DIR = OUTPUT_DIR / "FGSM"
EOT_DIR = OUTPUT_DIR / "EOT-PGD"

RESULTS_DIR = Path("comparison_results")
PLOTS_DIR = RESULTS_DIR / "adversarial_generation_plots"

EPSILON = 8 / 255
ALPHA = 1 / 255

EOT_STEPS = 20
EOT_SAMPLES = 5

MIN_SCALE = 0.90
MAX_SCALE = 1.10

MIN_JPEG_QUALITY = 65
MAX_JPEG_QUALITY = 95

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

for directory in [FGSM_DIR, EOT_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 65)
print("MULTI-IMAGE ADVERSARIAL GENERATION")
print("=" * 65)

print(f"[+] Device: {DEVICE}")
print("[+] Loading ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights).to(DEVICE)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

print("[+] Model loaded.")


# =========================================================
# LOAD BASELINE LABELS
# =========================================================

if not BASELINE_CSV.exists():
    raise FileNotFoundError(
        f"Baseline CSV not found: {BASELINE_CSV}"
    )

with open(BASELINE_CSV, "r", encoding="utf-8") as file:
    baseline_rows = list(csv.DictReader(file))

baseline = {}

for row in baseline_rows:
    if row["status"] != "SUCCESS":
        continue

    baseline[row["image"]] = int(
        row["predicted_class_id"]
    )

if not baseline:
    raise RuntimeError("No valid baseline labels found.")

print(f"[+] Baseline labels loaded: {len(baseline)}")


# =========================================================
# IMAGE UTILITIES
# =========================================================

def load_image(path):
    image = Image.open(path).convert("RGB")

    tensor = TF.to_tensor(image)

    return tensor.to(DEVICE)


def save_tensor(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)

    image = TF.to_pil_image(tensor)

    image.save(path, format="PNG")


def predict(tensor):
    model_input = preprocess(tensor).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(model_input)
        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = probabilities.max(dim=1)

    return prediction.item(), confidence.item()


# =========================================================
# EOT RANDOM RESIZE / CROP / PAD
# =========================================================

def random_resize_transform(tensor):

    _, height, width = tensor.shape

    scale = random.uniform(MIN_SCALE, MAX_SCALE)

    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    transformed = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # If larger: center crop
    if new_height > height:
        top = (new_height - height) // 2
        transformed = transformed[
            :, top:top + height, :
        ]

    if new_width > width:
        left = (new_width - width) // 2
        transformed = transformed[
            :, :, left:left + width
        ]

    # If smaller: replicate padding
    current_height = transformed.shape[1]
    current_width = transformed.shape[2]

    pad_h = height - current_height
    pad_w = width - current_width

    if pad_h > 0 or pad_w > 0:

        left = pad_w // 2
        right = pad_w - left

        top = pad_h // 2
        bottom = pad_h - top

        transformed = F.pad(
            transformed.unsqueeze(0),
            (left, right, top, bottom),
            mode="replicate"
        ).squeeze(0)

    return transformed


# =========================================================
# JPEG BPDA
# =========================================================

def jpeg_bpda(tensor, quality):

    # Forward pass: actual JPEG compression
    image = TF.to_pil_image(
        tensor.detach().cpu().clamp(0, 1)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")

    compressed_tensor = TF.to_tensor(
        compressed
    ).to(tensor.device)

    # Backward pass: identity approximation
    return tensor + (compressed_tensor - tensor).detach()


# =========================================================
# MODEL LOSS
# =========================================================

def calculate_loss(tensor, target):

    model_input = preprocess(tensor).unsqueeze(0)

    logits = model(model_input)

    loss = F.cross_entropy(
        logits,
        target
    )

    return loss


# =========================================================
# FGSM ATTACK
# =========================================================

def generate_fgsm(image, target):

    original = image.detach()

    adversarial = original.clone().detach()
    adversarial.requires_grad_(True)

    loss = calculate_loss(
        adversarial,
        target
    )

    gradient = torch.autograd.grad(
        loss,
        adversarial
    )[0]

    adversarial = (
        original + EPSILON * gradient.sign()
    )

    adversarial = torch.clamp(
        adversarial,
        min=original - EPSILON,
        max=original + EPSILON
    )

    adversarial = adversarial.clamp(0, 1)

    return adversarial.detach()


# =========================================================
# EOT-PGD ATTACK
# =========================================================

def generate_eot_pgd(image, target):

    original = image.detach()

    adversarial = original.clone().detach()

    for step in range(EOT_STEPS):

        adversarial = adversarial.detach()
        adversarial.requires_grad_(True)

        gradient_sum = torch.zeros_like(adversarial)

        for sample in range(EOT_SAMPLES):

            transformed = random_resize_transform(
                adversarial
            )

            quality = random.randint(
                MIN_JPEG_QUALITY,
                MAX_JPEG_QUALITY
            )

            transformed = jpeg_bpda(
                transformed,
                quality
            )

            loss = calculate_loss(
                transformed,
                target
            )

            gradient = torch.autograd.grad(
                loss,
                adversarial
            )[0]

            gradient_sum += gradient

        average_gradient = gradient_sum / EOT_SAMPLES

        adversarial = (
            adversarial.detach()
            + ALPHA * average_gradient.sign()
        )

        # Project to epsilon ball
        adversarial = torch.maximum(
            torch.minimum(
                adversarial,
                original + EPSILON
            ),
            original - EPSILON
        )

        adversarial = adversarial.clamp(0, 1)

        if (step + 1) % 5 == 0:
            print(
                f"       EOT step {step + 1}/{EOT_STEPS}"
            )

    return adversarial.detach()


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(clean, adversarial):

    delta = (adversarial - clean).abs()

    mae = delta.mean().item()

    rmse = torch.sqrt(
        ((adversarial - clean) ** 2).mean()
    ).item()

    max_delta = delta.max().item()

    return mae, rmse, max_delta


# =========================================================
# GENERATE ADVERSARIAL DATASET
# =========================================================

results = []

total = len(baseline)

print("\n[+] Starting adversarial generation...\n")

for index, (filename, target_id) in enumerate(
    baseline.items(),
    start=1
):

    image_path = DATASET_DIR / filename

    print(
        f"\n[{index}/{total}] Processing {filename}"
    )

    try:

        clean = load_image(image_path)

        target = torch.tensor(
            [target_id],
            dtype=torch.long,
            device=DEVICE
        )

        print(
            f"    Reference: {categories[target_id]}"
        )

        # -------------------------------------
        # Generate FGSM
        # -------------------------------------

        print("    [+] Generating FGSM...")

        fgsm_adv = generate_fgsm(
            clean,
            target
        )

        fgsm_path = FGSM_DIR / (
            Path(filename).stem + ".png"
        )

        save_tensor(
            fgsm_adv,
            fgsm_path
        )

        # -------------------------------------
        # Generate EOT-PGD
        # -------------------------------------

        print("    [+] Generating EOT-PGD...")

        eot_adv = generate_eot_pgd(
            clean,
            target
        )

        eot_path = EOT_DIR / (
            Path(filename).stem + ".png"
        )

        save_tensor(
            eot_adv,
            eot_path
        )

        # -------------------------------------
        # Evaluate predictions
        # -------------------------------------

        clean_pred, clean_conf = predict(clean)

        fgsm_pred, fgsm_conf = predict(fgsm_adv)

        eot_pred, eot_conf = predict(eot_adv)

        # -------------------------------------
        # Calculate perturbation metrics
        # -------------------------------------

        fgsm_mae, fgsm_rmse, fgsm_max = calculate_metrics(
            clean,
            fgsm_adv
        )

        eot_mae, eot_rmse, eot_max = calculate_metrics(
            clean,
            eot_adv
        )

        fgsm_success = fgsm_pred != target_id
        eot_success = eot_pred != target_id

        result = {
            "image": filename,
            "reference_class_id": target_id,
            "reference_class": categories[target_id],

            "clean_prediction": clean_pred,
            "clean_confidence": clean_conf,

            "fgsm_prediction": fgsm_pred,
            "fgsm_class": categories[fgsm_pred],
            "fgsm_confidence": fgsm_conf,
            "fgsm_success": fgsm_success,
            "fgsm_mae": fgsm_mae,
            "fgsm_rmse": fgsm_rmse,
            "fgsm_max_delta": fgsm_max,

            "eot_prediction": eot_pred,
            "eot_class": categories[eot_pred],
            "eot_confidence": eot_conf,
            "eot_success": eot_success,
            "eot_mae": eot_mae,
            "eot_rmse": eot_rmse,
            "eot_max_delta": eot_max,

            "status": "SUCCESS"
        }

        results.append(result)

        print(
            f"    FGSM: {categories[fgsm_pred]} "
            f"| Success: {fgsm_success}"
        )

        print(
            f"    EOT:  {categories[eot_pred]} "
            f"| Success: {eot_success}"
        )

        print(
            f"    FGSM L-inf: {fgsm_max:.6f} "
            f"| EOT L-inf: {eot_max:.6f}"
        )

    except Exception as error:

        print(f"    [!] Error: {error}")

        results.append({
            "image": filename,
            "reference_class_id": target_id,
            "reference_class": categories[target_id],
            "status": f"ERROR: {error}"
        })


# =========================================================
# SAVE RESULTS CSV
# =========================================================

csv_path = RESULTS_DIR / "adversarial_generation_results.csv"

fieldnames = [
    "image",
    "reference_class_id",
    "reference_class",
    "clean_prediction",
    "clean_confidence",
    "fgsm_prediction",
    "fgsm_class",
    "fgsm_confidence",
    "fgsm_success",
    "fgsm_mae",
    "fgsm_rmse",
    "fgsm_max_delta",
    "eot_prediction",
    "eot_class",
    "eot_confidence",
    "eot_success",
    "eot_mae",
    "eot_rmse",
    "eot_max_delta",
    "status"
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()

    writer.writerows(results)

print(f"\n[+] CSV saved: {csv_path}")


# =========================================================
# STATISTICAL SUMMARY
# =========================================================

valid = [
    result for result in results
    if result["status"] == "SUCCESS"
]

if not valid:
    raise RuntimeError(
        "No images were successfully processed."
    )

fgsm_asr = np.mean([
    result["fgsm_success"]
    for result in valid
]) * 100

eot_asr = np.mean([
    result["eot_success"]
    for result in valid
]) * 100

fgsm_mae_avg = np.mean([
    result["fgsm_mae"]
    for result in valid
])

eot_mae_avg = np.mean([
    result["eot_mae"]
    for result in valid
])

fgsm_rmse_avg = np.mean([
    result["fgsm_rmse"]
    for result in valid
])

eot_rmse_avg = np.mean([
    result["eot_rmse"]
    for result in valid
])

fgsm_max_avg = np.mean([
    result["fgsm_max_delta"]
    for result in valid
])

eot_max_avg = np.mean([
    result["eot_max_delta"]
    for result in valid
])


# =========================================================
# MATPLOTLIB VISUALIZATIONS
# =========================================================

attack_names = ["FGSM", "EOT-PGD"]

asr_values = [fgsm_asr, eot_asr]

mae_values = [fgsm_mae_avg, eot_mae_avg]

rmse_values = [fgsm_rmse_avg, eot_rmse_avg]

max_values = [fgsm_max_avg, eot_max_avg]


# Plot 1: Attack Success Rate

plt.figure(figsize=(8, 6))

bars = plt.bar(
    attack_names,
    asr_values,
    edgecolor="black"
)

plt.title("Attack Success Rate - 50 Images")
plt.ylabel("ASR (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, asr_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "attack_success_rate.png",
    dpi=300
)

plt.close()


# Plot 2: Perturbation Metrics

x = np.arange(2)
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    mae_values,
    width,
    label="MAE"
)

plt.bar(
    x,
    rmse_values,
    width,
    label="RMSE"
)

plt.bar(
    x + width,
    max_values,
    width,
    label="Max Delta"
)

plt.xticks(x, attack_names)

plt.title("FGSM vs EOT-PGD Perturbation Metrics")
plt.ylabel("Pixel Difference")
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "perturbation_metrics.png",
    dpi=300
)

plt.close()


# Plot 3: Per-image Success Comparison

indices = np.arange(len(valid))

fgsm_successes = [
    int(result["fgsm_success"])
    for result in valid
]

eot_successes = [
    int(result["eot_success"])
    for result in valid
]

plt.figure(figsize=(14, 6))

plt.plot(
    indices,
    fgsm_successes,
    marker="o",
    label="FGSM"
)

plt.plot(
    indices,
    eot_successes,
    marker="x",
    label="EOT-PGD"
)

plt.yticks([0, 1], ["Failed", "Success"])

plt.xlabel("Image Index")
plt.ylabel("Attack Result")

plt.title("Per-image Attack Success")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "per_image_success.png",
    dpi=300
)

plt.close()


# Plot 4: Combined Dashboard

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

axes[0, 0].bar(attack_names, asr_values)
axes[0, 0].set_title("Attack Success Rate")
axes[0, 0].set_ylabel("ASR (%)")
axes[0, 0].set_ylim(0, 100)

axes[0, 1].bar(attack_names, mae_values)
axes[0, 1].set_title("Average MAE")

axes[1, 0].bar(attack_names, rmse_values)
axes[1, 0].set_title("Average RMSE")

axes[1, 1].bar(attack_names, max_values)
axes[1, 1].set_title("Average Maximum Delta")

fig.suptitle(
    "Adversarial Generation Dashboard",
    fontsize=17
)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "adversarial_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("ADVERSARIAL GENERATION SUMMARY")
print("=" * 65)

print(f"Total baseline images: {total}")
print(f"Successful images:    {len(valid)}")
print(f"Failed images:        {total - len(valid)}")

print("\n--- FGSM ---")
print(f"ASR:        {fgsm_asr:.2f}%")
print(f"Average MAE: {fgsm_mae_avg:.6f}")
print(f"Average RMSE: {fgsm_rmse_avg:.6f}")
print(f"Average Max Delta: {fgsm_max_avg:.6f}")

print("\n--- EOT-PGD ---")
print(f"ASR:        {eot_asr:.2f}%")
print(f"Average MAE: {eot_mae_avg:.6f}")
print(f"Average RMSE: {eot_rmse_avg:.6f}")
print(f"Average Max Delta: {eot_max_avg:.6f}")

print("\nGenerated files:")
print(f"[+] {csv_path}")
print(f"[+] {FGSM_DIR}")
print(f"[+] {EOT_DIR}")
print(f"[+] {PLOTS_DIR}")

print("\n" + "=" * 65)
print("GENERATION COMPLETED")
print("=" * 65)
