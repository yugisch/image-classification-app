"""
utils.py
--------
Helper functions for dataset handling and image transforms.
"""

from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List

import torch
from torchvision import transforms


def get_device() -> torch.device:
    """Return the best available device (CUDA if possible)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> str:
    """Human-readable device information."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{name}  ·  {vram:.1f} GB"
    return "CPU only (training will be slow)"


def find_dataset_root(folder_path: str) -> Path:
    """
    Locate the real dataset root that contains train/ and test/ folders.
    Handles cases where the user points to a parent or nested folder.
    """
    root = Path(folder_path.strip().strip('"').strip("'"))

    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    train_dir = root / "train"
    test_dir = root / "test"

    # If train/ is not directly under root, search for it
    if not train_dir.exists():
        candidates = list(root.rglob("train"))
        if candidates:
            train_dir = min(candidates, key=lambda p: len(str(p)))
            root = train_dir.parent
            test_dir = root / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Could not find both 'train' and 'test' folders.")

    return root


def analyze_dataset(root: Path) -> Tuple[str, Dict]:
    """
    Scan the dataset and return a markdown summary + statistics dict.
    """
    train_dir = root / "train"
    test_dir = root / "test"

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError("No class folders found inside train/.")

    train_counts = {}
    test_counts = {}
    total_train = total_test = 0

    for cls in classes:
        n_train = len(list((train_dir / cls).glob("*.*")))
        n_test = len(list((test_dir / cls).glob("*.*")))
        train_counts[cls] = n_train
        test_counts[cls] = n_test
        total_train += n_train
        total_test += n_test

    lines = [
        f"**Dataset ready**",
        f"",
        f"`{root}`",
        f"",
        f"**{len(classes)} classes** · **{total_train + total_test:,} images**",
        f"",
        f"| Class | Train | Test |",
        f"|-------|------:|-----:|",
    ]
    for cls in classes:
        lines.append(f"| {cls} | {train_counts[cls]:,} | {test_counts[cls]:,} |")

    # Warn about imbalance
    if train_counts:
        max_c = max(train_counts.values())
        min_c = min(train_counts.values())
        if max_c / max(min_c, 1) > 4:
            lines.append("")
            lines.append("⚠️ Imbalanced dataset — class weights will be used.")

    stats = {
        "root": str(root),
        "classes": classes,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "total_train": total_train,
        "total_test": total_test,
    }
    return "\n".join(lines), stats


def get_transforms(img_size: int = 224):
    """Return train and validation transforms."""
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------- Persistence helpers ----------

CONFIG_FILE = Path.home() / ".image_classification_app.json"


def load_last_folder() -> str:
    """Load the last used folder path from config file."""
    try:
        if CONFIG_FILE.exists():
            import json
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return data.get("last_folder", "")
    except Exception:
        pass
    return ""


def save_last_folder(folder: str):
    """Save the last used folder path."""
    try:
        import json
        data = {}
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        data["last_folder"] = folder
        CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def export_results_csv(class_names, report_text: str, accuracy: float, output_dir: str = ".") -> str:
    """Export a simple classification report to CSV and return the path."""
    import csv
    from datetime import datetime

    out = Path(output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    lines = [l.strip() for l in report_text.strip().splitlines() if l.strip()]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Overall Accuracy", f"{accuracy:.4f}"])
        writer.writerow([])
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])

        # Parse sklearn classification_report lines
        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and parts[0] not in ("accuracy", "macro", "weighted", "precision"):
                try:
                    writer.writerow([parts[0], parts[1], parts[2], parts[3], parts[4]])
                except Exception:
                    pass

    return str(out)


def export_results_pdf(summary_md: str, fig_acc, fig_cm, accuracy: float, output_dir: str = ".") -> str:
    """
    Export a simple PDF report.
    Requires matplotlib which is already installed.
    """
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    out = Path(output_dir) / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    with PdfPages(out) as pdf:
        # Page 1 - summary text
        fig, ax = plt.subplots(figsize=(8.5, 11), facecolor="white")
        ax.axis("off")
        ax.set_title("Image Classification Report", fontsize=16, pad=20)
        text = summary_md.replace("###", "").replace("**", "").replace("```", "")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2 - accuracy curve
        if fig_acc is not None:
            pdf.savefig(fig_acc, bbox_inches="tight")

        # Page 3 - confusion matrix
        if fig_cm is not None:
            pdf.savefig(fig_cm, bbox_inches="tight")

    return str(out)
