"""
cleaning.py
-----------
Optional Cleanlab integration for finding and removing mislabelled images.
Creates a new cleaned dataset folder (original stays untouched).
"""

import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.model_selection import cross_val_predict

from utils import get_device, get_transforms

DEVICE = get_device()


def get_pred_probs(model, dataset, batch_size: int = 64) -> np.ndarray:
    """Get predicted probabilities for every sample in the dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_probs = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    out = model(x)
            else:
                out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def find_label_issues(
    model,
    dataset_root: str,
    scope: str = "Both",          # "Train only" | "Test only" | "Both"
    batch_size: int = 64,
    progress_callback=None
) -> Dict:
    """
    Use Cleanlab (or a simple confidence-based fallback) to find likely mislabelled images.

    Returns a dictionary with:
    - issues per split
    - summary statistics
    """
    try:
        from cleanlab.filter import find_label_issues as cl_find_label_issues
        use_cleanlab = True
    except ImportError:
        use_cleanlab = False
        if progress_callback:
            progress_callback(0.1, "Cleanlab not installed – using confidence fallback...")

    root = Path(dataset_root)
    train_tf, val_tf = get_transforms(224)

    splits_to_process = []
    if scope in ("Train only", "Both"):
        splits_to_process.append("train")
    if scope in ("Test only", "Both"):
        splits_to_process.append("test")

    results = {"splits": {}, "total_issues": 0}

    for i, split in enumerate(splits_to_process):
        if progress_callback:
            progress_callback(0.15 + i * 0.35, f"Analyzing {split} set...")

        ds = datasets.ImageFolder(root / split, transform=val_tf)
        labels = np.array(ds.targets)
        paths = [s[0] for s in ds.samples]

        # Get model predictions
        pred_probs = get_pred_probs(model, ds, batch_size=batch_size)

        if use_cleanlab:
            issue_idx = cl_find_label_issues(
                labels=labels,
                pred_probs=pred_probs,
                return_indices_ranked_by="self_confidence"
            )
        else:
            # Simple fallback: model is confident but disagrees with label
            pred_labels = pred_probs.argmax(axis=1)
            confidence = pred_probs.max(axis=1)
            disagree = pred_labels != labels
            high_conf_disagree = disagree & (confidence > 0.70)
            issue_idx = np.where(high_conf_disagree)[0]
            # Rank by confidence
            issue_idx = issue_idx[np.argsort(-confidence[issue_idx])]

        issue_paths = [paths[j] for j in issue_idx]
        issue_true = [ds.classes[labels[j]] for j in issue_idx]
        issue_pred = [ds.classes[pred_probs[j].argmax()] for j in issue_idx]

        results["splits"][split] = {
            "indices": issue_idx,
            "paths": issue_paths,
            "true_labels": issue_true,
            "pred_labels": issue_pred,
            "count": len(issue_idx)
        }
        results["total_issues"] += len(issue_idx)

    return results


def create_cleaned_dataset(
    dataset_root: str,
    issues: Dict,
    progress_callback=None
) -> Tuple[str, str]:
    """
    Create a new cleaned dataset folder by copying everything
    except the flagged mislabelled images.

    Returns (cleaned_root_path, markdown_summary)
    """
    root = Path(dataset_root)
    cleaned_root = root.parent / f"{root.name}_cleaned"

    if cleaned_root.exists():
        shutil.rmtree(cleaned_root)

    if progress_callback:
        progress_callback(0.7, "Creating cleaned dataset...")

    # Build set of paths to exclude
    exclude = set()
    for split_data in issues["splits"].values():
        exclude.update(split_data["paths"])

    # Copy files
    removed_per_class = defaultdict(int)
    kept_total = 0

    for split in ["train", "test"]:
        src_split = root / split
        dst_split = cleaned_root / split
        if not src_split.exists():
            continue

        for class_dir in src_split.iterdir():
            if not class_dir.is_dir():
                continue
            dst_class = dst_split / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)

            for img_path in class_dir.glob("*.*"):
                if str(img_path) in exclude:
                    removed_per_class[class_dir.name] += 1
                else:
                    shutil.copy2(img_path, dst_class / img_path.name)
                    kept_total += 1

    # Build summary
    lines = [
        f"**Cleaned dataset created**",
        f"",
        f"`{cleaned_root}`",
        f"",
        f"**Removed {issues['total_issues']} suspicious images**",
        f"**Kept {kept_total:,} images**",
        f"",
        f"| Class | Removed |",
        f"|-------|--------:|",
    ]
    for cls, cnt in sorted(removed_per_class.items()):
        lines.append(f"| {cls} | {cnt} |")

    if progress_callback:
        progress_callback(1.0, "Done")

    return str(cleaned_root), "\n".join(lines)
