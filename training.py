"""
training.py
-----------
Model definition, training loop, prediction, and checkpoint handling.
"""

import copy
import time
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights,
    EfficientNet_B2_Weights, EfficientNet_B3_Weights
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils import get_device, get_transforms

DEVICE = get_device()

MODEL_MAP = {
    "EfficientNet-B0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "EfficientNet-B1": (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "EfficientNet-B2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "EfficientNet-B3": (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Build an EfficientNet model with a new classifier head."""
    ctor, weights = MODEL_MAP[model_name]
    model = ctor(weights=weights)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


def plot_confusion_matrix(cm, class_names):
    """Create a dark-themed confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor="#0c1222")
    ax.set_facecolor("#0c1222")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.set_xlabel("Predicted", color="#cbd5e1")
    ax.set_ylabel("True", color="#cbd5e1")
    ax.set_title("Confusion Matrix", color="#f1f5f9", pad=10)
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    return fig


def train_model(
    dataset_root: str,
    model_name: str = "EfficientNet-B0",
    epochs_stage1: int = 6,
    epochs_stage2: int = 8,
    batch_size: int = 32,
    progress_callback=None
):
    """
    Two-stage transfer learning training.

    Stage 1: Train only the classifier head
    Stage 2: Fine-tune the last blocks
    """
    root = Path(dataset_root)

    # Choose image size based on model
    img_size = 224
    if "B2" in model_name:
        img_size = 260
    elif "B3" in model_name:
        img_size = 300

    train_tf, val_tf = get_transforms(img_size)

    start_time = time.time()
    if progress_callback:
        progress_callback(0.03, "Loading dataset...")

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=val_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)

    # Class weights for imbalance
    targets = train_ds.targets
    counts = Counter(targets)
    weights = torch.tensor([1.0 / counts[i] for i in range(num_classes)], dtype=torch.float)
    sampler = WeightedRandomSampler([weights[t] for t in targets], len(targets), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=int(batch_size), sampler=sampler,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(batch_size) * 2, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    if progress_callback:
        progress_callback(0.06, f"Building {model_name}...")

    model = build_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # ---------- Stage 1 ----------
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs_stage1))
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history = {"train": [], "val": []}
    total_epochs = int(epochs_stage1) + int(epochs_stage2)

    for ep in range(1, int(epochs_stage1) + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        train_acc = 100 * correct / total
        scheduler.step()

        # Validation
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        out = model(x)
                else:
                    out = model(x)
                _, pred = out.max(1)
                v_total += y.size(0)
                v_correct += pred.eq(y).sum().item()

        val_acc = 100 * v_correct / v_total
        history["train"].append(train_acc)
        history["val"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        if progress_callback:
            pct = 0.06 + (ep / total_epochs) * 0.55
            elapsed = time.time() - start_time
            # rough ETA based on current progress
            if pct > 0.07:
                eta = elapsed / pct * (1 - pct)
                eta_str = f" · ETA {int(eta//60)}m {int(eta%60)}s"
            else:
                eta_str = ""
            progress_callback(pct, f"Stage 1 · {ep}/{int(epochs_stage1)} · Val {val_acc:.1f}%{eta_str}")

    # ---------- Stage 2 ----------
    model.load_state_dict(best_wts)

    # Unfreeze last blocks
    for name, param in model.named_parameters():
        if any(k in name for k in ["features.6", "features.7", "features.8", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs_stage2))

    for ep in range(1, int(epochs_stage2) + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        train_acc = 100 * correct / total
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        out = model(x)
                else:
                    out = model(x)
                _, pred = out.max(1)
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(labels, preds) * 100
        history["train"].append(train_acc)
        history["val"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        if progress_callback:
            pct = 0.62 + (ep / int(epochs_stage2)) * 0.32
            elapsed = time.time() - start_time
            if pct > 0.07:
                eta = elapsed / pct * (1 - pct)
                eta_str = f" · ETA {int(eta//60)}m {int(eta%60)}s"
            else:
                eta_str = ""
            progress_callback(pct, f"Stage 2 · {ep}/{int(epochs_stage2)} · Val {val_acc:.1f}%{eta_str}")

    # Final evaluation
    if progress_callback:
        progress_callback(0.96, "Final evaluation...")

    model.load_state_dict(best_wts)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(x)
            else:
                out = model(x)
            _, pred = out.max(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    final_acc = accuracy_score(labels, preds) * 100
    report = classification_report(labels, preds, target_names=class_names, digits=3)
    cm = confusion_matrix(labels, preds)

    # Save checkpoint
    save_path = root / "best_model.pth"
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
        "accuracy": final_acc
    }, save_path)

    # Accuracy plot
    fig_acc, ax = plt.subplots(figsize=(7, 3.8), facecolor="#0c1222")
    ax.set_facecolor("#0c1222")
    ax.plot(history["train"], label="Train", color="#60a5fa", lw=2)
    ax.plot(history["val"], label="Val", color="#34d399", lw=2)
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("Accuracy %", color="#94a3b8")
    ax.set_title("Training Curve", color="#f1f5f9", pad=8)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, alpha=0.12)
    for s in ax.spines.values():
        s.set_color("#334155")
    plt.tight_layout()

    fig_cm = plot_confusion_matrix(cm, class_names)

    summary = f"""### Training Finished

**Model:** {model_name}  
**Accuracy:** **{final_acc:.2f}%**

```
{report}
```
"""
    if progress_callback:
        progress_callback(1.0, "Done")

    return {
        "model": model,
        "class_names": class_names,
        "summary": summary,
        "fig_acc": fig_acc,
        "fig_cm": fig_cm,
        "status": f"✓ {model_name} · {final_acc:.2f}%",
        "model_path": str(save_path),
        "accuracy": final_acc
    }


def load_checkpoint(model_file: str):
    """Load a previously saved .pth checkpoint."""
    checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)

    if "class_names" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Invalid model file. Please use a model exported from this app.")

    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "EfficientNet-B0")
    accuracy = checkpoint.get("accuracy", None)

    if model_name not in MODEL_MAP:
        model_name = "EfficientNet-B0"

    model = build_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, class_names, model_name, accuracy


def predict(model, class_names, image):
    """Run prediction on a single image."""
    from utils import get_transforms
    tf = get_transforms(224)[1]  # validation transform

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8")).convert("RGB")
    else:
        image = image.convert("RGB")

    x = tf(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                out = model(x)
        else:
            out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    pred = class_names[idx.item()]
    confidence = conf.item() * 100
    probs_np = probs.cpu().numpy()

    # Confidence bar chart
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="#0c1222")
    ax.set_facecolor("#0c1222")
    colors = ["#34d399" if i == idx.item() else "#3b82f6" for i in range(len(class_names))]
    bars = ax.barh(class_names, probs_np * 100, color=colors, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence %", color="#94a3b8")
    ax.set_title(f"{pred}  ·  {confidence:.1f}%", color="#f1f5f9", pad=10)
    ax.tick_params(colors="#94a3b8")
    for s in ax.spines.values():
        s.set_color("#334155")
    for bar, p in zip(bars, probs_np):
        ax.text(p * 100 + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="#e2e8f0", fontsize=9)
    plt.tight_layout()

    result_text = f"### {pred}\n**{confidence:.1f}%** confidence"
    return result_text, fig
