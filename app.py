import copy
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights,
    EfficientNet_B2_Weights, EfficientNet_B3_Weights
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CURRENT_MODEL = None
CLASS_NAMES = None
DATASET_ROOT = None

MODEL_MAP = {
    "EfficientNet-B0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "EfficientNet-B1": (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "EfficientNet-B2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "EfficientNet-B3": (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
}


def get_device_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{name}  ·  {vram:.1f} GB"
    return "CPU only (training will be slow)"


def analyze_folder(folder_path, progress=gr.Progress()):
    global DATASET_ROOT, CLASS_NAMES

    if not folder_path or not str(folder_path).strip():
        raise gr.Error("Please enter a folder path.")

    folder_path = str(folder_path).strip().strip('"').strip("'")
    root = Path(folder_path)

    if not root.exists():
        raise gr.Error(f"Folder not found:\n{root}")

    train_dir = root / "train"
    test_dir = root / "test"

    if not train_dir.exists():
        candidates = list(root.rglob("train"))
        if candidates:
            train_dir = min(candidates, key=lambda p: len(str(p)))
            root = train_dir.parent
            test_dir = root / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise gr.Error("Could not find 'train' and 'test' folders.")

    progress(0.4, desc="Scanning classes...")

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise gr.Error("No class folders found inside train/.")

    train_counts, test_counts = {}, {}
    total_train = total_test = 0

    for cls in classes:
        n_train = len(list((train_dir / cls).glob("*.*")))
        n_test = len(list((test_dir / cls).glob("*.*")))
        train_counts[cls] = n_train
        test_counts[cls] = n_test
        total_train += n_train
        total_test += n_test

    DATASET_ROOT = str(root)
    CLASS_NAMES = classes

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

    if train_counts:
        max_c = max(train_counts.values())
        min_c = min(train_counts.values())
        if max_c / max(min_c, 1) > 4:
            lines.append("")
            lines.append("⚠️ Imbalanced dataset — class weights will be used.")

    progress(1.0, desc="Done")
    return "\n".join(lines), f"✓ {len(classes)} classes · {total_train + total_test:,} images"


def get_transforms(img_size=224):
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


def build_model(model_name, num_classes):
    ctor, weights = MODEL_MAP[model_name]
    model = ctor(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


def plot_cm(cm, class_names):
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


def train_model(model_name, epochs1, epochs2, batch_size, progress=gr.Progress()):
    global CURRENT_MODEL, CLASS_NAMES, DATASET_ROOT

    if not DATASET_ROOT:
        raise gr.Error("Please analyze a dataset folder first.")

    root = Path(DATASET_ROOT)
    img_size = 224
    if "B2" in model_name:
        img_size = 260
    elif "B3" in model_name:
        img_size = 300

    train_tf, val_tf = get_transforms(img_size)

    progress(0.03, desc="Loading dataset...")
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=val_tf)

    CLASS_NAMES = train_ds.classes
    num_classes = len(CLASS_NAMES)

    targets = train_ds.targets
    counts = Counter(targets)
    weights = torch.tensor([1.0 / counts[i] for i in range(num_classes)], dtype=torch.float)
    sampler = WeightedRandomSampler([weights[t] for t in targets], len(targets), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=int(batch_size)*2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    progress(0.06, desc=f"Building {model_name}...")
    model = build_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs1))
    best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())
    history = {"train": [], "val": []}
    total_ep = int(epochs1) + int(epochs2)

    for ep in range(1, int(epochs1)+1):
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
        progress(0.06 + (ep / total_ep) * 0.55, desc=f"Stage 1 · {ep}/{int(epochs1)} · Val {val_acc:.1f}%")

    model.load_state_dict(best_wts)
    for name, param in model.named_parameters():
        if any(k in name for k in ["features.6", "features.7", "features.8", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs2))

    for ep in range(1, int(epochs2)+1):
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
        progress(0.62 + (ep / int(epochs2)) * 0.32, desc=f"Stage 2 · {ep}/{int(epochs2)} · Val {val_acc:.1f}%")

    progress(0.96, desc="Final evaluation...")
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
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=3)
    cm = confusion_matrix(labels, preds)

    CURRENT_MODEL = model

    torch.save({
        "model_state": model.state_dict(),
        "class_names": CLASS_NAMES,
        "model_name": model_name,
        "accuracy": final_acc
    }, Path(DATASET_ROOT) / "best_model.pth")

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

    fig_cm = plot_cm(cm, CLASS_NAMES)

    summary = f"""### Training Finished

**Model:** {model_name}  
**Accuracy:** **{final_acc:.2f}%**

```
{report}
```
"""
    progress(1.0, desc="Done")
    return summary, fig_acc, fig_cm, f"✓ {model_name} · {final_acc:.2f}%"


def predict_image(image, progress=gr.Progress()):
    global CURRENT_MODEL, CLASS_NAMES

    if CURRENT_MODEL is None or CLASS_NAMES is None:
        raise gr.Error("Train a model first.")

    if image is None:
        raise gr.Error("Upload an image.")

    progress(0.3, desc="Preprocessing...")
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8")).convert("RGB")
    else:
        image = image.convert("RGB")

    x = tf(image).unsqueeze(0).to(DEVICE)

    progress(0.7, desc="Predicting...")
    CURRENT_MODEL.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                out = CURRENT_MODEL(x)
        else:
            out = CURRENT_MODEL(x)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    pred = CLASS_NAMES[idx.item()]
    confidence = conf.item() * 100
    probs_np = probs.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="#0c1222")
    ax.set_facecolor("#0c1222")
    colors = ["#34d399" if i == idx.item() else "#3b82f6" for i in range(len(CLASS_NAMES))]
    bars = ax.barh(CLASS_NAMES, probs_np * 100, color=colors, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence %", color="#94a3b8")
    ax.set_title(f"{pred}  ·  {confidence:.1f}%", color="#f1f5f9", pad=10)
    ax.tick_params(colors="#94a3b8")
    for s in ax.spines.values():
        s.set_color("#334155")
    for bar, p in zip(bars, probs_np):
        ax.text(p*100 + 1.2, bar.get_y() + bar.get_height()/2, f"{p*100:.1f}%",
                va="center", color="#e2e8f0", fontsize=9)
    plt.tight_layout()

    progress(1.0, desc="Done")
    return f"### {pred}\n**{confidence:.1f}%** confidence", fig


# ====================== UI ======================

css = """
html, body, .gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container {
    padding: 12px 28px 20px 28px !important;
}
footer {display:none !important;}
.main-row {
    gap: 16px !important;
}
"""

with gr.Blocks(
    title="Image Classification",
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#080c18",
        body_background_fill_dark="#080c18",
        block_background_fill="#0f172a",
        block_background_fill_dark="#0f172a",
        border_color_primary="#1e293b",
        border_color_primary_dark="#1e293b",
        button_primary_background_fill="#7c3aed",
        button_primary_background_fill_hover="#6d28d9",
    ),
    css=css,
    fill_width=True
) as demo:

    gr.Markdown("""
    <div style="padding: 8px 0 4px 0;">
        <h1 style="margin:0; font-size:1.7rem; font-weight:650; color:#f8fafc; letter-spacing:-0.02em;">
            Image Classification
        </h1>
        <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.93rem;">
            Train EfficientNet models on any image folder · Local & private
        </p>
    </div>
    """)

    gr.Markdown(f"<div style='color:#64748b; font-size:0.8rem; margin-bottom:10px;'>{get_device_info()}</div>")

    with gr.Row(elem_classes=["main-row"], equal_height=False):

        # LEFT
        with gr.Column(scale=1, min_width=320):

            with gr.Group():
                gr.Markdown("#### Dataset")
                folder_input = gr.Textbox(
                    label="Folder path",
                    placeholder=r"D:\Datasets\my_dataset",
                    lines=1
                )
                analyze_btn = gr.Button("Analyze Folder", variant="primary")
                dataset_status = gr.Textbox(label="Status", value="No dataset loaded", interactive=False)

            with gr.Group():
                gr.Markdown("#### Training Settings")
                model_choice = gr.Dropdown(
                    choices=list(MODEL_MAP.keys()),
                    value="EfficientNet-B0",
                    label="Model"
                )
                with gr.Row():
                    epochs1 = gr.Number(value=6, label="Stage 1 Epochs", minimum=2, maximum=20, precision=0)
                    epochs2 = gr.Number(value=8, label="Stage 2 Epochs", minimum=2, maximum=25, precision=0)
                batch_size = gr.Slider(8, 64, value=32, step=8, label="Batch Size")
                train_btn = gr.Button("Start Training", variant="primary", size="lg")

            with gr.Accordion("Notes", open=True):
                gr.Markdown("""
- **EfficientNet-B0** → best speed / quality (recommended)
- **B1–B3** → higher accuracy, more VRAM
- Stage 1: train classifier head only
- Stage 2: fine-tune last blocks
- Class imbalance handled automatically
- 4 GB VRAM → use B0/B1, batch ≤ 32
                """)

        # RIGHT
        with gr.Column(scale=3):

            with gr.Group():
                gr.Markdown("#### Results")
                dataset_summary = gr.Markdown("*Analyze a folder to begin*")
                train_status = gr.Markdown("")
                model_status = gr.Textbox(label="Model", value="No model yet", interactive=False)

                with gr.Row():
                    acc_plot = gr.Plot(label="Accuracy")
                    cm_plot = gr.Plot(label="Confusion Matrix")

            with gr.Group():
                gr.Markdown("#### Live Prediction")
                with gr.Row():
                    input_img = gr.Image(type="pil", label="Upload image", height=300)
                    with gr.Column():
                        pred_out = gr.Markdown("Prediction will appear here")
                        pred_plot = gr.Plot(label="Confidence")
                predict_btn = gr.Button("Predict", variant="secondary")

    gr.Markdown("""
    <div style="text-align:center; color:#475569; font-size:0.78rem; margin-top:24px; padding-bottom:12px;">
        Made by <a href="https://github.com/yugisch" target="_blank" style="color:#818cf8;">yugisch</a>
        · Runs fully locally
    </div>
    """)

    analyze_btn.click(
        fn=analyze_folder,
        inputs=[folder_input],
        outputs=[dataset_summary, dataset_status]
    )

    train_btn.click(
        fn=train_model,
        inputs=[model_choice, epochs1, epochs2, batch_size],
        outputs=[train_status, acc_plot, cm_plot, model_status]
    )

    predict_btn.click(
        fn=predict_image,
        inputs=[input_img],
        outputs=[pred_out, pred_plot]
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
