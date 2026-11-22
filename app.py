"""
app.py
------
Gradio UI for the Image Classification app.
All heavy logic lives in training.py, cleaning.py and utils.py.
"""

import gradio as gr
from pathlib import Path

from utils import (
    get_device_info, find_dataset_root, analyze_dataset,
    load_last_folder, save_last_folder,
    export_results_csv, export_results_pdf
)
from training import train_model, load_checkpoint, predict, MODEL_MAP
from cleaning import find_label_issues, create_cleaned_dataset

# Global state
CURRENT_MODEL = None
CLASS_NAMES = None
DATASET_ROOT = None
LAST_RESULTS = None


def ui_analyze_folder(folder_path, progress=gr.Progress()):
    """Analyze dataset folder and show statistics."""
    global DATASET_ROOT

    if not folder_path or not str(folder_path).strip():
        raise gr.Error("Please enter a folder path.")

    try:
        root = find_dataset_root(folder_path)
        summary, stats = analyze_dataset(root)
        DATASET_ROOT = str(root)
        save_last_folder(str(root))
        status = f"✓ {len(stats['classes'])} classes · {stats['total_train'] + stats['total_test']:,} images"
        return summary, status
    except Exception as e:
        raise gr.Error(str(e))


def ui_train(model_name, epochs1, epochs2, batch_size, progress=gr.Progress()):
    """Train a model on the current dataset."""
    global CURRENT_MODEL, CLASS_NAMES, DATASET_ROOT

    if not DATASET_ROOT:
        raise gr.Error("Please analyze a dataset folder first.")

    def cb(pct, desc):
        progress(pct, desc=desc)

    result = train_model(
        dataset_root=DATASET_ROOT,
        model_name=model_name,
        epochs_stage1=int(epochs1),
        epochs_stage2=int(epochs2),
        batch_size=int(batch_size),
        progress_callback=cb
    )

    CURRENT_MODEL = result["model"]
    CLASS_NAMES = result["class_names"]

    global LAST_RESULTS
    LAST_RESULTS = result

    return (
        result["summary"],
        result["fig_acc"],
        result["fig_cm"],
        result["status"],
        result["model_path"]
    )


def ui_load_model(model_file, progress=gr.Progress()):
    """Load a previously saved .pth model."""
    global CURRENT_MODEL, CLASS_NAMES

    if model_file is None:
        raise gr.Error("Please upload a .pth model file.")

    progress(0.3, desc="Loading model...")
    try:
        model, class_names, model_name, accuracy = load_checkpoint(model_file)
        CURRENT_MODEL = model
        CLASS_NAMES = class_names

        acc_text = f" · Acc {accuracy:.2f}%" if accuracy is not None else ""
        status = f"✓ Loaded {model_name} ({len(class_names)} classes){acc_text}"
        info = f"**Model loaded successfully**\n\n- Model: `{model_name}`\n- Classes: {', '.join(class_names)}\n- Ready for prediction"

        progress(1.0, desc="Done")
        return status, info
    except Exception as e:
        raise gr.Error(str(e))


def ui_predict(image, progress=gr.Progress()):
    """Predict class of an uploaded image."""
    global CURRENT_MODEL, CLASS_NAMES

    if CURRENT_MODEL is None or CLASS_NAMES is None:
        raise gr.Error("Train or load a model first.")
    if image is None:
        raise gr.Error("Please upload an image.")

    progress(0.4, desc="Predicting...")
    result_text, fig = predict(CURRENT_MODEL, CLASS_NAMES, image)
    progress(1.0, desc="Done")
    return result_text, fig


def ui_cleanlab(scope, progress=gr.Progress()):
    """Run Cleanlab (or fallback) and create a cleaned dataset."""
    global CURRENT_MODEL, DATASET_ROOT

    if CURRENT_MODEL is None:
        raise gr.Error("Please train a model first (Cleanlab needs a trained model).")
    if not DATASET_ROOT:
        raise gr.Error("Please analyze a dataset folder first.")

    def cb(pct, desc):
        progress(pct, desc=desc)

    # 1. Find issues
    issues = find_label_issues(
        model=CURRENT_MODEL,
        dataset_root=DATASET_ROOT,
        scope=scope,
        progress_callback=cb
    )

    if issues["total_issues"] == 0:
        return "**No suspicious images found.** Your labels look clean!", ""

    # 2. Create cleaned dataset
    cleaned_path, summary = create_cleaned_dataset(
        dataset_root=DATASET_ROOT,
        issues=issues,
        progress_callback=cb
    )

    return summary, cleaned_path



def ui_export(format_choice, progress=gr.Progress()):
    """Export last training results as CSV or PDF."""
    global LAST_RESULTS
    if LAST_RESULTS is None:
        raise gr.Error("No training results to export. Train a model first.")

    progress(0.3, desc="Exporting...")
    summary = LAST_RESULTS["summary"]
    accuracy = LAST_RESULTS.get("accuracy", 0.0)
    fig_acc = LAST_RESULTS.get("fig_acc")
    fig_cm = LAST_RESULTS.get("fig_cm")

    # Extract report text roughly
    report_text = summary

    if format_choice == "CSV":
        path = export_results_csv(
            class_names=LAST_RESULTS.get("class_names", []),
            report_text=report_text,
            accuracy=accuracy,
            output_dir="."
        )
    else:
        path = export_results_pdf(
            summary_md=summary,
            fig_acc=fig_acc,
            fig_cm=fig_cm,
            accuracy=accuracy,
            output_dir="."
        )

    progress(1.0, desc="Done")
    return path


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
.main-row { gap: 16px !important; }
"""

theme = gr.themes.Base(
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
)

with gr.Blocks(title="Image Classification", fill_width=True) as demo:

    gr.Markdown("""
    <div style="padding: 8px 0 4px 0;">
        <h1 style="margin:0; font-size:1.7rem; font-weight:650; color:#f8fafc;">
            Image Classification
        </h1>
        <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.93rem;">
            Train EfficientNet models on any image folder · Optional Cleanlab · Local & private
        </p>
    </div>
    """)

    gr.Markdown(f"<div style='color:#64748b; font-size:0.8rem; margin-bottom:10px;'>{get_device_info()}</div>")

    with gr.Row(elem_classes=["main-row"], equal_height=False):

        # ---------- LEFT PANEL ----------
        with gr.Column(scale=1, min_width=320):

            with gr.Group():
                gr.Markdown("#### Dataset")
                folder_input = gr.Textbox(
                    label="Folder path",
                    placeholder=r"D:\Datasets\my_dataset",
                    value=load_last_folder(),
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

            with gr.Group():
                gr.Markdown("#### Load Model")
                model_upload = gr.File(label="Upload trained model (.pth)", file_types=[".pth"], type="filepath")
                load_btn = gr.Button("Load Model", variant="secondary")

            with gr.Group():
                gr.Markdown("#### Cleanlab (Optional)")
                clean_scope = gr.Dropdown(
                    choices=["Train only", "Test only", "Both"],
                    value="Both",
                    label="Analyze"
                )
                clean_btn = gr.Button("Find & Clean Mislabelled Images", variant="secondary")
                cleaned_path_box = gr.Textbox(label="Cleaned dataset path", interactive=False)

            with gr.Accordion("Notes", open=True):
                gr.Markdown("""
- **EfficientNet-B0** → best speed / quality (recommended)
- **B1–B3** → higher accuracy, more VRAM
- Stage 1: train classifier head only
- Stage 2: fine-tune last blocks
- Class imbalance handled automatically
- Cleanlab creates a **new** folder (original stays safe)
- 4 GB VRAM → use B0/B1, batch ≤ 32
                """)

        # ---------- RIGHT PANEL ----------
        with gr.Column(scale=3):

            with gr.Group():
                gr.Markdown("#### Results")
                dataset_summary = gr.Markdown("*Analyze a folder to begin*")
                train_status = gr.Markdown("")
                model_status = gr.Textbox(label="Model", value="No model yet", interactive=False)
                model_download = gr.File(label="Download Trained Model", interactive=False)

                with gr.Row():
                    export_format = gr.Dropdown(choices=["CSV", "PDF"], value="CSV", label="Export format")
                    export_btn = gr.Button("Export Results", variant="secondary")
                export_file = gr.File(label="Exported file", interactive=False)

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
        · Built with Grok · Runs fully locally
    </div>
    """)

    # ---------- Events ----------
    analyze_btn.click(
        fn=ui_analyze_folder,
        inputs=[folder_input],
        outputs=[dataset_summary, dataset_status]
    )

    train_btn.click(
        fn=ui_train,
        inputs=[model_choice, epochs1, epochs2, batch_size],
        outputs=[train_status, acc_plot, cm_plot, model_status, model_download]
    )

    load_btn.click(
        fn=ui_load_model,
        inputs=[model_upload],
        outputs=[model_status, train_status]
    )

    clean_btn.click(
        fn=ui_cleanlab,
        inputs=[clean_scope],
        outputs=[train_status, cleaned_path_box]
    )

    predict_btn.click(
        fn=ui_predict,
        inputs=[input_img],
        outputs=[pred_out, pred_plot]
    )

    export_btn.click(
        fn=ui_export,
        inputs=[export_format],
        outputs=[export_file]
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=css
    )
