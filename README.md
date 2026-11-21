# Image Classification App

A clean local web application to train and test **EfficientNet** models on any image classification dataset.

Works with any classes (emotions, cars, animals, gender, etc.) as long as your folder has the correct structure.

---

## Features

- Copy and Paste the local dataset folder (no need to zip)
- Choose between EfficientNet-B0 / B1 / B2 / B3
- Two-stage transfer learning (Stage 1 + Stage 2)
- Automatic class imbalance handling
- Live prediction on your own images
- Accuracy curve + Confusion matrix
- Fully local & private

---

## Dataset Structure

Your dataset folder must look like this:

```
my_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   ├── class3/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    ├── class3/
    └── ...
```

---

## Requirements

- Windows / Linux / macOS
- Python 3.10 or higher
- NVIDIA GPU recommended (RTX 3050 Ti or better works well)

---

## Installation (Step by Step)

### 1. Open terminal in the project folder

```bash
cd path\to\image-classification-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of the line.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install PyTorch with GPU support (Recommended)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> If you only have CPU, use:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 6. Verify GPU is detected (optional but recommended)

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

You should see `CUDA available: True` and your GPU name.

---

## Running the App

Make sure the virtual environment is activated (`(venv)` is visible), then run:

```bash
python app.py
```

Open your browser and go to:

**http://localhost:7860**

---

## How to Use

1. **Dataset tab (left side)**
   - Copy paste the location of your dataset folder
   - Click **Analyze Folder**

2. **Training Settings**
   - Choose model (EfficientNet-B0 is recommended)
   - Set Stage 1 and Stage 2 epochs
   - Adjust Batch Size if needed (32 is good for 4–6 GB VRAM)
   - Click **Start Training**

3. **Results**
   - View accuracy curve and confusion matrix after training finishes

4. **Live Prediction**
   - Upload any image to see the predicted class + confidence

---

## Notes

- **Stage 1**: Trains only the final classifier layer (fast)
- **Stage 2**: Fine-tunes the last blocks of the network (improves accuracy)
- Class imbalance is handled automatically using class weights
- On 4 GB VRAM (e.g. RTX 3050 Ti), stay with EfficientNet-B0 or B1 and batch size ≤ 32

---

## Credits

Made by [yugisch](https://github.com/yugisch)  
Built with Grok Agent  
Runs fully locally
