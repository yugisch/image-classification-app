# Image Classification App

A clean, modular local web app to train **EfficientNet** models on any image classification dataset, with optional **Cleanlab** support for finding mislabelled images.

---

## Features

- Works with **any** classes (emotions, cars, animals, etc.)
- EfficientNet-B0 / B1 / B2 / B3
- Two-stage transfer learning
- Automatic class imbalance handling
- **Save & Load** trained models
- Optional **Cleanlab** – find and remove mislabelled images
- Creates a new cleaned dataset (original stays safe)
- Remembers last used folder path
- Training progress with ETA
- Export results as **CSV** or **PDF**
- Live prediction
- Fully local & private

---

## Project Structure

```
image-classification-app/
├── app.py              # Gradio UI
├── training.py         # Model training & prediction
├── cleaning.py         # Cleanlab logic
├── utils.py            # Helpers & transforms
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Structure

```
my_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

---

## Installation

```bash
# 1. Go to project folder
cd image-classification-app

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install PyTorch with GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Run the App

```bash
python app.py
```

Open: **http://localhost:7860**

---

## How to Use

1. Paste the full path to your dataset folder → click **Analyze Folder**
2. Choose model & epochs → click **Start Training**
3. (Optional) Use **Cleanlab** to find mislabelled images and create a cleaned dataset
4. Download the trained model (`.pth`) so you can load it later
5. Export results as CSV or PDF if needed
6. Use **Live Prediction** to test any image

---

## Cleanlab Notes

- Cleanlab needs a trained model first
- It creates a **new folder** (`your_dataset_cleaned/`) – original is never modified
- You can choose to analyze Train only / Test only / Both
- After cleaning, point the app to the new cleaned folder and train again

---

## Credits

Made by [yugisch](https://github.com/yugisch)  
Built with Grok  
Runs fully locally
