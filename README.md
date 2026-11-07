# ScreenGuard AI — Screen Damage Detection

ScreenGuard AI is a web application that accepts a photo of a phone, tablet, or monitor display and classifies it as **Normal** or **Cracked/Damaged** using a transfer-learning image classifier.

## Contents

- `frontend/` — React 18 + Vite user interface
- `backend/` — FastAPI prediction API
- `model/train.py` — MobileNetV2 transfer-learning training script
- `data/` — expected local training-data structure (add your images here)
- `requirements.txt` — Python dependencies

## Features

- Upload or drag-and-drop JPG, PNG, and WEBP images
- Preview the selected image before analysis
- FastAPI endpoint for image prediction
- MobileNetV2 transfer-learning training pipeline
- Confidence score and practical next-step guidance
- Input validation, upload size limit, health endpoint, and clear model-not-ready errors
- Responsive React interface

## 1. Requirements

Recommended:
- Python **3.10 or 3.11** (TensorFlow compatibility is important)
- Node.js **18 or newer**
- npm
- Internet connection for the first training run if ImageNet weights need to be downloaded
- A labeled image dataset

A GPU is optional. CPU training works but can be slow.

## 2. Dataset preparation

Place real, correctly labeled screen photographs in this structure:

```text
data/
├── normal/
│   ├── normal_001.jpg
│   └── ...
└── cracked/
    ├── cracked_001.jpg
    └── ...
```

Use photographs of actual device screens, not generic photographs of broken glass, wallpapers, screenshots, or images containing unrelated cracks. Include different devices, lighting conditions, viewing angles, backgrounds, and damage severities.

Aim for a reasonably balanced dataset. Keep near-duplicate photos of the same device/session in only one split to avoid data leakage. The training script uses a stratified-style directory split provided by Keras; for a serious evaluation, create a separate device/session-level test set and report its metrics.

## 3. Backend setup

From the project root:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Start the API:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs  
Health check: http://127.0.0.1:8000/health

The API reports `model_loaded: false` until you train and save a model.

## 4. Train the model

After preparing the dataset, run from the project root:

```powershell
python model/train.py
```

The script:
1. Loads images from `data/normal` and `data/cracked`.
2. Resizes images to 224×224.
3. Applies training-only augmentation.
4. Trains a MobileNetV2 transfer-learning classifier.
5. Fine-tunes the final layers.
6. Saves the best model to `model/screen_guard_model.keras`.

Then restart the backend. Confirm `/health` returns `model_loaded: true`.

Training quality depends on dataset quality, class balance, and device diversity. Do not report training accuracy as real-world accuracy. Evaluate on an independent test set before presenting performance claims.

## 5. Frontend setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://localhost:5173).

The Vite development proxy forwards `/api/*` requests to `http://127.0.0.1:8000`.

## 6. Run the full project

Terminal 1 — project root, virtual environment active:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173, upload a screen photo, and select **Analyze screen**.

## API

### `GET /health`

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "model/screen_guard_model.keras"
}
```

### `POST /predict`

Multipart form field: `file` (image).

Example response:

```json
{
  "status": "success",
  "prediction": "Cracked",
  "confidence": 91.42,
  "normal_probability": 8.58,
  "cracked_probability": 91.42,
  "severity": "Possible damage",
  "recommendation": "Inspect the display and consider professional assessment."
}
```

The model produces a binary image classification. It does **not** locate cracks, measure physical damage, determine the device's remaining lifespan, or provide a guaranteed repair quote.

## Project structure

```text
ScreenGuard-AI/
├── backend/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── normal/
│   └── cracked/
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── main.jsx
├── model/
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Troubleshooting

- **Model not loaded:** run `python model/train.py`, verify `model/screen_guard_model.keras` exists, and restart FastAPI.
- **Dataset directory missing:** create both `data/normal/` and `data/cracked/` and add labeled images.
- **Only one class found:** check folder spelling and ensure both classes contain supported image files.
- **TensorFlow install error:** use Python 3.10 or 3.11 in a fresh virtual environment.
- **Frontend cannot connect:** start FastAPI on port 8000 and Vite on port 5173.
- **Prediction quality is poor:** add diverse, correctly labeled images; check class imbalance and evaluate on unseen devices. A model can learn irrelevant backgrounds or reflections if the dataset is not representative.

## Responsible use

This is an educational prototype, not a certified diagnostic or repair system. A photo may not show internal display faults, touch issues, or damage visible only when the screen is powered on. Use professional inspection for repair decisions. Do not present model confidence as a probability that the device is physically damaged unless it has been calibrated and validated on an independent representative dataset.
