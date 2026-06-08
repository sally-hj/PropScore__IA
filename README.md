# PropScore_AI

PropScore_AI is a Streamlit dashboard and multimodal prospect-scoring project for the classes:

- **Hot**
- **Warm**
- **Cold**

The app can work from:

- tabular data
- processed multimodal feature CSVs
- uploaded videos
- live camera snapshots / webcam input

This repository already includes the trained model files and the processed CSVs needed to open the dashboard.

---

## What is included in the launch bundle

The zip file that accompanies this repository should include only the files needed to run the project:

- `app.py`
- `prop_score_ai/`
- `utils/`
- `scripts/`
- `configs/`
- `models/`
- `data/processed/`
- `features/`
- `outputs/metrics/`
- `tensorflow/__init__.py`
- `requirements.txt`
- this `README.md`

It excludes:

- notebooks
- cache files
- logs
- raw video folders
- virtual environments

---

## Requirements

- **Python 3.10+** (Python 3.12 is recommended and works with the included code)
- **FFmpeg** installed and available on your `PATH`
- Camera / microphone permissions in your browser if you want to use live camera or upload audio/video flows

> The current live camera pipeline does **not** require TensorFlow at runtime.

---

## Step-by-step installation

### 1) Unzip the project

Extract the zip file anywhere you want.

Example:

```bash
unzip PropScore_AI_launch_bundle.zip
cd PropScore_AI
```

If you are on Windows, just unzip the archive and open a terminal inside the extracted folder.

---

### 2) Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
```

**Windows**

```bash
python -m venv .venv
```

---

### 3) Activate the virtual environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```bat
.venv\Scripts\activate.bat
```

---

### 4) Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### 5) Install dependencies

```bash
pip install -r requirements.txt
```

If installation is slow, that is normal because the project uses packages such as:

- `streamlit`
- `opencv-python`
- `mediapipe`
- `xgboost`
- `transformers`
- `torch`
- `librosa`
- `whisper`

---

### 6) Install FFmpeg

FFmpeg is required for video/audio processing.

**macOS**

```bash
brew install ffmpeg
```

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**

Install FFmpeg manually and add it to your `PATH`.

You can verify the installation with:

```bash
ffmpeg -version
```

---

### 7) Launch the app

```bash
streamlit run app.py
```

Streamlit will print a local URL such as:

```text
http://localhost:8501
```

Open that URL in your browser.

---

### 8) Allow camera / microphone permissions

If you use:

- **Live Camera**
- **Video Upload Prediction**
- **Audio-related pipeline steps**

your browser may ask for permissions. Click **Allow** when prompted.

---

## Recommended launch command

If you want to be explicit with the Python executable inside the virtual environment:

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

You can also change the port if 8501 is busy:

```bash
python -m streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

---

## Useful commands

### Check dataset leakage

```bash
python scripts/check_dataset_leakage.py --input data/processed/prospects.csv
```

### Prepare the tabular dataset

```bash
python scripts/00_prepare_tabular_dataset.py --input data/processed/dataset_final_catboost.csv
```

### Run the full pipeline

```bash
python scripts/run_full_pipeline.py --tabular-csv data/processed/dataset_final_catboost.csv
```

### Train the XGBoost model

```bash
python scripts/10_train_xgboost.py --dataset data/processed/fused_dataset.csv
```

If `data/processed/fused_dataset.csv` is missing, the training script falls back to `data/processed/prospects.csv`.

---

## Important files used by the dashboard

### Prediction / model pages

- `models/xgboost_model.pkl`
- `models/encoders.pkl`
- `models/feature_schema.json`

### Dataset / tabular pages

- `data/processed/prospects.csv`
- `data/processed/dataset_final_catboost.csv`

### Multimodal feature pages

- `features/hot.csv`
- `features/warm.csv`
- `features/cold.csv`
- `features/ravdess.csv`

### Metrics / performance pages

- `outputs/metrics/metrics.json`
- `outputs/metrics/classification_report.txt`
- `outputs/metrics/confusion_matrix.png`
- `outputs/metrics/leakage_report.txt`
- `outputs/metrics/tabular_dataset_report.json`

---

## Notes

- The app resolves its internal file paths relative to the project folder, so you can unzip the bundle into any location.
- If a file is missing, the dashboard usually shows a warning instead of crashing.
- For best results, always run the project from the extracted folder root.

