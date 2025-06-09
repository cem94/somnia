# Somnia Project: LLM Dataset Preparation, Tokenizer, and Training Pipeline

This repository contains the full pipeline for data preparation, custom tokenizer training, and transformer model training (TinyLLaMA-style) for the Somnia project, focused on narrative datasets (fairy tales and stories.) and developed as part of a CAS thesis.

## Key Features

- **Automatic download** of datasets from Kaggle.
- **Preprocessing**: cleaning, normalization, and chunking of texts.
- **Dataset preparation** in `.jsonl` format for training.
- **Custom tokenizer training** (Byte-Pair Encoding) with special token management.
- **Data analysis** with statistics and plots.
- **TinyLLaMA-style transformer model** with rotary embeddings, RMSNorm, KV caching, checkpointing, and text generation.
- **Automated hyperparameter optimization** with resume and progressive saving.
- **Advanced logging** to file and console.

## Project Structure

```
├── data/
│   ├── fetch/           # Scripts to download data from Kaggle
│   ├── processed/       # Scripts to prepare data for training
│   └── analysis/        # Data analysis and visualization
├── export_device/
│   ├── export_model.py  # Android export script (TorchScript)
│   └── android_model/   # Android export output
├── logs/                # Execution logs
├── model/
│   ├── tokenizer/       # Trained tokenizer
│   ├── transformer/     # Model, config, dataset, tokenizer
│   ├── metrics.py       # Training metrics tracking and plotting
│   ├── train_model.py   # Main training loop
│   ├── train_tokenizer.py # Custom tokenizer training
│   ├── hyperparameter.py # Hyperparameter optimization
├── utility/
│   ├── logger.py        # Centralized logger
│   └── paths.py         # Path and directory management
├── main.py              # Pipeline entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cem94/somnia.git
   cd somnia
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle credentials:**
   - Download your `kaggle.json` from your Kaggle account.
   - Place it in `~/.kaggle/kaggle.json`.

## Usage

### 1. Full pipeline (recommended)

Run the entire pipeline (data processing, tokenizer, model training):

```bash
python main.py
```

### 2. Download and prepare data

Download datasets from Kaggle:
```bash
python data/fetch/fetch_data.py
```

Prepare the dataset for training:
```bash
python data/processed/prepare_dataset.py
```

### 3. Data analysis

Generate statistics and plots:
```bash
python data/analysis/analysis.py
```

### 4. Custom tokenizer training

Train the tokenizer:
```bash
python model/train_tokenizer.py
```

### 5. Model training

Start model training (with or without hyperparameter search):
```bash
python model/hyperparameter.py
```
Or standard training only:
```bash
python model/train_model.py
```

### 6. Hyperparameter optimization

To perform automated hyperparameter search and final training with the best configuration, run:
```bash
python model/hyperparameter.py
```

- The search space is defined in `model/hyperparameter.py` (`HyperparameterSearchSpace` class).
- Intermediate and final results are saved in `model/hyperparameters/` and `model/output/`.
- You can customize the search space by editing the fields in `HyperparameterSearchSpace`.

### 7. Export model for Android

After training, you can export the model and tokenizer for Android deployment (optimized TorchScript):

```bash
python export_device/export_model.py
```

- The script loads the checkpoint from `model/output/llama_model.pt` and tokenizer from `model/tokenizer/`.
- The model is wrapped for mobile inference and exported in TorchScript format.
- Exported files (model, tokenizer, config) are saved in `export_device/android_model/`.
- Android-specific metadata is included for easy integration.

You can integrate these files directly into your Android app for on-device inference.

## Output

- **Model and checkpoints:** `model/output/` or `model/hyperparameters/`
- **Tokenizers:** `model/tokenizer/`
- **Android export:** `export_device/android_model/`
- **Plots and metrics:** `model/plots/`
- **Logs:** `logs/`

## About

This project implements a TinyLLaMA-style transformer (adapted from [train-tiny-llm](https://github.com/FareedKhan-dev/train-tiny-llm)) and is developed as part of a CAS thesis on efficient LLM training and data pipelines for narrative text.

## Contributing

Contributions and suggestions are welcome!  
Open an issue or pull request to propose improvements.

---

**Note:**  
Make sure you have enough disk space for data and models.  
For issues or questions, check the logs in `logs/` or open an issue.