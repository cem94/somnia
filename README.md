# Somnia Project: LLM Dataset Preparation, Tokenizer, and Training Pipeline

This repository contains the complete pipeline for data preparation, custom tokenizer training, and transformer model training (TinyLLaMA-style) for the Somnia project, focused on narrative datasets (fairy tales and stories) and developed as part of a CAS thesis.

## Key Features

- **Automatic download** of datasets from Kaggle with credential management
- **Robust preprocessing** with cleaning, normalization, and intelligent text chunking
- **Dataset preparation** in `.jsonl` format optimized for training
- **Custom BPE tokenizer training** with special token management and evaluation
- **Comprehensive data analysis** with statistics, visualizations, and quality metrics
- **TinyLLaMA-style transformer model** with:
  - Rotary positional embeddings (RoPE)
  - RMS normalization for stability
  - Multi-head attention with KV caching
  - Feed-forward networks with SiLU activation
  - Mixed precision training support
- **Advanced hyperparameter optimization** with targeted search and resume capability
- **Robust checkpoint system** with automatic recovery and state management
- **Comprehensive logging** with structured output and debug information
- **Model export capabilities** for Android deployment via TorchScript

## Project Structure

```
├── data/
│   ├── fetch/           # Kaggle dataset download and organization
│   ├── processed/       # Data preprocessing and cleaning pipelines
│   └── analysis/        # Statistical analysis and visualization tools
├── export_device/
│   ├── export_model.py  # Android export script (TorchScript format)
│   └── android_model/   # Exported model files for mobile deployment
├── logs/                # Centralized logging with rotation and filtering
├── model/
│   ├── tokenizer/       # Custom trained BPE tokenizer with special tokens
│   ├── transformer/     # Complete transformer implementation
│   │   ├── model.py     # SomniaTransformer with attention and generation
│   │   ├── llama_config.py # Comprehensive model configuration
│   │   ├── dataset.py   # PyTorch dataset with robust tokenization
│   │   └── tokenizer_config.py # Tokenizer configuration and validation
│   ├── metrics.py       # Training metrics tracking with visualization
│   ├── train_model.py   # Main training loop with cosine scheduling
│   ├── train_tokenizer.py # BPE tokenizer training pipeline
│   └── hyperparameter.py # Systematic hyperparameter optimization
├── utility/
│   ├── logger.py        # Centralized logging with file and console output
│   └── paths.py         # Path management and directory structure
├── main.py              # Pipeline orchestration and stage management
├── requirements.txt     # Python dependencies with version specifications
└── README.md           # Project documentation and usage guide
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cem94/somnia.git
   cd somnia
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle credentials:**
   - Create account at [kaggle.com](https://www.kaggle.com)
   - Download your `kaggle.json` API token from Account → API
   - Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\{username}\.kaggle\kaggle.json` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### 1. Complete Pipeline (Recommended)

Execute the entire pipeline with optimized defaults:

```bash
python main.py
```

**Pipeline stages:**
- **Stage 1:** Data fetching, preprocessing, and analysis
- **Stage 2:** Custom tokenizer training and evaluation  
- **Stage 3:** Model training with hyperparameter optimization
- **Stage 4:** Model export for deployment

### 2. Individual Components

#### Data Management
```bash
# Download datasets from Kaggle
python data/fetch/fetch_data.py

# Preprocess raw data for training
python data/processed/prepare_dataset.py

# Generate analysis and visualizations
python data/analysis/analysis.py
```

#### Tokenizer Training
```bash
# Train custom BPE tokenizer
python model/train_tokenizer.py
```

#### Model Training
```bash
# Standard training with default hyperparameters
python model/train_model.py

# Hyperparameter optimization with extended final training
python model/hyperparameter.py
```

#### Model Export
```bash
# Export trained model for Android deployment
python export_device/export_model.py
```

### 3. Hyperparameter Optimization

The system includes sophisticated hyperparameter optimization:

```bash
python model/hyperparameter.py
```

**Features:**
- **Targeted search space** optimized for ~80MB model size
- **Progressive trial execution** with persistent state management
- **Automatic resume** capability for interrupted runs
- **Extended final training** with optimal configuration (50 epochs)
- **Comprehensive result tracking** and analysis

**Search space includes:**
- Architecture parameters (dim, layers, heads, kv_heads)
- Training parameters (learning rate, batch size, dropout)
- Optimization settings (accumulation steps, gradient clipping)

### 4. Configuration Customization

**Model architecture** (`model/transformer/llama_config.py`):
```python
config = LLamaConfig(
    dim=512,              # Embedding dimension
    n_layers=8,           # Number of transformer layers
    n_heads=8,            # Attention heads
    n_kv_heads=2,         # Key-value heads for efficient attention
    learning_rate=5e-4,   # Training learning rate
    batch_size=8,         # Training batch size
    device='cuda'         # Training device
)
```

**Tokenizer settings** (`model/transformer/tokenizer_config.py`):
```python
VOCAB_SIZE = 20000      # Vocabulary size
MAX_SEQ_LEN = 2048      # Maximum sequence length
```

## Output Structure

- **Trained models:** `model/output/` (final model and checkpoints)
- **Hyperparameter trials:** `model/hyperparameters/` (optimization results)
- **Custom tokenizer:** `model/tokenizer/` (BPE tokenizer files)
- **Android export:** `export_device/android_model/` (TorchScript format)
- **Training metrics:** `model/plots/` (loss curves, visualizations)
- **Data analysis:** `data/plots/` (dataset statistics and insights)
- **Execution logs:** `logs/` (structured logging with rotation)

## Model Architecture

The **SomniaTransformer** implements a modern, efficient transformer architecture:

- **Multi-head attention** with rotary positional embeddings (RoPE)
- **Root Mean Square normalization** for training stability
- **Feed-forward networks** with SiLU activation and gating
- **KV caching** for efficient text generation
- **Mixed precision training** with automatic loss scaling
- **Gradient accumulation** for effective large batch training
- **Cosine learning rate scheduling** with linear warmup

**Estimated model sizes:**
- Small config (~60MB): 384 dim, 8 layers
- Medium config (~80MB): 512 dim, 8 layers  
- Large config (~100MB): 640 dim, 10 layers

## Advanced Features

### Checkpoint Management
- **Automatic checkpoint saving** during training
- **Resume capability** from any checkpoint
- **Best model tracking** throughout training
- **Metadata preservation** for reproducibility

### Text Generation
- **Multiple sampling strategies** (temperature, top-p, repetition penalty)
- **Efficient KV caching** for faster generation
- **Special token handling** (BOS, EOS, padding)
- **Configurable generation parameters**

### Training Monitoring
- **Real-time metrics tracking** (loss, perplexity, learning rate)
- **Comprehensive visualizations** with training progress plots
- **Training interruption detection** and recovery
- **Memory usage monitoring** and optimization

## Requirements

- **Python:** 3.8 or higher
- **PyTorch:** 2.0+ with CUDA support (recommended)
- **Storage:** ~5GB for datasets and models
- **Memory:** 8GB+ RAM (16GB+ for larger models)
- **GPU:** Optional but recommended for training (4GB+ VRAM)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow the existing code style** and documentation patterns
3. **Add tests** for new functionality where applicable
4. **Update documentation** for any API changes
5. **Submit a pull request** with clear description of changes

## License

This project is developed as part of a CAS thesis. Please refer to the institution's guidelines for usage and distribution.

## Acknowledgments

- **Base architecture** inspired by [train-tiny-llm](https://github.com/FareedKhan-dev/train-tiny-llm)
- **Datasets** sourced from Kaggle community contributors
- **Transformer architecture** based on LLaMA and similar modern designs

## Troubleshooting

### Common Issues

1. **Kaggle API errors:**
   - Verify `kaggle.json` placement and permissions
   - Check internet connectivity and API quota

2. **CUDA/GPU issues:**
   - Verify PyTorch CUDA installation: `torch.cuda.is_available()`
   - Reduce batch size if encountering OOM errors

3. **Training interruptions:**
   - Use checkpoint resume functionality
   - Check logs in `logs/` directory for detailed error information

4. **Memory issues:**
   - Reduce model size or batch size
   - Enable gradient checkpointing for memory efficiency

### Getting Help

- **Check the logs:** Detailed execution logs are available in `logs/`
- **Review configurations:** Ensure all paths and parameters are correctly set
- **Open an issue:** For bugs or feature requests, create a GitHub issue

---

**Note:** Ensure sufficient disk space for datasets, models, and logs. For optimal performance, use a GPU-enabled environment with adequate VRAM for your chosen model configuration.