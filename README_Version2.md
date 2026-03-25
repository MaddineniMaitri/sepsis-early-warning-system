# Deep Learning-based Continuous Monitoring and Early Warning System for Late-Onset Sepsis in Preterm Neonates

## Overview

This project implements a comprehensive early warning system for detecting late-onset sepsis (LOS) in preterm neonates using deep learning models. The system combines multiple advanced techniques including GRU networks, Swin Transformers, signal processing, and machine learning-based imputation.

## Features

- **Dataset Generation**: Synthetic dataset generator for preterm neonate vital signs and lab values
- **Data Preprocessing**:
  - Z-score normalization (standardization)
  - KNN and iterative (MICE) imputation for missing values
  - Butterworth low-pass filtering for noise reduction
  - Sliding window creation for time series
  
- **Deep Learning Models**:
  - **GRU Model**: Gated Recurrent Unit with attention mechanism
  - **Swin Transformer**: Shifted window attention-based model
  - **Hybrid Model**: Ensemble combining GRU and Swin Transformer

- **Training Pipeline**: Complete training with validation, early stopping, and model evaluation

## Dataset Features

The synthetic dataset includes 10 clinical features:

1. **Heart Rate** (bpm): 120-180 normal, increases with sepsis
2. **Respiratory Rate** (breaths/min): 40-60 normal, increases with sepsis
3. **Temperature** (°C): 36.5-37.4 normal, becomes unstable with sepsis
4. **Systolic Blood Pressure** (mmHg): 40-70 normal, decreases with sepsis
5. **O2 Saturation** (%): 90-95% normal, decreases with sepsis
6. **WBC Count** (×10³/μL): 9-30 normal, increases with sepsis
7. **I:T Ratio**: <0.2 normal, increases with sepsis
8. **CRP** (mg/L): <10 normal, increases with sepsis
9. **Procalcitonin** (ng/mL): <0.5 normal, increases with sepsis
10. **Platelet Count** (×10³/μL): 150-450 normal, decreases with sepsis

## Project Structure

```
sepsis-early-warning-system/
├── README.md
├── requirements.txt
├── generate_dataset.py      # Synthetic dataset generation
├── preprocessing.py         # Data preprocessing functions
├── models.py               # GRU, Swin Transformer, Hybrid models
├── train.py               # Training pipeline
└── data/                  # Generated datasets (created on first run)
    ├── X_data.npy
    ├── y_labels.npy
    ├── dataset_flattened.csv
    └── metadata.csv
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MaddineniMaitri/sepsis-early-warning-system.git
cd sepsis-early-warning-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Dataset

```bash
python generate_dataset.py
```

This creates synthetic vital signs data for 1000 preterm neonates (700 healthy, 300 with sepsis).

### 2. Train Models

```bash
python train.py
```

This trains three models:
- GRU Model
- Swin Transformer Model
- Hybrid Model (Ensemble)

## Model Architecture

### GRU Model
- Input: (batch_size, seq_len=100, input_dim=10)
- GRU layers: 2 layers with 64 hidden units
- Attention mechanism over sequence
- FC layers: 64 → 32 → 1
- Output: Binary classification (Normal/Sepsis)

### Swin Transformer Model
- Input: (batch_size, seq_len=100, input_dim=10)
- Input projection: 10 → 64 dimensions
- Positional encoding added
- 2 Swin Transformer blocks
- Global average pooling
- FC layers: 64 → 32 → 1

## Key Techniques

1. **Z-Score Normalization**: Standardizes features to mean=0, std=1
2. **KNN Imputation**: Estimates missing values using k-nearest neighbors
3. **Iterative Imputation (MICE)**: ML-based iterative regression for missing values
4. **Butterworth Low-Pass Filter**: Removes high-frequency noise
5. **GRU**: Processes sequential vital signs with attention
6. **Swin Transformer**: Shifted window self-attention for efficiency

## License

MIT License

## Contact

For questions, contact: MaddineniMaitri