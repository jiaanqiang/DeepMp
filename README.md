# 🧬 DeepMp: A deep learning framework for genome-wide microprotein identification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Architecture](https://img.shields.io/badge/Architecture-4CNN--BiGRU--Attention-blue)

**A Hybrid Deep Learning Model for Microprotein Identification (5-100 aa)**

</div>

## 🏗️ Model Architecture

DeepMp is a hybrid neural network combining three key components:

### 1. 4-Layer 1D-CNN Block
- **4 convolutional layers** with 3×1 kernels
- **ReLU activation** and **MaxPooling** (kernel size=2)
- **Purpose**: Extract local sequence motifs

### 2. 6-Layer Bi-GRU
- **Bidirectional Gated Recurrent Units** (6 layers)
- **Dropout rate**: 0.2
- **Purpose**: Model long-range sequence dependencies

### 3. 16-Head Self-Attention
- **16 attention heads** with dropout=0.2
- **Purpose**: Weight functionally critical residues
- **Output**: Focus on important amino acids

### Final Classification
- **Fully connected layer** with sigmoid activation
- **Binary output**: Microprotein (1) or Non-microprotein (0)

### Input Processing
- **Sequence length**: Standardized to 100 residues
- **Padding**: Zero-padding for shorter sequences
- **Amino acid encoding**: 20 standard amino acids (1-20)

### Optimized Hyperparameters
- **Learning rate**: 1e-5
- **Weight decay**: 1e-5
- **Batch size**: 32
- **Optimizer**: AdamW with cosine scheduler

## 📦 Quick Installation

```bash
# Clone the repository
git clone https://github.com/deepbio/DeepMp.git
cd DeepMp

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas biopython scikit-learn

# Or install all at once
pip install -r requirements.txt
