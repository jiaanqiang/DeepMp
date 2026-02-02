# 🧬 DeepMp: A deep learning framework for genome-wide microprotein identification

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model](https://img.shields.io/badge/Model-CNN--BiLSTM--Attention-blue)

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [High-confidence Dataset](#high-confidence-dataset)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Citation](#citation)
- [Contact](#contact)

## Overview
DeepMp is a hybrid deep learning framework that combines Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and Attention mechanisms for accurate microprotein (5-100 amino acids) prediction from genomic sequences. The model was trained on a **high-confidence reference set of 13,497 non-redundant microproteins** rigorously validated by extensive experimental evidence.

### Key Features
- **Hybrid Architecture**: 4 CNN layers + BiLSTM + Attention mechanism
- **High-confidence Training**: 13,497 experimentally validated microproteins
- **Multi-species Data**: 6,828 Arabidopsis + 6,816 rice + 123 public database entries
- **Rigorous Validation**: 62 Ribo-seq datasets + 89 proteomic datasets
- **Easy-to-use**: Simple command line interface and Python API

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.x (optional, for GPU acceleration)

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/deepbio/DeepMp.git
cd DeepMp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch (if not already installed)
# For CPU only:
pip install torch torchvision torchaudio

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "from deepmp import __version__; print(f'DeepMp version: {__version__}')"
