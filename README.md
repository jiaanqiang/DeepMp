# 🧬 DeepMp: Microprotein Prediction Framework

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model](https://img.shields.io/badge/Model-CNN--BiLSTM--Attention-blue)

**Hybrid Deep Learning Model for Microprotein Identification**

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Training](#training)
- [Prediction](#prediction)
- [Data Format](#data-format)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Citation](#citation)

## Overview
DeepMp is a hybrid deep learning model combining CNN, BiLSTM, and Attention mechanisms for microprotein prediction from genomic sequences. The model extracts local features using CNN layers, captures sequence dependencies with BiLSTM, and focuses on important residues through attention mechanisms.

## Installation
```bash
# Clone repository
git clone https://github.com/deepbio/DeepMp.git
cd DeepMp

# Install dependencies
pip install torch>=2.0.0 numpy>=1.21.0 pandas>=1.3.0 biopython>=1.79 scikit-learn>=1.0.0 matplotlib>=3.5.0 tqdm>=4.64.0
