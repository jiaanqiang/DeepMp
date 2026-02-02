# 🚀 DeepMp: Microprotein Prediction Deep Learning Framework

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)
![GitHub Actions](https://img.shields.io/github/actions/workflow/status/deepbio/DeepMp/tests.yml)

**A deep learning framework for systematic identification and prediction of microproteins**

[Quick Start](#-quick-start) | [Installation](#-installation) | [Model Training](#-model-training) | [Prediction](#-prediction) | [Fine-tuning](#-fine-tuning) | [Citation](#-citation)

</div>

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#-features)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Prediction with Pre-trained Model](#1-prediction-with-pre-trained-model)
  - [Model Training](#2-model-training)
  - [Fine-tuning for New Species](#3-fine-tuning-for-new-species)
  - [Evaluation](#4-evaluation)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [API Reference](#-api-reference)
- [Command Line Interface](#-command-line-interface)
- [Tutorials](#-tutorials)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

## Overview

**DeepMp** is a deep learning framework designed for the identification and prediction of microproteins (small proteins typically <100 amino acids). The framework enables systematic screening of genomically encoded microproteins through deep learning–based prediction.

### Key Advantages
- **Species Adaptability**: Core model trained on Arabidopsis and rice datasets with strong cross-species performance
- **Transfer Learning**: Easy fine-tuning for new plant species
- **High Accuracy**: State-of-the-art performance across key evaluation metrics
- **Interpretability**: Attention mechanisms provide biological insights

### Supported Tasks
- Microprotein identification from genomic sequences
- Functional annotation prediction
- Subcellular localization
- Conservation score calculation
- Novel microprotein discovery

## ✨ Features

- **🎯 Multi-species Support**: Pre-trained on Arabidopsis thaliana and Oryza sativa
- **🔄 Transfer Learning**: Easy adaptation to new plant species
- **📊 Comprehensive Evaluation**: AUC, F1-score, Precision, Recall metrics
- **🔬 Biological Interpretability**: Attention visualization for important residues
- **⚡ High Performance**: GPU-accelerated training and inference
- **📈 Progressive Learning**: Support for incremental dataset addition
- **🌐 Web Interface**: Optional Flask-based web application
- **📁 Multiple Formats**: Support for FASTA, GenBank, GFF3 inputs

## 📊 Datasets

### Pre-trained Model Datasets
| Species | Dataset Size | Positive Samples | Negative Samples | Source |
|---------|--------------|------------------|------------------|--------|
| Arabidopsis thaliana | 15,842 | 7,921 | 7,921 | [TAIR](https://www.arabidopsis.org/) |
| Oryza sativa | 12,573 | 6,286 | 6,287 | [RGAP](http://rice.plantbiology.msu.edu/) |

### Available Features
- Sequence-based features (k-mer frequencies, amino acid composition)
- Evolutionary conservation scores
- Physicochemical properties
- Structural propensity predictions
- Domain and motif information

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)
```bash
# Pull the Docker image
docker pull deepbio/deepmp:latest

# Run with GPU support
docker run --gpus all -p 8080:8080 deepbio/deepmp

# Access the web interface
# Open http://localhost:8080 in your browser
