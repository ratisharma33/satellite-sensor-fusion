# Satellite Multi-Sensor Data Fusion: Deep Generative SAR-to-Optical Translation

An end-to-end deep learning repository built using PyTorch that leverages convolutional generative networks to translate 1-channel Synthetic Aperture Radar (SAR) signals into realistic 3-channel synthetic optical (RGB) composites.

## 📁 Project Structure

* `model.py`: Embedded architecture logic specifying the encoder-decoder translation matrices.
* `dataset.py`: Structured disk imaging handler with live vector pixel scaling pipelines.
* `train.py`: Optimization runtime sequence utilizing spatial reconstruction constraints.
* `predict.py`: Standalone compilation script running feedforward synthesis loops.
* `requirements.txt`: Execution platform dependencies and setup blueprints.

## 🛠️ Getting Started

1. Set up dependencies:
```bash
pip install -r requirements.txt
