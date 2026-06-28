# 🛰️ Multi-Sensor Satellite Data Fusion: Deep Generative SAR-to-Optical Translation

An end-to-end deep learning repository built in **PyTorch** that leverages deep convolutional generative networks to translate 1-channel Synthetic Aperture Radar (SAR) signals into high-fidelity, 3-channel synthetic optical (RGB) composites. 

## 💡 The Problem & The Solution
* **The Problem:** Optical satellite imaging (standard photography) is highly vulnerable to weather conditions; it is completely blinded by cloud cover, storms, and nighttime. 
* **The Solution:** Radar data (SAR) penetrates clouds, smoke, and darkness seamlessly, but the resulting images are grainy and difficult for humans or downstream algorithms to interpret. This framework bridges the gap by translating physical radar backscatter into human-readable visible light approximations, enabling **uninterrupted, 24/7 earth observation.**

---

## 🛠️ Tech Stack & Dependencies
* **Core Framework:** `PyTorch` (Neural network initialization, backpropagation, and tensor matrix calculations)
* **Image Processing:** `Pillow (PIL)` (Disk file I/O handling and image decoding)
* **Data Engineering:** `Torchvision` (Live vector scaling, image normalization, and transformation pipelines)
* **Numerical Computing:** `NumPy`

---

## 📁 Repository Structure & Module Design

The repository is built with a clean, decoupled production structure:

* 🧠 **`model.py`** – Holds the core deep learning architecture. Implements a Convolutional U-Net Generator with downsampling encoder blocks, bottleneck layers, and upsampling decoder blocks utilizing skip-connections to preserve sharp spatial topography.
* 💾 **`dataset.py`** – The data engineering pipeline. Seamlessly opens raw files via **Pillow**, normalizes pixel dimensions, and converts them into Torch Tensors scaled to `[-1, 1]` ranges.
* ⚙️ **`train.py`** – The central optimization loop. Compiles the model, passes dummy/real simulation matrices, applies an L1 reconstruction loss penalty, and optimizes weights using the Adam optimizer.
* 🔮 **`predict.py`** – The standalone inference interface. Loads generated network parameters (`.pth` weights checkpoint) to synthesize test predictions on completely unseen data.
* 📋 **`requirements.txt`** – System-wide environment blueprint.

---

## 🚀 Getting Started

### 1. Installation
Clone this repository and configure your local runtime workspace:
```bash
git clone [https://github.com/YOUR_USERNAME/satellite-sensor-fusion.git](https://github.com/YOUR_USERNAME/satellite-sensor-fusion.git)
cd satellite-sensor-fusion
pip install -r requirements.txt