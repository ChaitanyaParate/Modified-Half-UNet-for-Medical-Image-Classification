# 🧠 Brain Tumor Classification using Modified Half-U-Net

This repository presents a **research-oriented implementation** of a modified **Half-U-Net architecture** designed for **binary classification** of brain MRI scans — detecting the presence or absence of tumors.  
Unlike traditional U-Net models that perform segmentation, this model repurposes the encoder (contracting) path of Half U-Net and couples it with a custom fully connected classification head.

---

## 🚀 Abstract

The Half U-Net architecture has been widely used in biomedical image segmentation due to its encoder–decoder structure. However, for classification tasks, the decoder section becomes redundant and computationally expensive.  
This work proposes a **Half-U-Net** that retains only the **encoder path** of U-Net and integrates a **custom linear classification module**. The model is trained end-to-end on MRI brain scans for tumor detection, achieving high accuracy with efficient computation.

---

## 🧩 Key Features

- 🧠 **Half-U-Net backbone** — encoder-only U-Net adapted for classification  
- ⚙️ **Dynamic classifier initialization** — input dimensions auto-detected during first forward pass  
- 📉 **Binary classification** (`tumor` / `no tumor`)  
- 🔄 **Augmentation** via Albumentations (rotation, flip, normalization)  
- 🪄 **Adaptive learning rate** using `ReduceLROnPlateau` scheduler  
- 💾 **Checkpointing** and `.pth` model export for later inference  

---

## 🗂 Dataset

**Dataset:** [Brain Tumor Detection – Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

The dataset consists of MRI images categorized into:
```
Data/
├── yes/   # MRI images with tumor
└── no/    # MRI images without tumor
```

Each image is resized to `315×315` and normalized to `[0, 1]`.

---

## 🧪 Training Details

| Parameter | Value |
|------------|--------|
| Epochs | 40 |
| Batch Size | 8 |
| Learning Rate | 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Loss Function | BCEWithLogitsLoss |
| Optimizer | Adam |
| Image Size | 315×315 |
| Device | CUDA / CPU |

---

## 📈 Results

| Metric | Value |
|--------|--------|
| Accuracy | **≈ 98-99 %** |
| Best Threshold | 0.5 |
| Loss | ~0.03 (final epochs) |

> The model demonstrates strong classification accuracy and robust convergence despite limited data, validating the Half-U-Net architecture’s capability for binary classification tasks.

---

## 🔬 Research Contribution

This implementation demonstrates that:
- Encoder-only CNNs derived from segmentation networks can effectively perform classification.  
- The **Half-U-Net** retains contextual feature extraction while being lighter and faster than full U-Net.  
- Dynamic linear head initialization ensures architecture flexibility across different input resolutions.

---

## 🧠 Inference Example

```bash
python predict.py
```

## 📁 File Structure

```
classification_Half_U-NET/
│
├── model.py              # Model architecture
├── train.py              # Training loop
├── dataset.py            # Custom dataset loader
├── utils.py              # Helper functions & checkpoint handling
├── convert_checkpoint.py # Convert .tar → .pth
├── predict.py            # Inference on new MRI images
└── README.md             # Project documentation
```

## 🧰 Requirements

```bash
torch
torchvision
albumentations
tqdm
matplotlib
opencv-python
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Author

**Chaitanya Parate**  
B.Tech in Computer Science & Engineering  
MIT World Peace University, Pune  
Specializing in AI, ML, and Deep Learning  

---

## ⭐ Acknowledgement

Dataset courtesy of [Ahmed Hamada on Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
