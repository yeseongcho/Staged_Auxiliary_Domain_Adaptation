# Smooth_And_Clear
**Auxiliary Domain and Inter-Class Contrast Adaptation for Semantic Foggy Scene Segmentation**

This repository provides the official implementation of **Smooth_And_Clear**, a staged unsupervised domain adaptation framework for robust semantic segmentation under foggy and adverse weather conditions. The method is designed for **lightweight segmentation models**, enabling stable deployment in real-world, resource-constrained environments.

---

## 📌 Overview

Semantic segmentation under fog suffers from:
- Large **domain gaps** (clean ↔ foggy)
- Severe **boundary ambiguity** caused by fog

To address these challenges, this project introduces:
- **Auxiliary-domain-based staged adaptation** for smooth domain transition
- **Inter-Class Contrast Adaptation (ICCA)** to mitigate boundary blurring
- A **lightweight training protocol** without architectural overhead

The framework achieves strong performance on **Foggy Zurich**, **Foggy Driving**, and **ACDC**, while maintaining robustness on clean data.

---

## 📂 Repository Structure

```
./data/                     # Dataset root directory
./model/                    # Network definitions
./pseudo_label_clean/       # Generated pseudo-labels (clean target)
./pseudo_label_foggy/       # Generated pseudo-labels (foggy target)
./stage1.py                 # Stage 1: Clean-style adaptation
./stage2.py                 # Stage 2: Fog-style adaptation
./stage3.py                 # Stage 3: Intra-target adaptation (EM + ICCA)
./stage4.py                 # Stage 4: Self-training with pseudo-labels
./pseudo_labeling.py        # Pseudo-label generation
./clean_target_extraction.ipynb
```

---

## 📊 Dataset Preparation

All datasets must be placed under the following path:

```
./data/
```

Each dataset should have its own subfolder. **All metadata files (e.g., file lists, label indices)** must also be placed in the corresponding directory.

> ⚠️ Image files are **not included** in this repository.

### Example

```
./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/
  └── gt_labelTrainIds_testv1_filenames.txt
```

---

## 🧠 Pretrained Models

We provide pretrained weights via Google Drive:

🔗 **Google Drive (Pretrained Models)**  
https://drive.google.com/drive/folders/11R3eBXvt1AN_M40mQ_Js5v4OnT7LA9fO?usp=sharing

### Required Files

1. **ResNet-101 (ImageNet pretrained)**
```
./model/resnet_pretrained/resnet101-5d3b4d8f.pth
```

2. **RefineNet-lw (Clean Cityscapes pretrained)**
```
./Cityscapes_pretrained_model.pth
```

> ✅ Optional: Stage-wise model checkpoints are also provided for analysis and reproduction.

---

## 🚀 Training Pipeline

The full training procedure follows **four progressive stages**.

### Step 0: Clean Target Extraction

Run the notebook to distinguish clean and foggy samples within the target domain:

```bash
clean_target_extraction.ipynb
```

> The extracted file lists are already organized under:
```
./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/
```

---

### Stage 1 – Clean Style Adaptation

```bash
python stage1.py
```
- Clean Source → Clean Target
- Adversarial Training (AT)

---

### Stage 2 – Fog Style Adaptation

```bash
python stage2.py
```
- Foggy Source → Foggy Target
- Style alignment under fog

---

### Stage 3 – Intra-Target Fog Adaptation

```bash
python stage3.py
```
- Clean Target → Foggy Target
- **Entropy Minimization (EM)** + **ICCA**

---

### Stage 4 – Self-Training

1. Generate pseudo-labels:
```bash
python pseudo_labeling.py
```

2. Final adaptation:
```bash
python stage4.py
```

📁 Pseudo-labels are saved to:
```
./pseudo_label_clean/
./pseudo_label_foggy/
```

---

## 📈 Notes

- Batch size is set to **2** due to GPU memory constraints
- ICCA is applied **only after sufficient warm-up** (Stage 3+)
- The framework avoids catastrophic forgetting on clean data

---

## 📜 License & Citation

If you use this code, please cite our paper:

> **Smooth and Clear: Auxiliary Domain and Inter-Class Contrast Adaptation for Semantic Foggy Scene Segmentation**  
> *Knowledge-Based Systems, 2024*

---

## 🙏 Acknowledgement

This codebase builds upon:
- RefineNet / RefineNet-lw
- Cityscapes & Foggy Cityscapes
- Prior work on UDA and contrastive adaptation

---

For questions or issues, feel free to open an issue or contact the authors.
