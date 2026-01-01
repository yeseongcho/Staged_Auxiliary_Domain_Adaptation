Smooth & Clear: Progressive Domain Adaptation for Foggy Scene Segmentation
This repository provides the official implementation of "Smooth_And_Clear: Auxiliary Domain and Inter-Class Contrast Adaptation for Semantic Foggy Scene Segmentation." Our framework bridges the gap between clean and foggy domains through a stable, staged adaptation pipeline.

🚀 Overview
Staged Adaptation: A multi-stage pipeline linking synthetic fog, clean target, and foggy target domains.

ICCA: Inter-Class Contrast Adaptation to resolve boundary blurring in foggy scenes.

Lightweight: High performance achieved on a RefineNet-lw backbone, ensuring deployability.

📂 Dataset Setup
Place your datasets in the ./data directory. Ensure the structure follows the requirement for training and testing.

Bash

./data
└── Foggy_Zurich
    └── Foggy_Zurich
        ├── lists_file_names
        │   └── gt_labelTrainIds_testv1_filenames.txt  # Filename list
        └── (images and labels)
Note: Filename lists (e.g., Foggy Zurich's label set) are already provided in the respective data paths.

📥 Pretrained Models
Download weights from our Google Drive and place them as follows:

ResNet-101 (ImageNet): ./model/resnet_pretrained/resnet101-5d3b4d8f.pth

RefineNet-lw (Cityscapes): ./Cityscapes_pretrained_model.pth

Weights for each training stage are also available in the link.

🛠 Training & Inference
Follow these steps in sequence to reproduce our results:

1. Data Categorization
Distinguish between clean and foggy targets using the provided notebook:

Bash

jupyter notebook clean_target_extraction.ipynb
(Pre-organized lists are available in ./data/Foggy_Zurich/Foggy_Zurich/lists_file_names)

2. Staged Adaptation
Run the stages sequentially to stabilize the domain transfer:

Bash

python stage1.py
python stage2.py
python stage3.py
3. Pseudo-Labeling & Final Stage
Generate pseudo-labels before the final refinement:

Bash

python pseudo_labeling.py  # Stores labels in ./pseudo_label_clean and ./pseudo_label_foggy
python stage4.py           # Final Adaptation Stage
📊 Results
Our model achieves competitive performance on Foggy Zurich, Foggy Driving, and ACDC datasets by effectively capturing structural information despite the lightweight backbone.
