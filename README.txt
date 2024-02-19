We made this README file to provide a temporary guide for re-running the code before uploading it to GitHub. 

First, you should locate the datasets used in this study in the './data' path. We created folder names for each dataset and when you want to run model, you should place the data in that path to use when training the model. (All files containing filenames except for image data are located in that path, i.e., the testfile names file containing Foggy_Zurich's label set, 'gt_labelTrainIds_testv1_filenames.txt')

We uploaded the pretrained weights for the model to the Google Drive link provided. The Google Drive contains the pretrained model and the final weights of our model.

Google Drive Link
https://drive.google.com/drive/folders/11R3eBXvt1AN_M40mQ_Js5v4OnT7LA9fO?usp=sharing

(1) ResNet-101 pretrained on ImageNet
After download, it should be positioned on './model/resnet_pretrained/resnet101-5d3b4d8f.pth'

(2) RefineNet-lw pretraiend on Clean Cityscapes
After download, it should be positioned on './Cityscapes_pretrained_model.pth'

Also, you could download the model weights for each stage of the model. 

For running,
(1) First, you could run "clean_target_extraction.ipynb" to distinguish between clean and fog targets. The results of the distinction are already organized in './data/Foggy_Zurich/Foggy_Zurich/lists_file_names'. 

(2) Next, proceed with "stage1.py", "stage2.py", and "stage3.py" in sequence. After that, we build a pseudo-label via "pseudo_labeling.py" before proceeding to "stage4.py", and finally run "stage4.py". Pseudo-labels are stored in the paths './pseudo_label_clean, ./pseudo_label_foggy' respectively. 

