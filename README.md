# Very-mild-AD-detection

This repository contains the code and resources for a comparative study on early-stage Alzheimer's disease (AD) detection using deep learning models. The project evaluates the performance, efficiency, and robustness of state-of-the-art convolutional architectures on the OASIS-1 MRI dataset.

Early diagnosis of Alzheimer’s disease is critical for timely intervention and improved patient outcomes. This research explores and compares the following architectures:

- **HRNet** 
- **ConvNeXt** 
- **MobileNet-V2**

  Each model is assessed in terms of:
- **Accuracy, Precision, Recall, F1-score**
- **Robustness to noise and blur**
- **Training efficiency and memory usage**

  # dataset
  **[OASIS-1](https://www.oasis-brains.org/)**: A publicly available structural MRI dataset with labeled Clinical Dementia Ratings (CDR).
- Only 2D **coronal slices** were used for training and evaluation.
- Oversampling was applied to address class imbalance in CDR ratings.

  The project uses the following main libraries:
-	Pandas 2.2.2, os, numpy 1.26.4, and csv: for data handling, file operations, and exporting results.
-	Torch 2.6.0, torch.nn, torch.optim, torch.utils.data, and Timm 1.0.15: for deep learning model implementation, including loading pretrained models and defining training procedures.
-	Torchvision.transforms 0.21.0 and PIL.Image 10.4.0: for image preprocessing and data augmentation
-	NiBabel 5.3.2: to load and process medical neuroimaging files in .hdr and .img formats.
-	Scikit-learn (sklearn) 1.5.1: for data splitting and calculating evaluation metrics
-	CV2 (OpenCV) 4.10.0: for additional image processing
-	Matplotlib.pyplot 3.9.2: for visualizing results
-	Time: to measure training time

