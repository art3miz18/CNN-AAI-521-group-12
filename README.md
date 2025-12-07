# Brain Tumor Classification Using Deep Learning

**Course**: Computer Vision AAI-521
**Author**: Balaji Rao
**Institution**: University of San Diego

## Project Overview

This project implements a deep learning solution for classifying brain tumors from MRI images using Transfer Learning with VGG16. The model achieves high accuracy in distinguishing between four tumor types: Glioma, Meningioma, No Tumor, and Pituitary tumors.

## Dataset

- **Source**: [Brain Tumor Classification (MRI) - Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Total Images**: 3,264 MRI scans
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Image Type**: T1-weighted contrast-enhanced MRI images

### Dataset Distribution

**Training Set** (2,870 images):
- Glioma Tumor: 826 images
- Meningioma Tumor: 822 images
- No Tumor: 395 images
- Pituitary Tumor: 827 images

**Testing Set** (394 images):
- Glioma Tumor: 100 images
- Meningioma Tumor: 115 images
- No Tumor: 105 images
- Pituitary Tumor: 74 images

## Project Structure

```
balaji-rao-final/
├── README.md                           # Project documentation
├── brain_tumor_classification.ipynb    # Main implementation notebook
├── requirements.txt                    # Python dependencies
├── models/                             # Saved models (created during training)
├── results/                            # Training results and visualizations
└── utils/                              # Helper functions (optional)
```

## Implementation Approach

### 1. Data Preprocessing
- Image resizing to 224×224 pixels (VGG16 input size)
- Normalization (pixel values scaled to [0, 1])
- Data augmentation:
  - Rotation (±15 degrees)
  - Width/Height shifts (5%)
  - Brightness variation (0.1-1.5)
  - Horizontal and vertical flips

### 2. Model Architecture
- **Base Model**: VGG16 pre-trained on ImageNet
- **Transfer Learning Strategy**:
  - Remove original classification head
  - Add custom Dense(4) layer with softmax activation
  - Fine-tune all layers
- **Total Parameters**: ~14.8M

### 3. Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 20-50 (with early stopping)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- ROC Curves and AUC scores

## Expected Results

Based on baseline implementation:
- **Validation Accuracy**: ~94%
- **F1-Score**: ~91%
- **Performance**: Strong classification across all tumor types

## Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
```

## Usage

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   - Dataset is located at: `../archive/`
   - Training data: `../archive/Training/`
   - Testing data: `../archive/Testing/`

3. **Run Notebook**:
   - Open `brain_tumor_classification.ipynb`
   - Execute cells sequentially
   - Models will be saved to `models/` directory
   - Results will be saved to `results/` directory

## Key Features

- **Clean Implementation**: No hard-coded paths or brittle pickle files
- **Modular Code**: Well-organized and commented
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Reproducible**: Fixed random seeds for consistency
- **Best Practices**: Proper train/validation split, data augmentation, early stopping

## References

### Baseline Repository
- [Brain Tumor Classification Using Deep Learning Algorithms](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms)

### Citations
```
@article{kadam2021brain,
  title={Brain tumor classification using deep learning algorithms},
  author={Kadam, Ankita and Bhuvaji, Sartaj and Deshpande, Sujit},
  journal={Int. J. Res. Appl. Sci. Eng. Technol},
  volume={9},
  pages={417--426},
  year={2021}
}

@article{bhuvaji2020brain,
  title={Brain tumor classification (MRI)},
  author={Bhuvaji, Sartaj and Kadam, Ankita and Bhumkar, Prajakta and Dedge, Sameer and Kanchan, Swati},
  journal={Kaggle},
  volume={10},
  year={2020}
}
```

## License

This project is for educational purposes as part of the AAI-521 Computer Vision course at the University of San Diego.
