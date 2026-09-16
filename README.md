# Skin Lesion Classification Using CNN

A deep learning-based image classification system for classifying dermoscopic skin lesion images into eight predefined diagnostic categories using Convolutional Neural Networks (CNNs) and transfer learning with EfficientNetB0.

> **Research / Educational Use Only**
>
> This project is developed for academic and research purposes. It is **not a medical diagnostic tool** and should not be used for clinical decision-making.

## Overview

Skin lesion classification from dermoscopic images is a challenging computer vision problem due to variations in lesion appearance, similarities between different lesion categories, and significant class imbalance.

This project explores deep learning approaches for automated classification of dermoscopic images. Two approaches are implemented and evaluated:

- A custom Convolutional Neural Network (CNN)
- EfficientNetB0 using transfer learning and fine-tuning

The final system uses the fine-tuned **EfficientNetB0** model for image classification and provides prediction probabilities, top predictions, and Grad-CAM-based visualization of model attribution.

A Streamlit web application is also included to provide an interactive interface for running the trained model.

## Features

- 8-class dermoscopic skin lesion classification
- Custom CNN baseline model
- EfficientNetB0 transfer learning
- EfficientNetB0 fine-tuning
- Lesion-level stratified dataset splitting
- Class-weighted model training to address class imbalance
- Accuracy, precision, recall, and F1-score evaluation
- Per-class performance analysis
- Confusion matrix evaluation
- Top-3 prediction probabilities
- Grad-CAM-based model attribution visualization
- Interactive Streamlit web application
- Pre-trained final EfficientNetB0 model included for inference

## Dataset

This project uses the **HAM10000 collection available through the ISIC Archive**.

The current ISIC collection contains **11,720 dermoscopic images**. For this project, an eight-class classification task was constructed from the available diagnostic metadata.

### Dataset Classes

| Class Code | Description |
|------------|-------------|
| `akiec` | Solar or actinic keratosis |
| `bcc` | Basal cell carcinoma |
| `bkl` | Pigmented benign keratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma, NOS |
| `nv` | Nevus |
| `scc` | Squamous cell carcinoma, NOS |
| `vasc` | Vascular lesion |

### Class Distribution

| Class | Number of Images |
|-------|-----------------:|
| `nv` | 7,737 |
| `bkl` | 1,338 |
| `mel` | 1,305 |
| `bcc` | 622 |
| `scc` | 229 |
| `vasc` | 180 |
| `df` | 160 |
| `akiec` | 149 |
| **Total** | **11,720** |

The dataset is highly imbalanced, with `nv` representing the majority class. Class weighting was therefore used during model training to reduce the effect of class imbalance.

### Dataset Source

The dataset can be obtained from the official ISIC Archive:

**HAM10000 Collection:**  
https://api.isic-archive.com/collections/212/

The dataset images are **not included in this repository**. Users should download the dataset directly from the official source and place it in the project's `data/HAM10000/` directory.

## Dataset Preparation and Splitting

The dataset preparation pipeline consists of two main stages: label preparation and dataset splitting.

### 1. Label Preparation

The script `src/prepare_dataset.py` processes the downloaded metadata and maps the available diagnostic information to the eight target classes used in this project.

It generates a `labels.csv` file containing the image identifiers and their corresponding class labels.

Run:

```bash
python src/prepare_dataset.py
```

### 2. Dataset Splitting

The script `src/split_dataset.py` divides the labeled dataset into training, validation, and testing sets.

The project uses **stratified group splitting based on `lesion_id`**. This ensures that images belonging to the same lesion are not distributed across different dataset splits, helping reduce data leakage.

The resulting dataset split is:

| Split      |     Images | Percentage |
| ---------- | ---------: | ---------: |
| Training   |      8,371 |     71.42% |
| Validation |      1,674 |     14.28% |
| Testing    |      1,675 |     14.29% |
| **Total**  | **11,720** |   **100%** |

The generated files are:

```text
data/HAM10000/train.csv
data/HAM10000/validation.csv
data/HAM10000/test.csv
```

Run:

```bash
python src/split_dataset.py
```

### Dataset Leakage Check

The splitting process groups samples using `lesion_id`, and the resulting training, validation, and test sets contain no overlapping lesion IDs.

## Project Structure

```text
Skin-Lesion-Classification/
│
├── app/
│   └── app.py
│
├── data/
│   └── HAM10000/
│       └── .gitkeep
│
├── models/
│   └── efficientnet_b0_finetuned_best.keras
│
├── src/
│   ├── __init__.py
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── transfer_data_pipeline.py
│   ├── efficientnet_model.py
│   ├── train_efficientnet.py
│   ├── finetune_efficientnet.py
│   ├── evaluate_efficientnet_finetuned.py
│   ├── inference.py
│   ├── gradcam_utils.py
│   └── gradcam.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

###  Directory Description

| Directory / File   | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `app/`             | Streamlit web application                                               |
| `data/`            | Dataset storage location                                                |
| `models/`          | Trained model files                                                     |
| `src/`             | Dataset preparation, training, evaluation, inference, and Grad-CAM code |
| `requirements.txt` | Python dependencies                                                     |
| `.gitignore`       | Files and directories excluded from version control                     |
| `README.md`        | Project documentation                                                   |

**Note:** The dataset images and generated results are excluded from the public repository. The final trained EfficientNetB0 model is included so that inference can be performed without retraining the model.


## Model Architectures

Two deep learning approaches are implemented in this project: a custom CNN and EfficientNetB0.

### Custom CNN

A custom Convolutional Neural Network is implemented as a baseline model.

The architecture consists of:

- Four convolutional blocks
- Batch normalization
- Max pooling
- Global average pooling
- Dropout regularization
- Fully connected layers
- An 8-class softmax output layer

The custom CNN provides a baseline for evaluating the benefits of transfer learning.

### EfficientNetB0

The final classification system uses **EfficientNetB0** with transfer learning and fine-tuning.

The training process consists of:

1. Loading a pre-trained EfficientNetB0 backbone.
2. Adding a classification head for the eight target classes.
3. Training the classification head.
4. Fine-tuning selected upper layers of the EfficientNetB0 backbone.
5. Using class-weighted training to address dataset imbalance.
6. Selecting the best model based on validation performance.

The final trained model is:

```text
models/efficientnet_b0_finetuned_best.keras
```

EfficientNetB0 was selected as the final model because the transfer learning approach provided improved classification performance compared with the custom CNN baseline.

## Model Training

The models are trained using the prepared training dataset, while the validation dataset is used to monitor model performance during training.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Image Size | `224 × 224` |
| Number of Classes | 8 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Class Weighting | Applied |
| Final Architecture | EfficientNetB0 |
| Fine-Tuning Learning Rate | `1e-5` |

Class weights are applied during training because the dataset contains a significant imbalance between the diagnostic categories.

### Transfer Learning

Initially, the pre-trained EfficientNetB0 backbone is used to extract visual features, while the newly added classification layers are trained for the eight target classes.

### Fine-Tuning

After the initial transfer learning stage, selected upper layers of EfficientNetB0 are unfrozen and fine-tuned using a lower learning rate.

Batch normalization layers are kept frozen during fine-tuning to improve training stability.

The best model checkpoint is selected based on validation performance and saved as:

```text
models/efficientnet_b0_finetuned_best.keras
```

## Model Evaluation and Results

The trained models are evaluated on the held-out test set using accuracy, precision, recall, and F1-score.

### Custom CNN Results

The stabilized custom CNN achieved:

| Metric | Score |
|--------|------:|
| Accuracy | 70.45% |
| Weighted Precision | 69.44% |
| Weighted Recall | 70.45% |
| Weighted F1-Score | 69.65% |
| Macro F1-Score | 41.59% |

### EfficientNetB0 Results

The fine-tuned EfficientNetB0 achieved:

| Metric | Score |
|--------|------:|
| Accuracy | **75.34%** |
| Weighted Precision | **77.66%** |
| Weighted Recall | **75.34%** |
| Weighted F1-Score | **76.25%** |
| Macro F1-Score | **52.70%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|----------:|-------:|---------:|
| `akiec` | 0.38 | 0.36 | 0.37 |
| `bcc` | 0.59 | 0.55 | 0.57 |
| `bkl` | 0.50 | 0.63 | 0.56 |
| `df` | 0.33 | 0.59 | 0.42 |
| `mel` | 0.51 | 0.59 | 0.55 |
| `nv` | 0.92 | 0.85 | 0.88 |
| `scc` | 0.21 | 0.21 | 0.21 |
| `vasc` | 0.65 | 0.65 | 0.65 |

The results show that the fine-tuned EfficientNetB0 provides stronger overall classification performance than the custom CNN baseline. Performance remains lower for several minority classes, reflecting the substantial class imbalance in the dataset.

## Grad-CAM Explainability

To provide an interpretable visualization of the model's predictions, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is implemented.

Grad-CAM highlights image regions that contribute more strongly to the selected model prediction by using gradients flowing into the final convolutional feature maps.

The implementation is provided in:

```text
src/gradcam_utils.py
src/gradcam.py
```

The `top_conv` layer of the EfficientNetB0 backbone is used as the target convolutional layer.

### Grad-CAM Workflow

1. **Load Model**: Load the trained `models/efficientnet_b0_finetuned_best.keras`.
2. **Preprocess Image**: Resize input dermoscopic image to 224×224 and scale pixel values.
3. **Inference**: Compute class probabilities across all 8 diagnostic categories.
4. **Target Selection**: Determine the predicted class index (or user-selected category).
5. **Gradient Computation**: Calculate gradients of the selected class score with respect to the `top_conv` convolutional feature maps.
6. **Importance Weighting**: Pool gradients spatially to determine weights for each feature channel.
7. **Heatmap Generation**: Compute the weighted combination of feature maps and apply ReLU to retain positive contributions.
8. **Normalization & Overlay**: Normalize the heatmap to [0, 1], apply the `jet` colormap, resize to the original image dimensions, and blend with the dermoscopic image (alpha = 0.40).

> **Important Caution:** Grad-CAM illustrates algorithmic feature attribution (the spatial regions influencing the model's output) and does **not** represent medical reasoning or diagnostic boundaries. Highlighted regions should never be used as standalone clinical evidence.

## Web Application

A web-based interface is built using **Streamlit** to make the trained model easy to test and explore interactively.

The application is implemented in:

```text
app/app.py
```

### Application Features

- **Dermoscopic Image Upload**: Supports `.jpg`, `.jpeg`, and `.png` images.
- **Automated Prediction**: Real-time classification using the pre-trained EfficientNetB0 model.
- **Diagnostic Finding Card**: Displays the top predicted class, formatted clinical name, and confidence score.
- **Grad-CAM Visualizations**: Side-by-side display of the original image, raw attribution heatmap, and heatmap overlay.
- **Probability Breakdown**: Top-3 ranked predictions and complete 8-class probability distribution.
- **Clinical Explanations & Disclaimers**: Guidance on interpreting model attributions and educational disclaimers.

### Running the Application

From the project root directory, start the Streamlit application:

```bash
streamlit run app/app.py
```

Alternatively, you can launch the app directly via Python:

```bash
python app/app.py
```

The application automatically loads the final trained model:

```text
models/efficientnet_b0_finetuned_best.keras
```

The model is cached in memory using `@st.cache_resource` for low-latency subsequent predictions.

> **Disclaimer:** The web application is intended for research and educational demonstrations only. Predictions must not be used for clinical diagnosis or patient triage.

## Installation and Setup

### Prerequisites

Ensure the following tools are installed:

- **Python 3.10 – 3.12** (Python 3.12 recommended)
- **Git**
- **pip**

A dedicated GPU is beneficial if retraining the models, but is **not required** to run inference or the Streamlit web application with the included pre-trained model.

### 1. Clone the Repository

```bash
git clone https://github.com/nabilkhan-01/Skin-Lesion-Classification.git
cd Skin-Lesion-Classification
```

### 2. Create a Virtual Environment

**Windows (Command Prompt / PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Web Application

The pre-trained model checkpoint (`models/efficientnet_b0_finetuned_best.keras`) is included with this repository. You can launch the application immediately without downloading the dataset or retraining:

```bash
streamlit run app/app.py
```

Then open your browser to `http://localhost:8501`.

## Training From Scratch

If you wish to reproduce the full training pipeline or retrain models from scratch, follow the dataset setup and training procedures below.

### 1. Dataset Setup

Download the HAM10000 collection from the official ISIC Archive:
- **ISIC HAM10000 Collection:** [https://api.isic-archive.com/collections/212/](https://api.isic-archive.com/collections/212/)

Place the downloaded dataset images and metadata inside:

```text
data/HAM10000/
```

### 2. Prepare the Dataset Labels

Map the ISIC metadata to the eight target diagnostic classes:

```bash
python src/prepare_dataset.py
```

This generates `data/HAM10000/labels.csv`.

### 3. Split the Dataset

Create lesion-grouped, stratified train/validation/test splits to prevent data leakage:

```bash
python src/split_dataset.py
```

This generates `train.csv`, `validation.csv`, and `test.csv` in `data/HAM10000/`.

### 4. EfficientNetB0 Training Pipeline

The training and evaluation pipeline follows a multi-stage process:

```text
Dataset Preparation (src/prepare_dataset.py)
        ↓
Lesion-Grouped Splitting (src/split_dataset.py)
        ↓
TF Data Pipeline with Class Weights (src/transfer_data_pipeline.py)
        ↓
Initial Transfer Learning Head Training (src/train_efficientnet.py)
        ↓
Fine-Tuning Upper Backbone Layers (src/finetune_efficientnet.py)
        ↓
Test Set Evaluation & Metrics (src/evaluate_efficientnet_finetuned.py)
```

The relevant pipeline scripts in `src/` are:

```text
src/prepare_dataset.py
src/split_dataset.py
src/transfer_data_pipeline.py
src/efficientnet_model.py
src/train_efficientnet.py
src/finetune_efficientnet.py
src/evaluate_efficientnet_finetuned.py
```

#### Step A: Initial Transfer Learning
Train the classification head while keeping the EfficientNetB0 base frozen:

```bash
python src/train_efficientnet.py
```

#### Step B: Fine-Tuning
Unfreeze the top layers of the backbone and fine-tune with a reduced learning rate (`1e-5`):

```bash
python src/finetune_efficientnet.py
```

The optimal checkpoint based on validation performance is saved as:

```text
models/efficientnet_b0_finetuned_best.keras
```

#### Step C: Evaluation
Evaluate the final fine-tuned model against the held-out test split:

```bash
python src/evaluate_efficientnet_finetuned.py
```

### Computational Requirements

- **Inference & Streamlit Web App:** Runs smoothly on standard CPU; GPU not required.
- **Training from Scratch:** A CUDA-compatible NVIDIA GPU is recommended for reasonable training times.


## Limitations

Although the proposed system demonstrates the use of deep learning for dermoscopic image classification, several limitations should be considered:

- The dataset is significantly class-imbalanced, with some diagnostic categories containing substantially fewer images than others.
- Model performance varies across individual classes.
- Minority classes generally have lower classification performance than the majority class.
- The model is trained on a specific dermoscopic image dataset and may not generalize to images from different sources, devices, or clinical environments.
- The system has not undergone clinical validation.
- Prediction confidence represents the model's estimated probability distribution and should not be interpreted as clinical certainty.
- Grad-CAM provides a visualization of model attribution and does not provide a clinical explanation of the prediction.
- The system should not be used for diagnosis, treatment decisions, or other clinical decision-making.

## Future Scope

The project can be further extended in several directions:

- Use larger and more diverse dermoscopic image datasets.
- Improve classification of minority classes through advanced data augmentation and class-balancing techniques.
- Explore additional CNN architectures and ensemble learning approaches.
- Investigate improved probability calibration to make model confidence estimates more reliable.
- Evaluate the model on external datasets to study its generalization capability.
- Explore additional explainability techniques alongside Grad-CAM.
- Optimize the model for faster and more resource-efficient inference.
- Conduct more extensive hyperparameter optimization.
- Perform further validation in collaboration with domain experts before considering any clinical application.

## Disclaimer

This project is developed for **research and educational purposes only**.

The model is designed to classify dermoscopic images into predefined categories based on patterns learned from the training dataset. It has **not been clinically validated** and should not be considered a substitute for professional medical evaluation.

**This system is not a medical diagnostic tool and must not be used for diagnosis, treatment decisions, or clinical decision-making.**