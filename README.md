# Research-Codebase
# A Unified Hybrid Framework for Ocular Disease Classification

### Integrating Few-Shot Learning with PCA-refined Multi-backbone Neural Embedding Fusion and Symbolic Classifiers

[![Research](https://img.shields.io/badge/Domain-Medical%20Image%20Analysis-blue)]()
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20ViT-purple)]()
[![Few-Shot Learning](https://img.shields.io/badge/Learning-Few--Shot-orange)]()
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Flower-green)]()

## Overview

This repository presents a unified hybrid machine learning framework for multi-class ocular disease classification from retinal fundus images.

The proposed approach addresses two major challenges in medical image classification:

- **Severe class imbalance:** Rare ocular diseases are significantly underrepresented compared to common disease categories.
- **Limited discriminative feature representations:** Individual deep learning architectures may struggle to capture complementary visual information across different disease categories.

To address these challenges, the framework integrates deep feature extraction using multiple Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), few-shot learning, Principal Component Analysis (PCA), multi-backbone feature fusion, symbolic machine learning classifiers, and a Flower-based federated learning workflow.

Rather than relying exclusively on conventional end-to-end classification, the proposed framework separates feature learning from downstream classification, enabling systematic evaluation of different feature representations and machine learning classifiers.

## Research Contributions

The major contributions of this work include:

- A hybrid ocular disease classification framework combining CNN and Vision Transformer architectures for deep feature extraction.
- Integration of Siamese and Prototypical Networks to investigate few-shot learning under severe class imbalance.
- PCA-refined multi-backbone feature fusion to combine complementary feature representations while reducing dimensionality and redundancy.
- Integration of five symbolic machine learning classifiers for downstream classification.
- Hyperparameter optimization using RandomizedSearchCV with three-fold cross-validation.
- A Flower-based federated learning proof of concept integrating FedAvg with Prototypical Network training.
- A progressive experimental evaluation examining the contribution of individual components and their combinations.

## Dataset

The experiments were conducted using a publicly available retinal disease dataset containing **5,335 retinal fundus images across 10 ocular disease categories**.

The dataset was collected from:

- Anwara Hamida Eye Hospital, Faridpur, Bangladesh.
- BNS Zahrul Haque Eye Hospital, Faridpur district, Bangladesh.

### Dataset Distribution

| Disease Category | Images |
|---|---:|
| Central Serous Chorioretinopathy | 101 |
| Diabetic Retinopathy | 1,509 |
| Disc Edema | 127 |
| Glaucoma | 1,349 |
| Healthy | 1,024 |
| Macular Scar | 444 |
| Myopia | 500 |
| Pterygium | 17 |
| Retinal Detachment | 125 |
| Retinitis Pigmentosa | 139 |
| **Total** | **5,335** |

The dataset exhibits substantial class imbalance, making it a challenging benchmark for evaluating multi-class ocular disease classification.

**Dataset source:** [Mendeley Data](https://doi.org/10.17632/s9bfhswzjb.1)

### Dataset Splitting

The dataset was divided using stratified sampling to preserve class distributions across the experimental subsets.

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 3,734 | 70% |
| Validation | 800 | 15% |
| Testing | 801 | 15% |
| **Total** | **5,335** | **100%** |

## Proposed Framework

The proposed framework consists of several interconnected stages.

### 1. Image Preprocessing and Augmentation

All retinal images were resized to 224 × 224 pixels and normalized.

Training-time augmentation included:

- Random resized cropping.
- Random horizontal flipping.
- Random rotation up to 10°.

Corrupted images were inspected and removed before training.

### 2. Multi-Backbone Deep Feature Extraction

Eight deep learning architectures were investigated as feature extractors.

**Convolutional Neural Networks (CNNs)**

| Architecture | Purpose |
|---|---|
| ResNet50 | Residual feature representation |
| DenseNet121 | Dense feature propagation |
| VGG16 | Hierarchical feature extraction |
| MobileNetV2 | Lightweight feature extraction |
| EfficientNet-B0 | Computationally efficient feature extraction |

**Vision Transformers (ViTs)**

| Architecture | Purpose |
|---|---|
| ConViT-Small | Local and global feature representation |
| MobileViT-S | Lightweight hybrid feature extraction |
| EfficientViT-B0 | Efficient transformer-based feature extraction |

The classification heads were removed, and the architectures were employed exclusively as feature extractors.

### 3. Few-Shot Learning

To investigate learning under limited and imbalanced samples, two metric-learning approaches were evaluated.

**Siamese Network**

Learns similarity relationships between image pairs by minimizing the distance between embeddings of similar samples and increasing the distance between dissimilar samples.

**Prototypical Network**

Learns class-level representations by constructing prototype vectors from support samples within episodic training tasks.

The episodic configuration used in the federated experiments was:

| Parameter | Configuration |
|---|---:|
| Number of classes (N-way) | 10 |
| Support samples (K-shot) | 3 |
| Query samples per class | 4 |
| Episodes per round | 1,000 |
| Federated communication rounds | 10 |

Prototypical Network demonstrated superior performance compared with Siamese Network in the evaluated configurations.

### 4. PCA-Refined Feature Fusion

To exploit complementary information from multiple architectures, feature fusion was investigated for both CNN-based and ViT-based representations.

Principal Component Analysis (PCA) was applied to reduce feature dimensionality and eliminate redundant information before combining the extracted representations.

This process aimed to generate more compact and discriminative feature representations for downstream classification.

### 5. Symbolic Machine Learning Classification

Five symbolic machine learning classifiers were investigated:

| Classifier | Description |
|---|---|
| Random Forest | Ensemble of decision trees |
| AdaBoost | Adaptive boosting |
| CatBoost | Gradient boosting |
| XGBoost | Extreme gradient boosting |
| LightGBM | Histogram-based gradient boosting |

Each classifier was independently optimized using RandomizedSearchCV.

Hyperparameter optimization employed:

- Three-fold cross-validation.
- Predefined hyperparameter search spaces.
- Training features and labels for cross-validation.
- Validation data for model comparison.

### 6. Flower-Based Federated Learning

A federated learning workflow was implemented using the Flower framework.

The implementation follows a server-client architecture with the FedAvg aggregation strategy.

The experimental configuration included:

- One simulated client (proof of concept).
- Ten federated communication rounds.
- Prototypical Network-based local training.
- Adam optimizer.
- Learning rate of 0.0001.
- 1,000 episodes per round.
- Episodic configuration of N=10, K=3, Q=4.

At each communication round, the server distributed the current model parameters to the client. The client performed local Prototypical Network training and returned updated parameters to the server.

The resulting feature representations were subsequently used in the PCA-refined feature fusion and symbolic classification pipeline.

**Note:** The Flower implementation serves as an initial proof of concept. It does not establish multi-client federated generalization, non-IID robustness, or deployment-level privacy guarantees.

## Experimental Evaluation

The proposed framework was evaluated progressively to investigate the contribution of different learning strategies.

| Experimental Strategy | Best Validation Accuracy |
|---|---:|
| CNN + Symbolic ML | 64.5% |
| ViT + Symbolic ML | 62.62% |
| Siamese Network + CNN + Symbolic ML | 70.75% |
| Siamese Network + ViT + Symbolic ML | 77.5% |
| Prototypical Network + CNN + Symbolic ML | 77.38% |
| Prototypical Network + ViT + Symbolic ML | 78% |
| CNN Feature Fusion + Prototypical Network + PCA | 78.5% |
| ViT Feature Fusion + Prototypical Network + PCA | 77.62% |
| Federated Learning + CNN Feature Fusion + Prototypical Network + PCA | 89.12% |
| **Federated Learning + ViT Feature Fusion + Prototypical Network + PCA** | **89.75% ~ 90%** |

The best-performing configuration was obtained using ViT-based feature fusion, Prototypical Network training within the Flower-based workflow, PCA, and CatBoost classification.

### Best-Performing Configuration

<pre>
Retinal Fundus Images
         |
         v
Image Preprocessing
         |
         v
ViT-Based Feature Extraction
         |
         v
Prototypical Network Training
         |
         v
Flower-Based Federated Workflow
         |
         v
Multi-Backbone Feature Fusion
         |
         v
Principal Component Analysis
         |
         v
CatBoost Classification
         |
         v
Ocular Disease Prediction
</pre>

### Performance Metrics

The best-performing configuration achieved the following validation results:

| Evaluation Metric | Performance |
|---|---:|
| Accuracy | **89.75 ~ 90%** |
| Precision | **92.11%** |
| Recall | **91.17%** |
| F1-score | **91.52%** |

These results demonstrate the potential of combining few-shot representation learning, feature fusion, dimensionality reduction, federated learning workflows, and symbolic machine learning for imbalanced ocular disease classification.

## Evaluation Metrics

The framework was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-class classification performance

Additional feature-space analysis was performed using t-SNE to investigate class clustering and representation separability.

## Limitations and Future Work

Although the proposed framework demonstrated promising validation performance, several limitations remain:

1. The Flower-based federated learning implementation used only one simulated client.
2. Multi-client aggregation and heterogeneous (non-IID) data distributions require further investigation.
3. Privacy and security against potential attacks were not independently evaluated.
4. The experiments were conducted using a single retinal image dataset.
5. External validation using independent datasets is necessary to assess broader generalizability.

Future research will focus on multi-client federated experiments, heterogeneous data distributions, external dataset validation, and broader assessment of the framework under real-world clinical conditions.

## Repository Structure

```text
Research-Codebase/
├── README.md
├── LICENSE
│
├── Initial Implementation using imbalanced dataset/                 
├── Baseline Implementation/                           
├── Siamese Network Training/                            
├── Prototypical Network Training/                       
├── Prototypical Network Training with Feature Fusion/        
├── Federated Training using CNN models (Hybrid approach)/                      
├── Federated Training using ViT models (Hybrid approach)/                      
│
└── Figures/                               
```

## Reproducibility

To reproduce the reported experiments:

1. Obtain the retinal disease dataset from the provided Mendeley Data source.
2. Perform image preprocessing and stratified dataset splitting.
3. Extract deep features using the selected CNN and ViT architectures.
4. Evaluate Siamese and Prototypical Network-based few-shot learning.
5. Apply PCA and multi-backbone feature fusion.
6. Execute the Flower-based federated training workflow.
7. Optimize symbolic machine learning classifiers using RandomizedSearchCV.
8. Evaluate the resulting configurations using the validation dataset.

## Data Availability

The retinal disease dataset used in this research is publicly available through Mendeley Data:

https://doi.org/10.17632/s9bfhswzjb.1

The dataset is not included in this repository.

## Authors

**Sabrina Jahan**  
Department of Computer Science and Engineering  
BRAC University, Bangladesh

**Dr. Md. Golam Rabiul Alam**  
Department of Computer Science and Engineering  
BRAC University, Bangladesh

**B. M. Shahria Alam**  
Department of Computer Science and Engineering  
East West University, Bangladesh

## Acknowledgments

The authors acknowledge the publicly available retinal disease dataset provided through Mendeley Data and the Flower federated learning framework used in this research.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Developed as part of research on hybrid deep learning, few-shot learning, and federated machine learning for medical image classification.**
