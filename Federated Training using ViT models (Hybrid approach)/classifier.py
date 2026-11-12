# Classifier.py
import os
import gc
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import timm
#  SETTINGS 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 32
PCA_DIM = 1280

FL_PTH_DIR = "C:/Users/MSC055/Desktop/Sabrina/CSE711/Using_Federated_Learning_Framework/for_ViT/output_pth_file"

#  TRANSFORMS 

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
#  DATASET 
DATA_PATH = "C:/Users/MSC055/Desktop/Sabrina/CSE711/Eye Image Dataset"
CLASSES = sorted(os.listdir(DATA_PATH))

file_paths, labels = [], []
for idx, cls in enumerate(CLASSES):
    for f in os.listdir(os.path.join(DATA_PATH, cls)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            file_paths.append(os.path.join(DATA_PATH, cls, f))
            labels.append(idx)

file_paths = np.array(file_paths)
labels = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    file_paths, labels, test_size=0.3, stratify=labels, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

print(f"Train_size: {len(X_train)} | Val_size: {len(X_val)} | Test_size: {len(X_test)}")

#  PROTONET 
class ProtoNetwork(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,      # removing classifier head
            global_pool="avg"
        )

    def forward(self, x):
        return self.backbone(x)

#  FEATURE EXTRACTION 
def extract_embeddings(model, file_paths, transform, batch_size=16):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(file_paths), batch_size), desc="Extracting Features"):
            batch_paths = file_paths[i:i+batch_size]
            imgs = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                imgs.append(transform(img))
            imgs = torch.stack(imgs).to(DEVICE)
            feats = model(imgs).cpu().numpy().astype(np.float32)
            embeddings.append(feats)
            del imgs, feats
            gc.collect()
    return np.vstack(embeddings)

#  LOAD FL BACKBONES 
backbones = ["convit_small", "mobilevit_s", "efficientvit_b0"]
proto_models = {}

for name in backbones:
    proto = ProtoNetwork(name).to(DEVICE)

    pth_path = os.path.join(FL_PTH_DIR, f"global_{name}.pth")
    proto.load_state_dict(
        torch.load(pth_path, map_location=DEVICE)
    )
    proto.eval()
    proto_models[name] = proto
    print(f" Successfully loaded FL weights for {name} from {pth_path}")

#  SYMBOLIC MODELS 
symbolic_models = {
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),
        {
            "n_estimators": randint(50, 100),
            "max_depth": randint(5, 50),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }
    ),
    "XGBoost": (
        xgb.XGBClassifier(eval_metric="mlogloss", random_state=SEED),
        {
            "n_estimators": randint(50, 100),
            "max_depth": randint(3, 15),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "gamma": uniform(0, 5),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0, 1)
        }
    ),
    
    "LightGBM": (
        lgb.LGBMClassifier(verbose=-1, random_state=SEED),
        {
            "n_estimators": randint(50, 100),
            "max_depth": randint(3, 15),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "gamma": uniform(0, 5),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0, 1)
        }
    ),
    "CatBoost": (
        cb.CatBoostClassifier(verbose=0,random_state=SEED),
        {
            "iterations": randint(50, 100),
            "depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.3),
            "l2_leaf_reg": uniform(1, 10),
            "border_count": randint(32, 255),
            "random_strength": uniform(0, 10)
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=SEED),
        {
           "n_estimators": randint(50, 100),
            "learning_rate": uniform(0.01, 1.0)
        }
    )
}

#  FEATURE FUSION + PCA 
def fuse_features_train(models_dict, file_paths, transform, pca_dim=PCA_DIM):
    all_feats = []
    pca_models = {}

    for name, model in models_dict.items():
        print(f"\nExtracting TRAIN features for {name}...")
        feats = extract_embeddings(model, file_paths, transform)

        n_samples, feat_dim = feats.shape
        effective_pca_dim = min(pca_dim, feat_dim, n_samples)

        print(
            f"Applying PCA for {name} | "
            f"samples={n_samples}, feat_dim={feat_dim}, pca_dim={effective_pca_dim}"
        )

        pca = PCA(
            n_components=effective_pca_dim,
            whiten=True,
            random_state=SEED
        )

        feats_pca = pca.fit_transform(feats)

        all_feats.append(feats_pca)
        pca_models[name] = pca

        del feats, feats_pca
        gc.collect()

    fused = np.concatenate(all_feats, axis=1)
    print(f"\nFused TRAIN feature shape: {fused.shape}")

    return fused, pca_models

def fuse_features_eval(models_dict, file_paths, transform, pca_models):
    all_feats = []

    for name, model in models_dict.items():
        print(f"\nExtracting VAL features for {name}...")
        feats = extract_embeddings(model, file_paths, transform)

        feats_pca = pca_models[name].transform(feats)
        all_feats.append(feats_pca)

        del feats, feats_pca
        gc.collect()

    fused = np.concatenate(all_feats, axis=1)
    print(f"\nFused VAL feature shape: {fused.shape}")

    return fused

# TRAIN FEATURES
X_train_feat, pca_models = fuse_features_train(
    proto_models, X_train, val_transform
)

# VALIDATION FEATURES
X_val_feat = fuse_features_eval(
    proto_models, X_val, val_transform, pca_models
)

#  TRAIN + EVAL 
for sym_name, (clf, param_dist) in symbolic_models.items():
    print(f"\nTraining Symbolic Model: {sym_name}")
    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=3,
                                scoring="accuracy", random_state=SEED, n_jobs=-1)
    search.fit(X_train_feat, y_train)
    tuned_model = search.best_estimator_
    best_params = search.best_params_

    y_train_pred = tuned_model.predict(X_train_feat)
    y_val_pred = tuned_model.predict(X_val_feat)
    y_train_proba = tuned_model.predict_proba(X_train_feat)
    y_val_proba = tuned_model.predict_proba(X_val_feat)

    train_loss_calc = log_loss(y_train, y_train_proba)
    val_loss_calc = log_loss(y_val, y_val_proba)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    overall_acc = val_acc
    precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    class_report = classification_report(y_val, y_val_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    print(f"\nFeature Fusion + {sym_name} Results:")
    print(f"Train Loss: {train_loss_calc:.4f} | Val Loss: {val_loss_calc:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Accuracy: {overall_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Best Params: {best_params}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")