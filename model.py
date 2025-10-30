# ==============================
# train.py — Train Food Recognition Model (HOG + YOLOv8 option)
# ==============================

# ---------------------------
# ⚠️ Suppress Warnings
# ---------------------------
import warnings
warnings.filterwarnings("ignore", message=".*libpng.*")  # libpng warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # sklearn warnings

# ---------------------------
# Imports
# ---------------------------
import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
from ultralytics import YOLO

# ====================================
# ⚙️ 1. HOG + Random Forest Training
# ====================================

train_folder = r"C:\Users\HP\.streamlit\internship\training"

X, y = [], []

print("🔍 Loading training images...")

# Iterate through folders (labels)
for label in os.listdir(train_folder):
    label_path = os.path.join(train_folder, label)
    if not os.path.isdir(label_path):
        continue

    files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for file in tqdm(files, desc=f"Loading {label}", unit="img"):
        img_path = os.path.join(label_path, file)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

        X.append(features)
        y.append(label)

X, y = np.array(X), np.array(y)
print(f"\n✅ Loaded {len(X)} samples across {len(set(y))} classes")

# -----------------------
# Train Model
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

print("🎯 HOG + RandomForest training complete!")
print(classification_report(y_test, model.predict(X_test), zero_division=0))

# -----------------------
# Save Model
# -----------------------
with open("train_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 HOG model saved as train_model.pkl")

# -----------------------
# Compute class centroids
# -----------------------
centroids = {label: np.mean(X[y == label], axis=0) for label in np.unique(y)}

with open("train_features.pkl", "wb") as f:
    pickle.dump(centroids, f)
print("💾 Class centroids saved as train_features.pkl")

# ====================================
# ⚙️ 2. YOLOv8 Training (Optional)
# ====================================
def train_yolo():
    print("\n🚀 Starting YOLOv8 training...")
    try:
        model = YOLO("yolov8s.pt")  # Pretrained base model
        model.train(
            data="data.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            name="food_yolo_model"
        )
        print("✅ YOLOv8 training completed successfully!")
    except Exception as e:
        print("❌ YOLO training error:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_yolo", action="store_true", help="Train YOLOv8 model instead of HOG")
    args = parser.parse_args()

    if args.use_yolo:
        train_yolo()
    else:
        print("\n✅ Finished HOG + RandomForest training process.")
