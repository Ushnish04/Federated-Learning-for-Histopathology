# ==========================================================
# Predict Tumor Types for All Images in outputs Folder
# ==========================================================
import os
import sys
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import random

# ==========================================================
# Configuration
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="output",
                    help="Folder containing restained images to classify")
parser.add_argument("--train_ratio", type=float, default=0.8,
                    help="Fraction of images used for training (default: 0.8)")
args, _ = parser.parse_known_args()

INPUT_DIR            = args.input_dir
TRAIN_RATIO          = args.train_ratio
MODEL_PATH           = "resnet18_restained_model_best.pth"
OUTPUT_CSV           = "predictions_results.csv"
CONFIDENCE_THRESHOLD = 90.0
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Input directory: {INPUT_DIR}")
print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD}%")
print(f"[INFO] Train/Test split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
print(f"[INFO] Loading model from: {MODEL_PATH}\n")

# ==========================================================
# Load Model
# ==========================================================
checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["classes"]
num_classes = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"[INFO] Model loaded successfully!")
print(f"[INFO] Classes: {class_names}\n")

# ==========================================================
# Image Preprocessing
# ==========================================================
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================================
# Prediction Function
# ==========================================================
def predict_image(img_path):
    try:
        image      = Image.open(img_path).convert("RGB")
        tensor_img = infer_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs  = model(tensor_img)
            probs    = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        predicted_class = class_names[pred_idx]
        confidence      = probs[0][pred_idx].item() * 100
        top3_probs, top3_indices = torch.topk(probs[0], 3)
        top3_classes = [(class_names[idx.item()], prob.item() * 100)
                        for idx, prob in zip(top3_indices, top3_probs)]
        return {'success': True, 'predicted_class': predicted_class,
                'confidence': confidence, 'top3': top3_classes}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ==========================================================
# Validate input directory
# ==========================================================
if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] Directory '{INPUT_DIR}' not found!")
    sys.exit(1)

image_files = [f for f in os.listdir(INPUT_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) == 0:
    print(f"[ERROR] No images found in '{INPUT_DIR}'!")
    sys.exit(1)

print(f"[INFO] Found {len(image_files)} images to process\n")
print("="*80)

# ==========================================================
# Process All Images
# ==========================================================
results                = []
correct_predictions    = 0
total_predictions      = 0
high_confidence_images = []
low_confidence_images  = []

for img_file in tqdm(image_files, desc="Processing images"):
    img_path   = os.path.join(INPUT_DIR, img_file)
    true_class = next((cls for cls in class_names
                       if img_file.startswith(cls + "_")), None)
    result = predict_image(img_path)

    if result['success']:
        predicted_class = result['predicted_class']
        confidence      = result['confidence']
        top3            = result['top3']

        is_correct = (predicted_class == true_class) if true_class else None
        if is_correct is not None:
            total_predictions += 1
            if is_correct:
                correct_predictions += 1

        results.append({
            'filename':        img_file,
            'true_class':      true_class if true_class else 'Unknown',
            'predicted_class': predicted_class,
            'confidence':      confidence,
            'correct':         '✓' if is_correct else ('✗' if is_correct is not None else 'N/A'),
            'top1_class': top3[0][0], 'top1_conf': top3[0][1],
            'top2_class': top3[1][0], 'top2_conf': top3[1][1],
            'top3_class': top3[2][0], 'top3_conf': top3[2][1],
        })

        if confidence >= CONFIDENCE_THRESHOLD:
            high_confidence_images.append({
                'filename': img_file, 'path': img_path,
                'predicted_class': predicted_class, 'confidence': confidence
            })
        else:
            low_confidence_images.append({
                'filename': img_file,
                'predicted_class': predicted_class, 'confidence': confidence
            })

        status      = "✓ CORRECT" if is_correct else ("✗ WRONG" if is_correct is not None else "")
        conf_status = "✓ HIGH CONF" if confidence >= CONFIDENCE_THRESHOLD else "✗ LOW CONF"
        print(f"\n📁 {img_file}")
        if true_class:
            print(f"   True Class: {true_class}")
        print(f"   Predicted:  {predicted_class} ({confidence:.2f}%) {status} {conf_status}")
    else:
        print(f"\n❌ Error processing {img_file}: {result['error']}")
        results.append({
            'filename': img_file, 'true_class': true_class or 'Unknown',
            'predicted_class': 'ERROR', 'confidence': 0.0, 'correct': 'N/A',
            'top1_class': '', 'top1_conf': 0.0,
            'top2_class': '', 'top2_conf': 0.0,
            'top3_class': '', 'top3_conf': 0.0,
        })

# ==========================================================
# Save CSV
# ==========================================================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total images processed: {len(image_files)}")
print(f"High confidence (>={CONFIDENCE_THRESHOLD}%): {len(high_confidence_images)}")
print(f"Low confidence  (<{CONFIDENCE_THRESHOLD}%):  {len(low_confidence_images)}")

if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nCorrect predictions: {correct_predictions}/{total_predictions}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("\n📊 Predictions by True Class:")
    for cls in class_names:
        cls_results = [r for r in results if r['true_class'] == cls]
        if cls_results:
            cls_correct = len([r for r in cls_results if r['correct'] == '✓'])
            cls_total   = len(cls_results)
            print(f"   {cls:25s}: {cls_correct:2d}/{cls_total:2d} correct "
                  f"({cls_correct/cls_total*100:.1f}%)")

print(f"\n💾 Results saved to: {OUTPUT_CSV}")

# ==========================================================
# Build ImageFolder-compatible train/test structure:
#
#   <INPUT_DIR>/
#     train/
#       adenosis/
#       ductal_carcinoma/
#       ...
#     test/
#       adenosis/
#       ductal_carcinoma/
#       ...
#
# This is exactly what torchvision.datasets.ImageFolder expects.
# ==========================================================
print("\n" + "="*80)
print("REORGANIZING FILES INTO train/ AND test/ SPLITS")
print("="*80)

# Create train/<class> and test/<class> folders
for split in ("train", "test"):
    for cls in class_names:
        os.makedirs(os.path.join(INPUT_DIR, split, cls), exist_ok=True)

# Group high-confidence images by predicted class, then split
by_class    = defaultdict(list)
for img_info in high_confidence_images:
    by_class[img_info['predicted_class']].append(img_info)

train_count = 0
test_count  = 0

for cls, images in by_class.items():
    random.shuffle(images)
    split_idx  = max(1, int(len(images) * TRAIN_RATIO))
    train_imgs = images[:split_idx]
    test_imgs  = images[split_idx:]

    for img_info in train_imgs:
        dest = os.path.join(INPUT_DIR, "train", cls, img_info['filename'])
        try:
            shutil.move(img_info['path'], dest)
            print(f"   [train] {img_info['filename']:50s} → train/{cls}/")
            train_count += 1
        except Exception as e:
            print(f"   ✗ Error moving {img_info['filename']}: {e}")

    for img_info in test_imgs:
        dest = os.path.join(INPUT_DIR, "test", cls, img_info['filename'])
        try:
            shutil.move(img_info['path'], dest)
            print(f"   [test]  {img_info['filename']:50s} → test/{cls}/")
            test_count += 1
        except Exception as e:
            print(f"   ✗ Error moving {img_info['filename']}: {e}")

print(f"\n✅ {train_count} images → train/   |   {test_count} images → test/")

# Remove low-confidence images (still at root of INPUT_DIR)
print(f"\n❌ Removing {len(low_confidence_images)} low-confidence images...")
for img_info in low_confidence_images:
    img_path = os.path.join(INPUT_DIR, img_info['filename'])
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"   ✗ {img_info['filename']:50s} ({img_info['confidence']:.2f}%) - REMOVED")
    except Exception as e:
        print(f"   ✗ Error removing {img_info['filename']}: {e}")

# ==========================================================
# Excluded Images Report
# ==========================================================
print("\n" + "="*80)
print("EXCLUDED IMAGES (Confidence < 90%)")
print("="*80)
if not low_confidence_images:
    print("🎉 No images were excluded! All images met the confidence threshold.")
else:
    print(f"\nTotal excluded: {len(low_confidence_images)} images\n")
    excluded_by_class = defaultdict(list)
    for img_info in low_confidence_images:
        excluded_by_class[img_info['predicted_class']].append(img_info)
    for cls in sorted(excluded_by_class.keys()):
        images = excluded_by_class[cls]
        print(f"\n{cls} ({len(images)} images):")
        for img_info in images:
            print(f"   • {img_info['filename']:50s} - {img_info['confidence']:.2f}%")

# ==========================================================
# Final Statistics
# ==========================================================
print("\n" + "="*80)
print("FINAL DATASET STATISTICS")
print("="*80)
print(f"\n📁 Images organized in: {INPUT_DIR}/")
for split in ("train", "test"):
    split_total = 0
    for cls in class_names:
        folder = os.path.join(INPUT_DIR, split, cls)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if count > 0:
                print(f"   {split:5s}/{cls:25s}: {count:3d} images")
                split_total += count
    print(f"   {'':5s}{'TOTAL':25s}: {split_total:3d} images\n")

print("="*80)
print("\n✅ Prediction complete!")