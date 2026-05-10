# histo_filter.py
import os
import glob
import shutil
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# ==========================================================
# 1. FEATURE EXTRACTOR
# ==========================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # drop last fc layer
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ==========================================================
# 2. LOAD MODELS
# ==========================================================
def load_models():
    extractor = FeatureExtractor().eval()

    # if you saved an extractor state_dict into models/feature_extractor.pth, load it
    extractor_path = Path("models/feature_extractor.pth")
    if extractor_path.exists():
        extractor.load_state_dict(torch.load(str(extractor_path)))

    # load svm
    svm_path = Path("histo_classifier_svm.pkl")
    if not svm_path.exists():
        raise FileNotFoundError(f"Cannot find classifier at {svm_path.resolve()}")
    with open(str(svm_path), "rb") as f:
        clf = pickle.load(f)

    return extractor, clf


extractor_loaded, clf_loaded = load_models()


# ==========================================================
# 3. PREDICTOR
# ==========================================================
def is_histopathology(image_path: str) -> int:
    """
    Returns 1 if histopathology, 0 otherwise.
    """
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = extractor_loaded(img).cpu().numpy().reshape(1, -1)

    pred = int(clf_loaded.predict(feat)[0])
    return pred


# ==========================================================
# 4. SAFELY MOVE (unique name if dest exists)
# ==========================================================
def _get_unique_dest(dest_path: Path) -> Path:
    """
    If dest_path exists, return a new Path with suffix _1, _2 ... until unique.
    """
    if not dest_path.exists():
        return dest_path

    base = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent

    counter = 1
    while True:
        candidate = parent / f"{base}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ==========================================================
# 5. FILTER IMAGES
# ==========================================================
def filter_histopathology_images(input_folder: str, output_hist_folder: str = "filtered_histo"):
    """
    Moves histopathology images from input_folder -> output_hist_folder.
    Returns list of absolute paths to moved histo images.
    Non-histo images are deleted from input_folder.
    """
    input_folder = Path(input_folder)
    output_hist_folder = Path(output_hist_folder)
    output_hist_folder.mkdir(parents=True, exist_ok=True)

    valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp", "*.tiff"]

    all_imgs = []
    for ext in valid_exts:
        all_imgs.extend(sorted(glob.glob(str(input_folder / ext))))

    kept = []
    removed = []
    failed = []

    print(f"\n🔍 Checking {len(all_imgs)} uploaded images...\n")

    for img_path in all_imgs:
        try:
            pred = is_histopathology(img_path)
        except Exception as e:
            failed.append((img_path, str(e)))
            print(f"⚠️ Error predicting {img_path}: {e}")
            continue

        try:
            src = Path(img_path)
            dest = output_hist_folder / src.name

            if pred == 1:
                # if dest exists, pick a unique filename (prevents FileExistsError)
                unique_dest = _get_unique_dest(dest)
                shutil.move(str(src), str(unique_dest))
                kept.append(str(unique_dest.resolve()))
                print(f"✔ Histopathology → KEPT: {src} -> {unique_dest.name}")
            else:
                # non-histo: remove original input file
                src.unlink(missing_ok=True)
                removed.append(str(src.resolve()))
                print(f"✖ Not Histopathology → REMOVED: {src}")

        except Exception as e:
            failed.append((img_path, str(e)))
            print(f"⚠️ Error moving/removing {img_path}: {e}")
            # don't re-raise — continue with other files

    print("\n============================================")
    print(f"✔ Histopathology images kept: {len(kept)}")
    print(f"✖ Non-histopathology removed: {len(removed)}")
    print(f"⚠️ Failed operations: {len(failed)}")
    print("============================================\n")

    return kept


# ==========================================================
# 6. WRAPPER
# ==========================================================
def preprocess_before_restaining(upload_dir: str, histo_output_dir: str = "filtered_histo"):
    kept = filter_histopathology_images(upload_dir, histo_output_dir)
    if not kept:
        print("❌ No histopathology images found. Pipeline stopping.")
        return []
    print("🎉 Filtering complete. Ready for restaining.\n")
    return kept


if __name__ == "__main__":
    # quick local test
    print(preprocess_before_restaining("input_images", "filtered_histo"))
