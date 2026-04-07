"""
inference.py
────────────
Run the unified MultiTaskPerceptionModel on a single image or folder.

Usage
─────
python inference.py --image path/to/pet.jpg
python inference.py --folder path/to/images/
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.pets_dataset import get_transforms, IMG_SIZE
from multitask import MultiTaskPerceptionModel   # autograder path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = np.array([0.485, 0.456, 0.406])
STD    = np.array([0.229, 0.224, 0.225])
CMAP   = np.array([[0,0,0],[0,200,0],[255,165,0]], dtype=np.uint8)

# Oxford-IIIT Pet breed names (alphabetical order, 0-indexed)
BREED_NAMES = [
    "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
    "Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue",
    "Siamese","Sphynx","American_Bulldog","American_Pit_Bull_Terrier",
    "Basset_Hound","Beagle","Boxer","Chihuahua","English_Cocker_Spaniel",
    "English_Setter","German_Shorthaired","Great_Pyrenees","Havanese",
    "Japanese_Chin","Keeshond","Leonberger","Miniature_Pinscher",
    "Newfoundland","Pomeranian","Pug","Saint_Bernard","Samoyed",
    "Scottish_Terrier","Shiba_Inu","Staffordshire_Bull_Terrier",
    "Wheaten_Terrier","Yorkshire_Terrier",
]


def load_model() -> MultiTaskPerceptionModel:
    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()
    return model


def preprocess(img_path: str) -> torch.Tensor:
    img = np.array(Image.open(img_path).convert("RGB"))
    tfm = get_transforms(train=False, img_size=IMG_SIZE)
    return tfm(image=img, mask=np.zeros(img.shape[:2], dtype=np.int64))["image"] \
             .unsqueeze(0).to(DEVICE)


def run_inference(model: MultiTaskPerceptionModel, img_path: str,
                  save_path: str = None):
    inp = preprocess(img_path)

    with torch.no_grad():
        cls_logits, bbox, seg_logits = model(inp)

    breed_idx  = cls_logits.argmax(1).item()
    breed_name = BREED_NAMES[breed_idx] if breed_idx < len(BREED_NAMES) \
                 else f"class_{breed_idx}"
    confidence = torch.softmax(cls_logits, 1).max().item()
    bbox_np    = bbox[0].cpu().numpy()         # (cx, cy, w, h) pixels
    seg_pred   = seg_logits.argmax(1)[0].cpu().numpy()
    seg_vis    = CMAP[seg_pred]

    # Denormalise for display
    img_np = inp[0].cpu().permute(1,2,0).numpy()
    img_np = np.clip(img_np * STD + MEAN, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original + bbox
    axes[0].imshow(img_np)
    cx, cy, bw, bh = bbox_np
    x1 = cx - bw/2;  y1 = cy - bh/2
    rect = patches.Rectangle((x1, y1), bw, bh,
                               linewidth=2, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"{breed_name} ({confidence:.2%})")
    axes[0].axis("off")

    # Segmentation
    axes[1].imshow(img_np)
    axes[1].imshow(seg_vis, alpha=0.5)
    axes[1].set_title("Segmentation overlay")
    axes[1].axis("off")

    # Pure seg mask
    axes[2].imshow(seg_vis)
    axes[2].set_title("Predicted trimap")
    axes[2].axis("off")

    plt.tight_layout()
    out = save_path or (os.path.splitext(img_path)[0] + "_prediction.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved prediction to {out}")

    return {
        "breed":      breed_name,
        "confidence": confidence,
        "bbox_pixels": bbox_np.tolist(),
        "seg_pred":   seg_pred,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA6401 A2 Inference")
    parser.add_argument("--image",  type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model()

    if args.image:
        run_inference(model, args.image,
                      os.path.join(args.output_dir,
                                   os.path.basename(args.image)))
    elif args.folder:
        exts = {".jpg",".jpeg",".png",".bmp"}
        for fn in os.listdir(args.folder):
            if os.path.splitext(fn)[1].lower() in exts:
                run_inference(model,
                              os.path.join(args.folder, fn),
                              os.path.join(args.output_dir, fn))
    else:
        parser.print_help()
