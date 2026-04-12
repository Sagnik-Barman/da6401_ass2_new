import torch
import numpy as np
import os
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from data.pets_dataset import get_transforms, IMG_SIZE
from models.vgg11 import VGG11
from models.classification import ClassificationModel
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel

wandb.init(project="da6401_assignment2", name="section2.7_wild_pipeline")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
CMAP = np.array([[50,50,50],[0,200,0],[255,165,0]], dtype=np.uint8)

BREED_NAMES = [
    "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
    "Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue",
    "Siamese","Sphynx","American_Bulldog","American_Pit_Bull_Terrier",
    "Basset_Hound","Beagle","Boxer","Chihuahua","English_Cocker_Spaniel",
    "English_Setter","German_Shorthaired","Great_Pyrenees","Havanese",
    "Japanese_Chin","Keeshond","Leonberger","Miniature_Pinscher",
    "Newfoundland","Pomeranian","Pug","Saint_Bernard","Samoyed",
    "Scottish_Terrier","Shiba_Inu","Staffordshire_Bull_Terrier",
    "Wheaten_Terrier","Yorkshire_Terrier"
]

cls_model = ClassificationModel().to(device)
cls_model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
cls_model.eval()

enc1 = VGG11()
loc_model = LocalizationModel(encoder=enc1).to(device)
loc_model.load_state_dict(torch.load("checkpoints/localizer.pth", map_location=device))
loc_model.eval()

enc2 = VGG11()
seg_model = SegmentationModel(encoder=enc2).to(device)
seg_model.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device))
seg_model.eval()

transform = get_transforms(train=False, img_size=IMG_SIZE)
wild_dir = "wild_images"
os.makedirs("wild_outputs", exist_ok=True)

table = wandb.Table(columns=["image","breed","confidence","bbox","segmentation"])

for fname in os.listdir(wild_dir):
    if not fname.lower().endswith((".jpg",".jpeg",".png")):
        continue
    img_path = os.path.join(wild_dir, fname)
    raw = np.array(Image.open(img_path).convert("RGB"))
    aug = transform(image=raw, mask=np.zeros(raw.shape[:2], dtype=np.int64))
    inp = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits = cls_model(inp)
        bbox = loc_model(inp)
        seg_logits = seg_model(inp)

    breed_idx  = cls_logits.argmax(1).item()
    breed_name = BREED_NAMES[breed_idx] if breed_idx < len(BREED_NAMES) else f"class_{breed_idx}"
    confidence = torch.softmax(cls_logits, 1).max().item()
    bbox_np    = bbox[0].cpu().numpy()
    seg_pred   = seg_logits.argmax(1)[0].cpu().numpy()
    seg_vis    = CMAP[seg_pred]

    img_np = np.clip(inp[0].cpu().permute(1,2,0).numpy()*STD+MEAN, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    cx, cy, bw, bh = bbox_np
    rect = patches.Rectangle((cx-bw/2, cy-bh/2), bw, bh,
                               linewidth=2, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"{breed_name}\n({confidence:.1%})", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(seg_vis, alpha=0.5)
    axes[1].set_title("Seg overlay")
    axes[1].axis("off")

    axes[2].imshow(seg_vis)
    axes[2].set_title("Predicted trimap")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join("wild_outputs", fname + "_result.png")
    plt.savefig(out_path, dpi=100)
    plt.close()

    table.add_data(
        wandb.Image(out_path),
        breed_name,
        round(confidence, 4),
        str(bbox_np.round(1)),
        wandb.Image(seg_vis)
    )
    print(f"Processed {fname} -> {breed_name} ({confidence:.1%})")

wandb.log({"2.7_wild_pipeline": table})
wandb.finish()
print("Done section 2.7")
