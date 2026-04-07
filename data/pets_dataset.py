"""
data/pets_dataset.py
────────────────────
Oxford-IIIT Pet Dataset — returns (image, label, bbox, mask).

bbox format : [x_center, y_center, width, height]  in PIXEL coordinates
              (as required by the assignment README).
mask values : 0 = background, 1 = foreground, 2 = boundary
              (trimap values 1,2,3 remapped to 0,1,2)
"""

import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Constants ────────────────────────────────────────────────────────────
IMG_SIZE   = 224          # VGG11 standard input
NUM_CLASSES = 37

_URL_IMGS  = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
_URL_ANNS  = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Download helpers ─────────────────────────────────────────────────────
def _download(url: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, url.split("/")[-1])
    if not os.path.exists(fname):
        print(f"Downloading {url} …")
        urllib.request.urlretrieve(url, fname)
    print(f"Extracting {fname} …")
    with tarfile.open(fname, "r:gz") as t:
        t.extractall(dest_dir)


def prepare_dataset(root: str = "./data") -> None:
    if not os.path.isdir(os.path.join(root, "images")):
        _download(_URL_IMGS, root)
    if not os.path.isdir(os.path.join(root, "annotations")):
        _download(_URL_ANNS, root)


# ── Annotation parsers ───────────────────────────────────────────────────
def _build_class_map(ann_dir: str) -> dict:
    """Returns {img_stem: class_id_0indexed}."""
    mapping = {}
    with open(os.path.join(ann_dir, "list.txt")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = int(parts[1]) - 1   # 0-indexed
    return mapping


def _parse_bbox_xmls(ann_dir: str) -> dict:
    """
    Returns {img_stem: [cx, cy, w, h]} in PIXEL coordinates
    relative to the original image size — we rescale to IMG_SIZE later.
    """
    xmls_dir = os.path.join(ann_dir, "xmls")
    bboxes = {}
    if not os.path.isdir(xmls_dir):
        return bboxes
    for xf in Path(xmls_dir).glob("*.xml"):
        tree = ET.parse(xf)
        root = tree.getroot()
        size = root.find("size")
        W = int(size.find("width").text)
        H = int(size.find("height").text)
        obj = root.find("object")
        if obj is None:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        # Store as normalised (0-1); we convert to pixels in __getitem__
        # after the image is resized to IMG_SIZE × IMG_SIZE.
        cx = (xmin + xmax) / 2 / W
        cy = (ymin + ymax) / 2 / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        bboxes[xf.stem] = [cx, cy, bw, bh]   # normalised fractions
    return bboxes


# ── Dataset ──────────────────────────────────────────────────────────────
class OxfordPetDataset(Dataset):
    """
    Parameters
    ----------
    root      : dataset root that contains images/ and annotations/
    split     : 'trainval' | 'test'
    transform : albumentations Compose (applied to both image and mask)
    img_size  : spatial size to resize to (default 224)
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "trainval",
        transform=None,
        img_size: int = IMG_SIZE,
    ):
        self.img_dir  = os.path.join(root, "images")
        self.ann_dir  = os.path.join(root, "annotations")
        self.img_size = img_size

        self.cls_map  = _build_class_map(self.ann_dir)
        self.bbox_map = _parse_bbox_xmls(self.ann_dir)

        self.transform = transform if transform is not None \
                         else get_transforms(train=(split == "trainval"),
                                             img_size=img_size)

        # Build sample list
        list_file = os.path.join(self.ann_dir, f"{split}.txt")
        self.samples = []
        with open(list_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                name = parts[0]
                img_path  = os.path.join(self.img_dir, name + ".jpg")
                mask_path = os.path.join(self.ann_dir, "trimaps", name + ".png")
                if os.path.exists(img_path) and name in self.cls_map:
                    self.samples.append((name, img_path, mask_path))

    # ── helpers ──────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, img_path, mask_path = self.samples[idx]

        # Image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Mask  (trimap: 1=fg, 2=bg, 3=boundary → remap 0,1,2)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path)).astype(np.int64) - 1
            mask = np.clip(mask, 0, 2)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.int64)

        # Augment / resize
        aug   = self.transform(image=image, mask=mask)
        image = aug["image"]          # FloatTensor [3, H, W]
        mask  = aug["mask"].long()    # LongTensor  [H, W]

        # Label
        label = torch.tensor(self.cls_map[name], dtype=torch.long)

        # BBox  — convert normalised fractions → pixel coords at IMG_SIZE
        nbox  = self.bbox_map.get(name, [0.5, 0.5, 1.0, 1.0])
        bbox  = torch.tensor(
            [nbox[0] * self.img_size,   # cx (pixels)
             nbox[1] * self.img_size,   # cy (pixels)
             nbox[2] * self.img_size,   # w  (pixels)
             nbox[3] * self.img_size],  # h  (pixels)
            dtype=torch.float32,
        )

        return image, label, bbox, mask


# ── Transforms ───────────────────────────────────────────────────────────
def get_transforms(train: bool = True, img_size: int = IMG_SIZE) -> A.Compose:
    if train:
        return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                           rotate_limit=20, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── DataLoader factory ───────────────────────────────────────────────────
def get_dataloaders(
    root: str       = "./data",
    batch_size: int = 32,
    val_frac: float = 0.2,
    num_workers: int= 4,
    img_size: int   = IMG_SIZE,
    seed: int       = 42,
):
    """Returns (train_loader, val_loader, test_loader)."""
    train_ds_full = OxfordPetDataset(
        root=root, split="trainval",
        transform=get_transforms(train=True, img_size=img_size),
    )
    test_ds = OxfordPetDataset(
        root=root, split="test",
        transform=get_transforms(train=False, img_size=img_size),
    )

    n_val   = int(len(train_ds_full) * val_frac)
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
