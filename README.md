# DA6401 Assignment 2 — Visual Perception Pipeline

**W&B Report:** `<paste your public W&B report link here>`  
**GitHub Repo:** `<paste your GitHub repo link here>`

---

## Project Structure

```
.
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── inference.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.py
├── models/
│   ├── __init__.py
│   ├── classification.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── multitask.py
├── README.md
├── requirements.txt
└── train.py
```

## Quick Start

```bash
pip install -r requirements.txt

# Task 1 – VGG11 Classification
python train.py --task 1 --epochs 30 --batch_size 32 --lr 1e-3

# Task 2 – Localisation
python train.py --task 2 --epochs 25

# Task 3 – Segmentation (three TL strategies for W&B section 2.3)
python train.py --task 3 --freeze_mode full_freeze   --epochs 30
python train.py --task 3 --freeze_mode partial       --epochs 30
python train.py --task 3 --freeze_mode full_finetune --epochs 30

# Task 4 – Multi-task joint fine-tuning
python train.py --task 4 --epochs 50

# Inference on a novel image
python inference.py --image my_pet.jpg
```

## Autograder Imports

```python
from models.vgg11    import VGG11
from models.layers   import CustomDropout
from losses.iou_loss import IoULoss
from multitask       import MultiTaskPerceptionModel
```

## Architecture Summary

| Component | Details |
|-----------|---------|
| VGG11 backbone | 5 conv blocks (config A), BN after every Conv, CustomDropout in FC head |
| BBox output | pixel coords `(cx, cy, w, h)` at 224×224 |
| Localisation loss | MSE + IoULoss |
| Segmentation decoder | 5 TransposedConv stages + skip connections |
| Seg loss | 0.5 × Focal + 0.5 × Dice |
| Multi-task | shared encoder → 3 heads in one `forward()` |
