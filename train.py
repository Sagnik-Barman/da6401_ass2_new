"""
train.py
────────
Training entry point for all four tasks.

Usage
─────
# Task 1 – Classification
python train.py --task 1 --epochs 30 --lr 1e-3

# Task 2 – Localisation
python train.py --task 2 --epochs 25 --lr 5e-4

# Task 3 – Segmentation  (three transfer-learning modes for section 2.3)
python train.py --task 3 --freeze_mode full_freeze   --epochs 30
python train.py --task 3 --freeze_mode partial       --epochs 30
python train.py --task 3 --freeze_mode full_finetune --epochs 30

# Task 4 – Multi-task (joint training, used to build final checkpoints)
python train.py --task 4 --epochs 50 --lr 5e-4
"""

import argparse
import os
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import wandb

from data.pets_dataset import get_dataloaders, prepare_dataset, NUM_CLASSES, IMG_SIZE
from models.vgg11         import VGG11
from models.classification import ClassificationModel
from models.localization   import LocalizationModel
from models.segmentation   import SegmentationModel
from losses.iou_loss       import IoULoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def giou_loss(pred, target, eps=1e-7):
    """Generalised IoU loss — handles non-overlapping boxes better."""
    px1 = pred[:,0]-pred[:,2]/2;   py1 = pred[:,1]-pred[:,3]/2
    px2 = pred[:,0]+pred[:,2]/2;   py2 = pred[:,1]+pred[:,3]/2
    tx1 = target[:,0]-target[:,2]/2; ty1 = target[:,1]-target[:,3]/2
    tx2 = target[:,0]+target[:,2]/2; ty2 = target[:,1]+target[:,3]/2

    ix1 = torch.max(px1,tx1); iy1 = torch.max(py1,ty1)
    ix2 = torch.min(px2,tx2); iy2 = torch.min(py2,ty2)
    inter = torch.clamp(ix2-ix1,min=0)*torch.clamp(iy2-iy1,min=0)

    pa = torch.clamp(pred[:,2],min=0)*torch.clamp(pred[:,3],min=0)
    ta = torch.clamp(target[:,2],min=0)*torch.clamp(target[:,3],min=0)
    union = pa + ta - inter

    iou = inter / (union + eps)

    # Enclosing box
    cx1 = torch.min(px1,tx1); cy1 = torch.min(py1,ty1)
    cx2 = torch.max(px2,tx2); cy2 = torch.max(py2,ty2)
    c_area = torch.clamp(cx2-cx1,min=0)*torch.clamp(cy2-cy1,min=0)

    giou = iou - (c_area - union) / (c_area + eps)
    return (1 - giou).mean()
# ── Segmentation losses (inline, no extra import needed) ─────────────────
class _FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        import torch.nn.functional as F
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class _DiceLoss(nn.Module):
    def __init__(self, num_classes=3, eps=1e-6):
        super().__init__()
        self.C   = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        import torch.nn.functional as F
        probs  = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets, self.C).permute(0, 3, 1, 2).float()
        dims   = (0, 2, 3)
        inter  = (probs * tgt_oh).sum(dims)
        card   = (probs + tgt_oh).sum(dims)
        dice   = (2 * inter + self.eps) / (card + self.eps)
        return 1 - dice.mean()


class _SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = _FocalLoss()
        self.dice  = _DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)


# ── Metrics helpers ───────────────────────────────────────────────────────
def _macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", zero_division=0)


def _iou_batch(pred, gt, eps=1e-7):
    """Vectorised IoU for (cx,cy,w,h) pixel boxes → mean scalar."""
    with torch.no_grad():
        px1 = pred[:,0]-pred[:,2]/2; py1 = pred[:,1]-pred[:,3]/2
        px2 = pred[:,0]+pred[:,2]/2; py2 = pred[:,1]+pred[:,3]/2
        gx1 = gt[:,0]-gt[:,2]/2;    gy1 = gt[:,1]-gt[:,3]/2
        gx2 = gt[:,0]+gt[:,2]/2;    gy2 = gt[:,1]+gt[:,3]/2
        ix  = torch.clamp(torch.min(px2,gx2)-torch.max(px1,gx1), min=0)
        iy  = torch.clamp(torch.min(py2,gy2)-torch.max(py1,gy1), min=0)
        inter = ix * iy
        pa = torch.clamp(pred[:,2],min=0)*torch.clamp(pred[:,3],min=0)
        ga = torch.clamp(gt[:,2],  min=0)*torch.clamp(gt[:,3],  min=0)
        union = pa + ga - inter
        iou = inter / (union + eps)
        return iou.mean().item()


def _dice(pred_masks, gt_masks, C=3, eps=1e-6):
    scores = []
    for c in range(C):
        p = (pred_masks==c).float(); g = (gt_masks==c).float()
        scores.append(((2*(p*g).sum()+eps)/((p+g).sum()+eps)).item())
    import numpy as np
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════════
# Task 1 – Classification
# ══════════════════════════════════════════════════════════════════════════
def train_task1(args):
    wandb.init(project="da6401_assignment2", name=f"task1_cls_dp{args.dropout_p}",
               config=vars(args))

    train_ld, val_ld, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers)

    model     = ClassificationModel(dropout_p=args.dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_ld), epochs=args.epochs,
            pct_start=0.2)
    scaler    = GradScaler()
    best_f1   = 0.0

    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        t_loss, correct, n = 0.0, 0, 0
        for imgs, labels, _, _ in train_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            t_loss  += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n       += imgs.size(0)

        # validate
        model.eval()
        v_loss, preds_all, labels_all = 0.0, [], []
        with torch.no_grad():
            for imgs, labels, _, _ in val_ld:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                v_loss += criterion(logits, labels).item() * imgs.size(0)
                preds_all.extend(logits.argmax(1).cpu().tolist())
                labels_all.extend(labels.cpu().tolist())
        v_f1 = _macro_f1(preds_all, labels_all)
        

        wandb.log({"epoch": epoch,
                   "train/loss": t_loss/n, "train/acc": correct/n,
                   "val/loss": v_loss/len(val_ld.dataset),
                   "val/macro_f1": v_f1,
                   "lr": scheduler.get_last_lr()[0]})
        print(f"[T1] E{epoch:03d}  t_loss={t_loss/n:.4f}  val_f1={v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/classifier.pth")

    wandb.finish()
    return model


# ══════════════════════════════════════════════════════════════════════════
# Task 2 – Localisation
# ══════════════════════════════════════════════════════════════════════════
def train_task2(args):
    wandb.init(project="da6401_assignment2", name="task2_localizer",
               config=vars(args))

    train_ld, val_ld, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers)

    encoder = VGG11()
    ckpt = "checkpoints/classifier.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        enc_state = {k.replace("vgg.",""): v for k, v in state.items()
                     if k.startswith("vgg.block") or k.startswith("vgg.pool")
                     or k.startswith("vgg.avgpool")}
        encoder.load_state_dict(enc_state, strict=False)
        print("Loaded encoder from classifier.pth")

    model = LocalizationModel(encoder=encoder,
                               freeze_encoder=False).to(DEVICE)

    # Two param groups: low LR for encoder, high LR for head
    encoder_params = list(model.encoder.parameters())
    head_params    = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},
        {'params': head_params,    'lr': args.lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=[args.lr * 0.1, args.lr],
                    steps_per_epoch=len(train_ld),
                    epochs=args.epochs, pct_start=0.15,
                    div_factor=25, final_div_factor=1000)

    scaler   = GradScaler()
    best_iou = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        t_loss, t_iou, n = 0.0, 0.0, 0

        for imgs, _, bboxes, _ in train_ld:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                pred = model(imgs)
                loss = giou_loss(pred, bboxes)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            t_loss += loss.item() * imgs.size(0)
            t_iou  += _iou_batch(pred, bboxes) * imgs.size(0)
            n      += imgs.size(0)

        model.eval()
        v_loss, v_iou, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, _, bboxes, _ in val_ld:
                imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
                pred    = model(imgs)
                v_loss += giou_loss(pred, bboxes).item() * imgs.size(0)
                v_iou  += _iou_batch(pred, bboxes) * imgs.size(0)
                vn     += imgs.size(0)

        wandb.log({"epoch": epoch,
                   "train/loss": t_loss/n, "train/iou": t_iou/n,
                   "val/loss":   v_loss/vn, "val/iou":  v_iou/vn})
        print(f"[T2] E{epoch:03d}  t_iou={t_iou/n:.4f}  val_iou={v_iou/vn:.4f}")

        if v_iou/vn > best_iou:
            best_iou = v_iou/vn
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  Saved best localizer (val_iou={best_iou:.4f})")

    wandb.finish()
    return model
#def train_task2(args):
    wandb.init(project="da6401_assignment2", name="task2_localizer",
               config=vars(args))

    train_ld, val_ld, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers)

    encoder = VGG11()
    ckpt = "checkpoints/classifier.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        enc_state = {k.replace("vgg.",""): v for k, v in state.items()
                     if k.startswith("vgg.block") or k.startswith("vgg.pool")
                     or k.startswith("vgg.avgpool")}
        encoder.load_state_dict(enc_state, strict=False)
        print("Warm-started localizer encoder from classifier.pth")

    model     = LocalizationModel(encoder=encoder,
                                   freeze_encoder=False).to(DEVICE)
    mse_loss  = nn.MSELoss()        # more robust than MSE
    iou_loss  = IoULoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=args.lr,
                    steps_per_epoch=len(train_ld),
                    epochs=args.epochs, pct_start=0.1,
                    div_factor=10, final_div_factor=100)
    scaler    = GradScaler()
    best_iou  = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        t_loss, t_iou, n = 0.0, 0.0, 0
        for imgs, _, bboxes, _ in train_ld:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                pred  = model(imgs)
                loss = giou_loss(pred, bboxes)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()   
            t_loss += loss.item() * imgs.size(0)
            t_iou  += _iou_batch(pred, bboxes) * imgs.size(0)
            n      += imgs.size(0)

        model.eval()
        v_loss, v_iou, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, _, bboxes, _ in val_ld:
                imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
                pred    = model(imgs)
                v_loss += giou_loss(pred, bboxes).item() * imgs.size(0)
                v_iou  += _iou_batch(pred, bboxes) * imgs.size(0)
                vn     += imgs.size(0)
        #scheduler.step()
        wandb.log({"epoch": epoch,
                   "train/loss": t_loss/n, "train/iou": t_iou/n,
                   "val/loss":   v_loss/vn, "val/iou":  v_iou/vn})
        print(f"[T2] E{epoch:03d}  t_iou={t_iou/n:.4f}  val_iou={v_iou/vn:.4f}")

        if v_iou/vn > best_iou:
            best_iou = v_iou/vn
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  ✓ Saved best localizer (val_iou={best_iou:.4f})")

    wandb.finish()
    return model

# ══════════════════════════════════════════════════════════════════════════
# Task 3 – Segmentation
# ══════════════════════════════════════════════════════════════════════════
def train_task3(args):
    wandb.init(project="da6401_assignment2",
               name=f"task3_seg_{args.freeze_mode}", config=vars(args))

    train_ld, val_ld, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers)

    encoder = VGG11()
    ckpt = "checkpoints/classifier.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        enc_state = {k.replace("vgg.",""): v for k, v in state.items()
                     if k.startswith("vgg.block") or k.startswith("vgg.pool")
                     or k.startswith("vgg.avgpool")}
        encoder.load_state_dict(enc_state, strict=False)

    # Transfer learning strategy (section 2.3)
    if args.freeze_mode == "full_freeze":
        freeze_encoder = True
    elif args.freeze_mode == "partial":
        for name, p in encoder.named_parameters():
            if name.startswith("block0") or name.startswith("block1") \
                    or name.startswith("pool0") or name.startswith("pool1"):
                p.requires_grad_(False)
        freeze_encoder = False
    else:   # full_finetune
        freeze_encoder = False

    model     = SegmentationModel(encoder=encoder,
                                   freeze_encoder=freeze_encoder).to(DEVICE)
    criterion = _SegLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()
    best_dice = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        t_loss, n = 0.0, 0
        for imgs, _, _, masks in train_ld:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                seg_logits = model(imgs)
                loss       = criterion(seg_logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            t_loss += loss.item() * imgs.size(0)
            n      += imgs.size(0)

        model.eval()
        v_loss, v_dice, v_pacc, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, _, _, masks in val_ld:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                seg_logits  = model(imgs)
                v_loss     += criterion(seg_logits, masks).item() * imgs.size(0)
                preds       = seg_logits.argmax(1)
                v_dice     += _dice(preds, masks) * imgs.size(0)
                v_pacc     += (preds==masks).float().mean().item() * imgs.size(0)
                vn         += imgs.size(0)

        scheduler.step()
        wandb.log({"epoch": epoch,
                   "train/loss": t_loss/n,
                   "val/loss": v_loss/vn,
                   "val/dice": v_dice/vn,
                   "val/pixel_acc": v_pacc/vn})
        print(f"[T3-{args.freeze_mode}] E{epoch:03d}  "
              f"val_dice={v_dice/vn:.4f}  val_pacc={v_pacc/vn:.4f}")

        if v_dice/vn > best_dice:
            best_dice = v_dice/vn
            torch.save(model.state_dict(), "checkpoints/unet.pth")

    wandb.finish()
    return model


# ══════════════════════════════════════════════════════════════════════════
# Task 4 – Multi-task joint training
# ══════════════════════════════════════════════════════════════════════════
def train_task4(args):
    """
    Joint fine-tuning of a shared backbone + all three task heads.
    Loads pretrained head weights from checkpoints/ if present.
    """
    wandb.init(project="da6401_assignment2", name="task4_multitask",
               config=vars(args))

    train_ld, val_ld, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers)

    from models.multitask import MultiTaskPerceptionModel
    # Skip gdown during training — weights already local
    import models.multitask as mt_mod
    mt_mod._CLASSIFIER_PATH = "checkpoints/classifier.pth"
    mt_mod._LOCALIZER_PATH  = "checkpoints/localizer.pth"
    mt_mod._UNET_PATH       = "checkpoints/unet.pth"

    # Monkey-patch gdown.download to a no-op during training
    import sys, types
    fake_gdown = types.ModuleType("gdown")
    fake_gdown.download = lambda **kw: None
    sys.modules["gdown"] = fake_gdown

    model = MultiTaskPerceptionModel().to(DEVICE)

    cls_crit  = nn.CrossEntropyLoss()
    mse_crit  = nn.MSELoss()
    iou_crit  = IoULoss()
    seg_crit  = _SegLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()
    best      = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        t_total, t_cls, t_bbox, t_seg, n = 0.,0.,0.,0.,0
        for imgs, labels, bboxes, masks in train_ld:
            imgs   = imgs.to(DEVICE);   labels = labels.to(DEVICE)
            bboxes = bboxes.to(DEVICE); masks  = masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                cls_l, bbox_p, seg_l = model(imgs)
                l_cls  = cls_crit(cls_l, labels)
                l_bbox = mse_crit(bbox_p, bboxes) + iou_crit(bbox_p, bboxes)
                l_seg  = seg_crit(seg_l, masks)
                loss   = l_cls + l_bbox + l_seg
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            bs       = imgs.size(0)
            t_total += loss.item()*bs; t_cls += l_cls.item()*bs
            t_bbox  += l_bbox.item()*bs; t_seg += l_seg.item()*bs
            n       += bs

        model.eval()
        v_f1_preds, v_f1_labels = [], []
        v_iou, v_dice, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, labels, bboxes, masks in val_ld:
                imgs   = imgs.to(DEVICE);   labels = labels.to(DEVICE)
                bboxes = bboxes.to(DEVICE); masks  = masks.to(DEVICE)
                cls_l, bbox_p, seg_l = model(imgs)
                v_f1_preds.extend(cls_l.argmax(1).cpu().tolist())
                v_f1_labels.extend(labels.cpu().tolist())
                v_iou  += _iou_batch(bbox_p, bboxes) * imgs.size(0)
                v_dice += _dice(seg_l.argmax(1), masks) * imgs.size(0)
                vn     += imgs.size(0)

        v_f1 = _macro_f1(v_f1_preds, v_f1_labels)
        scheduler.step()
        wandb.log({"epoch": epoch,
                   "train/total_loss": t_total/n, "train/cls_loss": t_cls/n,
                   "train/bbox_loss": t_bbox/n,   "train/seg_loss": t_seg/n,
                   "val/macro_f1": v_f1, "val/iou": v_iou/vn,
                   "val/dice": v_dice/vn})
        print(f"[T4] E{epoch:03d}  f1={v_f1:.4f}  iou={v_iou/vn:.4f}  "
              f"dice={v_dice/vn:.4f}")

        composite = (v_f1 + v_iou/vn + v_dice/vn) / 3
        if composite > best:
            best = composite
            # Overwrite individual checkpoint files for submission
            torch.save(model.state_dict(), "checkpoints/multitask.pth")
            # Also save per-head weights so gdown step still works
            _save_split_checkpoints(model)

    wandb.finish()
    return model


def _save_split_checkpoints(model):
    """Save per-task weights into the three submission checkpoint files."""
    state = model.state_dict()
    cls_state = {k: v for k, v in state.items()
                 if k.startswith("encoder") or k.startswith("cls_head")
                 or k.startswith("avgpool")}
    loc_state = {k: v for k, v in state.items()
                 if k.startswith("bbox_head")}
    seg_state = {k: v for k, v in state.items()
                 if k.startswith("dec") or k.startswith("seg_head")}
    torch.save(cls_state, "checkpoints/classifier.pth")
    torch.save(loc_state, "checkpoints/localizer.pth")
    torch.save(seg_state, "checkpoints/unet.pth")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DA6401 A2 trainer")
    parser.add_argument("--task",          type=int,   default=4,
                        choices=[1,2,3,4])
    parser.add_argument("--data_root",     type=str,   default="./data")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--dropout_p",     type=float, default=0.5)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--freeze_encoder",action="store_true")
    parser.add_argument("--freeze_mode",   type=str,   default="full_finetune",
                        choices=["full_freeze","partial","full_finetune"])
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    prepare_dataset(args.data_root)

    {1: train_task1, 2: train_task2,
     3: train_task3, 4: train_task4}[args.task](args)


if __name__ == "__main__":
    main()
