import os
import random
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from models.refiner import SketchToPhotoRefiner
from dataset import SketchDataset


# -------------------------------------------------
# Config
# -------------------------------------------------

SEED = 42
BATCH_SIZE = 8
EPOCHS = 200
LR = 5e-5
NUM_WORKERS = 4
TIMESTEPS = 200
IMG_SAVE_EVERY = 5
CKPT_SAVE_EVERY = 10
GRAD_CLIP = 1.0
EMA_DECAY = 0.99
VAL_RATIO = 0.1
EMA_WARMUP_EPOCHS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples", exist_ok=True)


# -------------------------------------------------
# EMA helper
# -------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
base_train_dataset = SketchDataset(
    sketch_dir="dataset/trainA",
    xco_dir="dataset/cyclegan",
    photo_dir="dataset/trainB",
    image_size=224,
    augment=True,
)

base_val_dataset = SketchDataset(
    sketch_dir="dataset/trainA",
    xco_dir="dataset/cyclegan",
    photo_dir="dataset/trainB",
    image_size=224,
    augment=False,
)

indices = list(range(len(base_train_dataset)))
random.Random(SEED).shuffle(indices)

val_size = max(1, int(len(indices) * VAL_RATIO))
val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_dataset = Subset(base_train_dataset, train_indices)
val_dataset = Subset(base_val_dataset, val_indices)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)

# -------------------------------------------------
# Model
# -------------------------------------------------

model = SketchToPhotoRefiner(
    sketch_channels=1,
    photo_channels=3,
    base=64,
    time_dim=256,
    timesteps=TIMESTEPS,
    dropout=0.05,
).to(device)

ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)

ema = EMA(model, decay=EMA_DECAY)
ema.copy_to(ema_model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


# -------------------------------------------------
# Utility
# -------------------------------------------------

def denorm(x):
    # assumes dataset tensors are in [-1, 1]
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def evaluate(model_to_eval):
    model_to_eval.eval()
    total = 0.0
    count = 0

    for xsk, xco, x0 in val_loader:
        xsk = xsk.to(device, non_blocking=True)
        xco = xco.to(device, non_blocking=True)
        x0 = x0.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            eps_pred, eps_true = model_to_eval.training_forward(xsk, xco, x0)
            loss = F.mse_loss(eps_pred, eps_true)

        total += loss.item()
        count += 1

    return total / max(count, 1)


@torch.no_grad()
def save_samples(model_to_sample, epoch):
    model_to_sample.eval()
    batch = next(iter(val_loader))
    xsk, xco, x0 = batch
    xsk = xsk[:4].to(device)
    xco = xco[:4].to(device)
    x0 = x0[:4].to(device)

    pred = model_to_sample.sample(xsk, xco)

    save_image(denorm(xsk.repeat(1, 3, 1, 1)), f"samples/epoch_{epoch:03d}_sketch.png", nrow=4)
    save_image(denorm(xco), f"samples/epoch_{epoch:03d}_coarse.png", nrow=4)
    save_image(denorm(pred), f"samples/epoch_{epoch:03d}_refined.png", nrow=4)
    save_image(denorm(x0), f"samples/epoch_{epoch:03d}_gt.png", nrow=4)


# -------------------------------------------------
# Training loop
# -------------------------------------------------

best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xsk, xco, x0 in train_loader:
        xsk = xsk.to(device, non_blocking=True)
        xco = xco.to(device, non_blocking=True)
        x0 = x0.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=
        (device == "cuda")):
            eps_pred, eps_true = model.training_forward(xsk, xco, x0)
            loss = F.mse_loss(eps_pred, eps_true)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if epoch > EMA_WARMUP_EPOCHS:
            ema.update(model)
        total_loss += loss.item()

    scheduler.step()

    # eval current model
    train_loss = total_loss / len(train_loader)
    val_loss = evaluate(model)

    # eval ema model
    if epoch > EMA_WARMUP_EPOCHS:
        ema.copy_to(ema_model)
        ema_val_loss = evaluate(ema_model)
    else:
        ema_val_loss = val_loss

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"EMA Val Loss: {ema_val_loss:.6f} | "
        f"LR: {scheduler.get_last_lr()[0]:.7f}"
    )

    metric_for_best = ema_val_loss if epoch > EMA_WARMUP_EPOCHS else val_loss

    if metric_for_best < best_val:
        best_val = metric_for_best
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
            },
            "checkpoints/best.pth",
        )

    if epoch % IMG_SAVE_EVERY == 0:
        save_samples(ema_model, epoch)

    if epoch % CKPT_SAVE_EVERY == 0:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
            },
            f"checkpoints/epoch_{epoch:03d}.pth",
        )

torch.save(
    {
        "epoch": EPOCHS,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
    },
    "checkpoints/last.pth",
)