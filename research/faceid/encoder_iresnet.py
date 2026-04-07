import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from iresnet import iresnet50

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class SketchFaceIDDataset(Dataset):
    def __init__(self, sketch_dir, embed_dir, filenames, transform=None):
        self.sketch_dir = sketch_dir
        self.embed_dir  = embed_dir
        self.transform  = transform
        self.filenames  = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename   = self.filenames[idx]
        base_name  = os.path.splitext(filename)[0]
        embed_base = base_name.replace("sketches", "photos")

        sketch = Image.open(
            os.path.join(self.sketch_dir, filename)
        ).convert('RGB')

        if self.transform:
            sketch = self.transform(sketch)

        target_embedding = torch.load(
            os.path.join(self.embed_dir, f"{embed_base}.pt"),
            weights_only=True
        )
        return sketch, target_embedding


# IResNet expects exactly 112x112, [-1, 1] normalized — same as buffalo_l's
# preprocessing. No hue jitter (sketches are grayscale). RandomErasing
# simulates incomplete strokes. No ColorJitter saturation for same reason.

train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=5,              # less rotation — CycleGAN outputs have real structure
        translate=(0.03, 0.03), # small shifts only
        scale=(0.95, 1.05),
    ),
    transforms.ColorJitter(
        brightness=0.2,         # was 0.4 — CycleGAN color is already fragile
        contrast=0.2,           # was 0.4
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Removed RandomErasing — not appropriate for CycleGAN outputs
])

val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

""" Freeze strategy notes:
    ── Model ──────────────────────────────────────────────────────────────────────
    IResNet50 named parameter groups (verify with check_layer_sensitivity below):

    conv1, bn1                    ← stem, detects raw edges/strokes
    layer1                        ← low-level: edges, stroke junctions
    layer2                        ← mid-level: starts encoding texture
    layer3                        ← mid-high: photo texture heavy, freeze
    layer4                        ← high-level: identity geometry, freeze
    bn2                           ← output BN, ALWAYS unfreeze (dist. shifts)
    fc (if present)               ← final projection, unfreeze

    Default strategy: train conv1+bn1+layer1+layer2+bn2, freeze layer3+layer4.
    Override after running check_layer_sensitivity() on your actual data.
"""

class SketchToFaceEncoder(nn.Module):
    def __init__(self, weights_path: str):
        super().__init__()

        self.backbone = iresnet50(pretrained=False)

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state)
            print(f"Loaded InsightFace weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Always unfreeze output BN layers — they normalize the distribution
        # of backbone outputs, which shifts when input domain changes from
        # photos to sketches. These are only 1024 params each, negligible risk.
        for layer_name in ['bn2', 'features']:
            for param in getattr(self.backbone, layer_name).parameters():
                param.requires_grad = True

        # Translator carries the full domain adaptation burden.
        # Wider (1024 hidden) to compensate for frozen backbone.
        # Heavy dropout (0.5 in early layers) is the primary overfitting defense.
        # LayerNorm instead of BatchNorm — more stable with small batches,
        # and doesn't accumulate running stats that overfit to training distribution.
        self.translator = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=0.3),      # was 0.5

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=0.4),     # was 0.4

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),     # was 0.3

            nn.Linear(512, 512),
        )

        self._init_translator_small()

    def _init_translator_small(self):
        # Initialize translator weights small so residual output starts
        # close to the pretrained embedding. Without this, early epochs
        # produce garbage embeddings and waste the warmup budget.
        for m in self.translator.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        with torch.no_grad():
            # Explicit no_grad on backbone — prevents any accidental gradient
            # flow even if requires_grad is somehow set
            features = self.backbone(x)

        translated = self.translator(features)
        out = features + translated          # residual keeps output near pretrained space
        return F.normalize(out, p=2, dim=1)
    
    def apply_freeze_strategy(self, unfreeze_layers: list[str]):
        """
        Freeze entire backbone, then selectively unfreeze by layer name.
        Always keeps bn2 trainable regardless of unfreeze_layers.

        Args:
            unfreeze_layers: list of backbone attribute names to unfreeze,
                             e.g. ['conv1', 'bn1', 'layer1', 'layer2']
        """
        # Step 1: freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Step 2: unfreeze requested layers
        for layer_name in unfreeze_layers:
            layer = getattr(self.backbone, layer_name, None)
            if layer is None:
                print(f"  WARNING: backbone has no attribute '{layer_name}', skipping")
                continue
            for param in layer.parameters():
                param.requires_grad = True
            print(f"  Unfrozen: {layer_name}")

        # Step 3: always unfreeze bn2 — output distribution shifts when input
        # domain changes from photos to sketches, even if deep layers are frozen
        if hasattr(self.backbone, 'bn2'):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
            print("  Unfrozen: bn2 (always)")

        # Translator is always trainable (handled outside, but log it)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")


# ── Loss ───────────────────────────────────────────────────────────────────────
# MSE is WRONG here: after L2 normalization all embeddings are on the unit
# hypersphere. MSE measures Euclidean distance in ambient 512-d space, but
# what actually matters is the angle between vectors (cosine similarity).
# MSE and cosine send contradictory gradient signals on normalized vectors.
# Use cosine alignment + triplet only.

class AlignmentLoss(nn.Module):
    def __init__(self, triplet_margin=0.3, triplet_weight=0.5):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a, b: 1 - F.cosine_similarity(a, b),
            margin=triplet_margin,
        )

    def forward(self, sketch_emb, face_emb):
        # Direct alignment: minimize angular distance to target
        cosine_loss = (1 - F.cosine_similarity(sketch_emb, face_emb)).mean()

        # Contrastive: push away from other identities in the same batch.
        # roll(1) gives every sample a different identity as its negative.
        # Works best with large batch sizes (>=64) so negatives are diverse.
        neg_emb      = torch.roll(face_emb, shifts=1, dims=0)
        triplet_loss = self.triplet(sketch_emb, face_emb, neg_emb)

        return cosine_loss + self.triplet_weight * triplet_loss


# ── Sensitivity check ──────────────────────────────────────────────────────────

def check_layer_sensitivity(model, loader, n_batches=30):
    """
    Run BEFORE committing to a freeze strategy.
    Temporarily unfreeze everything, measure gradient norms per layer group.
    High norm = that layer needs to change a lot = keep trainable.
    Sharp drop = safe freeze boundary.
    """
    print("Running layer sensitivity check...")
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    criterion  = AlignmentLoss()
    grad_norms = {}

    for i, (sketches, face_embs) in enumerate(loader):
        if i >= n_batches:
            break
        sketches, face_embs = sketches.to(device), face_embs.to(device)
        model.zero_grad()
        loss = criterion(model(sketches), face_embs)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # group by top-level component: conv1, layer1, layer2, etc.
            parts = name.split('.')
            key   = parts[1] if parts[0] == 'backbone' else parts[0]
            grad_norms.setdefault(key, []).append(param.grad.norm().item())

    print("\n── Layer sensitivity (avg gradient norm) ──")
    for k, v in sorted(grad_norms.items(), key=lambda x: -np.mean(x[1])):
        bar = '█' * int(np.mean(v) * 500)
        print(f"  {k:<20} {np.mean(v):.6f}  {bar}")
    print("\n  Freeze layers where the norm drops sharply.\n")


# ── Loader ─────────────────────────────────────────────────────────────────────

def loader(sketch_path, embed_path):
    valid_filenames = [
        f for f in os.listdir(sketch_path)
        if f.endswith(('.jpg', '.png'))
        and os.path.exists(
            os.path.join(
                embed_path,
                os.path.splitext(f)[0].replace("sketches", "photos") + ".pt"
            )
        )
    ]

    random.shuffle(valid_filenames)
    split_idx   = int(len(valid_filenames) * 0.9)
    train_files = valid_filenames[:split_idx]
    val_files   = valid_filenames[split_idx:]

    train_dataset = SketchFaceIDDataset(sketch_path, embed_path, train_files, train_transform)
    val_dataset   = SketchFaceIDDataset(sketch_path, embed_path, val_files,   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_files, val_files


def model_check(model):
    for name, module in model.backbone.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"{name:<20} {type(module).__name__:<30} params: {n_params:,}")
# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sketch_folder  = "/home/tichar/Documents/ugp_vlm/diffuse/dataset/results_new"
    embed_folder   = "/home/tichar/Documents/ugp_vlm/diffuse/dataset/embeddings"
    weights_path   = "insightface_r50_weights.pth"

    train_loader, val_loader, train_files, val_files = loader(sketch_folder, embed_folder)

    model = SketchToFaceEncoder(weights_path).to(device)
    # model_check(model)
    # check_layer_sensitivity(model, train_loader)

    model.apply_freeze_strategy(
        # unfreeze_layers=['conv1', 'layer2', 'layer3', 'layer4', 'fc']
        unfreeze_layers=['features']
    )

    for param in model.translator.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-2,   # stronger weight decay than before — small data needs it
    )


    # ── Scheduler: warmup then cosine decay ────────────────────────────────
    # Warmup stabilizes the translator before backbone layers start moving.
    # Without it, first-batch gradients are huge and corrupt pretrained weights.
    total_epochs  = 150
    warmup_epochs = 10  
    patience     = 20
    no_improve   = 0

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_epochs
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            ),
        ],
        milestones=[warmup_epochs],
    )

    criterion  = AlignmentLoss(triplet_margin=0.3, triplet_weight=0.5)
    best_val   = float('inf')

    print(f"Training | {len(train_files)} train | {len(val_files)} val")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>12}")
    model.load_state_dict(torch.load("best_sketch_encoder.pth", map_location=device), strict=True)
    for epoch in range(total_epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_train = 0.0

        for sketches, face_embs in train_loader:
            sketches, face_embs = sketches.to(device), face_embs.to(device)

            optimizer.zero_grad()
            out  = model(sketches)
            loss = criterion(out, face_embs)
            loss.backward()

            # Gradient clipping prevents translator blowup during warmup
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()

        scheduler.step()
        avg_train = total_train / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        total_val    = 0.0
        total_sim    = 0.0
        total_active = 0.0   # accumulate across batches

        with torch.no_grad():
            for sketches, face_embs in val_loader:
                sketches, face_embs = sketches.to(device), face_embs.to(device)
                out  = model(sketches)
                loss = criterion(out, face_embs)
                total_val += loss.item()
                total_sim += F.cosine_similarity(out, face_embs).mean().item()

                # Active triplet diagnostic — must be inside the loop
                # because out and face_embs are defined per batch
                neg_emb  = torch.roll(face_embs, shifts=1, dims=0)
                pos_dist = 1 - F.cosine_similarity(out, face_embs)
                neg_dist = 1 - F.cosine_similarity(out, neg_emb)
                total_active += (pos_dist - neg_dist + 0.3 > 0).float().mean().item()

        avg_val    = total_val    / len(val_loader)
        avg_sim    = total_sim    / len(val_loader)
        avg_active = total_active / len(val_loader)   # average across batches
        cur_lr     = scheduler.get_last_lr()[0]

        print(f"{epoch+1:>6}  {avg_train:>10.4f}  {avg_val:>10.4f}  "
            f"{cur_lr:>12.2e}  sim={avg_sim:.4f}  active={avg_active:.2%}")
        
        if avg_val < best_val:
            best_val   = avg_val
            no_improve = 0
            torch.save(model.state_dict(), "best_sketch_encoder.pth")
            print("         └─ saved best_sketch_encoder.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Done.")