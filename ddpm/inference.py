import os

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from models.refiner import SketchToPhotoRefiner


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_input_image(path: str, mode: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert(mode)
    img = TF.resize(img, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    x = TF.to_tensor(img)          # [0,1]
    x = x * 2.0 - 1.0              # [-1,1]
    return x


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


# --------------------------------
# Load model
# --------------------------------
model = SketchToPhotoRefiner(
    sketch_channels=1,
    photo_channels=3,
    base=64,
    time_dim=256,
    timesteps=200,
    dropout=0.05,
).to(device)

ckpt_path = "checkpoints/best.pth"
ckpt = torch.load(ckpt_path, map_location=device)

# Prefer EMA weights if present
if "ema_model" in ckpt:
    model.load_state_dict(ckpt["ema_model"])
elif "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)

model.eval()


# --------------------------------
# Load inputs
# --------------------------------
xsk = load_input_image("test/sketch.png", mode="L", image_size=224).unsqueeze(0).to(device)
xco = load_input_image("test/cyclegan.png", mode="RGB", image_size=224).unsqueeze(0).to(device)


# --------------------------------
# Generate
# --------------------------------
with torch.no_grad():
    output = model.sample(xsk, xco, clamp=True)

print("Output shape:", output.shape)


# --------------------------------
# Save output
# --------------------------------
os.makedirs("outputs", exist_ok=True)

out_img = denorm(output[0]).cpu()
TF.to_pil_image(out_img).save("outputs/refined.png")

print("Saved refined image to outputs/refined.png")