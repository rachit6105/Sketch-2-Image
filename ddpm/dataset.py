import os
import random
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def extract_id(filename: str) -> str:
    name = filename.split(".")[0]
    name = name.replace("sketches_", "").replace("photos_", "")
    return name


def is_image(f: str) -> bool:
    return f.lower().endswith((".jpg", ".jpeg", ".png"))


def pil_loader(path: str, mode: str) -> Image.Image:
    img = Image.open(path).convert(mode)
    return img


def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """
    Converts PIL image to tensor in [-1, 1].
    """
    x = TF.to_tensor(img)  # [0,1]
    x = x * 2.0 - 1.0
    return x


class SketchDataset(Dataset):
    def __init__(
        self,
        sketch_dir: str,
        xco_dir: str,
        photo_dir: str,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.sketch_dir = sketch_dir
        self.xco_dir = xco_dir
        self.photo_dir = photo_dir
        self.image_size = image_size
        self.augment = augment

        self.sketch_map = {
            extract_id(f): f for f in os.listdir(sketch_dir) if is_image(f)
        }
        self.photo_map = {
            extract_id(f): f for f in os.listdir(photo_dir) if is_image(f)
        }
        self.xco_map = {
            extract_id(f): f for f in os.listdir(xco_dir) if is_image(f)
        }

        self.ids = sorted(
            set(self.sketch_map.keys()) &
            set(self.photo_map.keys()) &
            set(self.xco_map.keys())
        )

        if len(self.ids) == 0:
            raise RuntimeError("No matched sketch/xco/photo triplets found.")

        print(f"Total matched samples: {len(self.ids)}")

    def __len__(self) -> int:
        return len(self.ids)

    def _resize_triplet(
        self,
        xsk: Image.Image,
        xco: Image.Image,
        x0: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        size = [self.image_size, self.image_size]
        xsk = TF.resize(xsk, size, interpolation=TF.InterpolationMode.BILINEAR)
        xco = TF.resize(xco, size, interpolation=TF.InterpolationMode.BILINEAR)
        x0 = TF.resize(x0, size, interpolation=TF.InterpolationMode.BILINEAR)
        return xsk, xco, x0

    def _paired_augment(
        self,
        xsk: Image.Image,
        xco: Image.Image,
        x0: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Paired augmentation: same geometric transform to all 3 images.
        Mild photometric augmentation only on xco and x0.
        """
        # Horizontal flip
        if random.random() < 0.5:
            xsk = TF.hflip(xsk)
            xco = TF.hflip(xco)
            x0 = TF.hflip(x0)

        # Small affine transform
        angle = random.uniform(-5.0, 5.0)
        translate = (
            int(random.uniform(-0.03, 0.03) * self.image_size),
            int(random.uniform(-0.03, 0.03) * self.image_size),
        )
        scale = random.uniform(0.97, 1.03)
        shear = random.uniform(-2.0, 2.0)

        xsk = TF.affine(
            xsk, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR, fill=255
        )
        xco = TF.affine(
            xco, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR, fill=255
        )
        x0 = TF.affine(
            x0, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR, fill=255
        )

        # Mild color jitter only for photo-like inputs
        if random.random() < 0.5:
            brightness = random.uniform(0.95, 1.05)
            contrast = random.uniform(0.95, 1.05)
            saturation = random.uniform(0.98, 1.02)

            xco = TF.adjust_brightness(xco, brightness)
            xco = TF.adjust_contrast(xco, contrast)
            xco = TF.adjust_saturation(xco, saturation)

            x0 = TF.adjust_brightness(x0, brightness)
            x0 = TF.adjust_contrast(x0, contrast)
            x0 = TF.adjust_saturation(x0, saturation)

        return xsk, xco, x0

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]

        xsk = pil_loader(os.path.join(self.sketch_dir, self.sketch_map[id_]), mode="L")
        xco = pil_loader(os.path.join(self.xco_dir, self.xco_map[id_]), mode="RGB")
        x0  = pil_loader(os.path.join(self.photo_dir, self.photo_map[id_]), mode="RGB")

        xsk, xco, x0 = self._resize_triplet(xsk, xco, x0)

        if self.augment:
            xsk, xco, x0 = self._paired_augment(xsk, xco, x0)

        xsk = to_tensor_normalized(xsk)   # [1,H,W], [-1,1]
        xco = to_tensor_normalized(xco)   # [3,H,W], [-1,1]
        x0  = to_tensor_normalized(x0)    # [3,H,W], [-1,1]

        return xsk, xco, x0