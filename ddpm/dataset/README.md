# Dataset Directory

This directory is configured to hold the image data required for training and processing. To keep the repository lightweight, the actual image files are ignored by Git.

## Setup Instructions

Before running the training or inference scripts, you need to manually create the following folder structure within this `dataset/` directory to ensure the paths in the code resolve correctly:

1.  **`trainA/`**: Place your source **sketch** images here.
    * *Code Reference:* `sketch_dir`
2.  **`trainB/`**: Place your target **real photos** here.
    * *Code Reference:* `photo_dir`
3.  **`cyclegan/`**: This folder will store the output/intermediate images from the **CycleGAN** process.
    * *Code Reference:* `xco_dir`

## Quick Setup (Terminal)
If you are on Linux or Mac, you can run this command from inside the `ddpm/dataset/` folder to create them all at once:

```bash
mkdir -p trainA trainB cyclegan