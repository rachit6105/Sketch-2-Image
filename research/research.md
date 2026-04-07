# Project Overview

This repository contains the following components:

* **ControlNet – Image with Text Conditioning**
  `skthc2img.py`

* **ControlNet – Image-to-Image Conditioning**
  `img2img.py`
  (with scribble, canny, lineart, HED)

* **QWEN Extractor for JSON Text Generation**
  `qwen_extract.py`

* **FaceID Embeddings (Alternative to CLIP)**
  `faceid/`
----------------------------------------------
# Things Tried

* **Benchmarking:**
  * [DeepFace Drawing](https://github.com/IGLICT/DeepFaceDrawing-Jittor.git), [DiffFaceSketch](https://github.com/puckikk1202/difffacesketch2023.git) 
  * <u>Issues</u>: Optimized for scribbles, not detailed forensic sketches, Confused for detailed sketches

* **Coarse Generator:**
  * We trained a cycleGAN to first colourize images then use DDPM on it following [this](https://ieeexplore.ieee.org/document/10547051) paper.
  * Issue: averaging effect → poor guidance, diffusion hallucinations,you can see the results [here](generated_photos/).

* **Diffusion Training:**

  * Custom DDPM pipeline with coarse generator
  * Issue: ineffective learning on small dataset

* **Sketch Encoder Training:**

  * Goal: align sketch embeddings with photo embedding space (InsightFace Buffalo / iResNet)
  * Issue: overfitting on small dataset (~2.6k samples)

* **Encoder Architectures Tried:**

  * ResNet50
  * VGGFace2
  * iResNet
  * Outcome: Similar overfitting; iResNet performed best

* **ControlNet Pipelines:**

  * Scribble, Lineart, Canny + Stable Diffusion
  * Issue: outputs too artistic / unrealistic

* **Realism Improvement:**

  * Used Realistic Vision V5.1 for better facial quality

* **Schedulers Tried:**

  * UniPCMultistepScheduler
  * DPMSolverSDEScheduler
  * Goal: improve skin texture and fine details

* **Evaluation Metric:**

  * InsightFace facial authentication score
  * facial_recognition authentication

