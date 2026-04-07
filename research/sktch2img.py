import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import PIL
from controlnet_aux import HEDdetector

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", 
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # Saves VRAM

hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

raw_image = Image.open("./../test_dataset/test_sketches/312.jpg").convert("RGB").resize((512, 512))
sketch = hed(raw_image, scribble=True,detect_resolution=128)

# sketch.show()

print("Generating image from sketch...")
prompt = "Generate a photo of a white female with medium-toned skin, an oval face, and round lips. She has green eyes " \
"and straight blonde hair. She does not have any freckles or moles, and her distinguishing features include wearing " \
"round glasses."
image = pipe(
    prompt,
    image=sketch,
    num_inference_steps=20,
    controlnet_conditioning_scale= 0.9, # How strictly to follow the sketch (0.0 to 1.0)
).images[0]
image.show()
# image.save("tomato_result.png")