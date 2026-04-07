import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
# 1. Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", 
    torch_dtype=torch.float16
)

# 2. Load the Img2Img Pipeline
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

# 3. Optimizations
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # Saves VRAM

raw_image = Image.open("final_generated_photo2.jpg").convert("RGB").resize((512, 512))
image_np = np.array(raw_image)

# Run Canny edge detection (the numbers control how strict the line detection is)
# Lower numbers catch more faint lines; higher numbers only catch distinct, hard lines
edges = cv2.Canny(image_np, 100, 200)

# Convert back to 3-channel PIL Image for the pipeline
edges = np.stack([edges, edges, edges], axis=2)
sketch = Image.fromarray(edges)
# sketch.show()

print("Generating image from sketch...")
prompt = "Generate a RGB photo of a female with a medium complexion and brown eyes. She has black, curly hair that frames an oval-shaped face. Her eyebrows are straight, and she has noticeably long eyelashes. Her lips are full with a rounded shape. There are no visible moles or freckles on her face. Overall, her facial features appear well-defined, with no prominent marks or irregularities beyond the described characteristics."
negative_prompt = "black and white,grayscale,cartoon, anime, 3d render, ugly, deformed, poorly drawn face, mutated, unnatural skin, hyperrealistic, 4K"

# 5. Run the Pipeline
generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=raw_image,
    control_image=sketch,         # <--- The structural guide for ControlNet
    strength=0.5,                 # <--- NEW: Controls how much the starting image changes (0.0 to 1.0)
    num_inference_steps=25,
    controlnet_conditioning_scale=0.75, 
).images[0]

generated_image.save("tomato_result.png")