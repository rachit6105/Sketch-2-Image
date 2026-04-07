import torch
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
import PIL.ImageOps
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize InsightFace (The Biometric Extractor)
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load the specific FaceID LoRA and Models
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16
).to(device)

# 3. Load the IP-Adapter FaceID Weights
# This specifically tells the pipeline to accept InsightFace embeddings
pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder="", weight_name="ip-adapter-faceid_sd15.bin")

# 4. Process Inputs
path = '35.jpg'
target_size = (512, 512)
sketch_image = load_image(f"test_sketches/{path}").convert("RGB").resize(target_size)
inverted_sketch = PIL.ImageOps.invert(sketch_image)

# Load CycleGAN image specifically for Face Extraction
cycle_image_cv2 = cv2.imread(f"generated_photos/{path}") 
faces = app.get(cycle_image_cv2)

if len(faces) == 0:
    print("InsightFace could not detect a face in the CycleGAN image!")
    # Handle error...

# Extract the biometric embedding from the CycleGAN image
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

# 5. Prompts
text_prompt = "RAW documentary portrait photo, unflattering police mugshot style, unairbrushed, skin pores, plain background"
negative_prompt = "illustration, 3d render, smooth skin, airbrushed, beauty filter, cartoon"

# 6. Generate with BOTH FaceID and ControlNet
final_photo = pipe(
    prompt=text_prompt,
    negative_prompt=negative_prompt,
    image=inverted_sketch,                 # Goes to ControlNet
    ip_adapter_image_embeds=[faceid_embeds], # Goes to IP-Adapter FaceID
    controlnet_conditioning_scale=0.8,     # Slightly lowered to let FaceID influence the structure
    cross_attention_kwargs={"scale": 0.8}, # IP-Adapter strength (0.8 is usually the sweet spot)
    guidance_scale=6.0,
    num_inference_steps=30,
).images[0]

final_photo.save("35_FaceID_Verified.jpg")