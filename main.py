import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import PIL.ImageOps

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load the Photorealistic VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", 
    torch_dtype=torch.float16
)

# 2. Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart",
    torch_dtype=torch.float16
)

longclip_id = "zer0int/LongCLIP-L-Diffusers"
text_encoder = CLIPTextModel.from_pretrained(longclip_id, torch_dtype=torch.float16)
tokenizer = CLIPTokenizer.from_pretrained(longclip_id, model_max_length=248)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE", # Much better for real skin than base SD1.5
    controlnet=controlnet,
    vae=vae,                                # Inject the new VAE
    text_encoder = text_encoder,
    tokenizer = tokenizer,
    torch_dtype=torch.float16
).to(device)

pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
pipe.load_lora_weights("Skin Tone Slider - LoRA_v1.0.safetensors", adapter_name="skin_tone")
pipe.set_adapters(["skin_tone"], adapter_weights=[5.0])
# 4. Load inputs
target_size = (512, 512)
path = '143.jpg'
actual_size = load_image(f"test_dataset/test_sketches/{path}").size
sketch_image = load_image(f"test_dataset/test_sketches/{path}").convert("RGB").resize(target_size)

inverted_sketch = PIL.ImageOps.invert(sketch_image)

clinical_description = (
    "The person I saw was a female, likely of Mediterranean or Caucasian descent, in her late 20s to mid-30s. She had a light olive complexion and an oval-shaped face with a smooth jawline tapering down to a slightly pointed chin. Her hair was dark red, straight, parted to one side, and tucked behind her ears. Her eyes were dark brown, set beneath moderately thick, dark eyebrows. She was wearing thin-framed, oval glasses with a dak colour. Her nose was straight, and her lips were of average thickness, resting in a neutral expression. Her skin looked entirely natural and unretouched, showing visible pores, slight unevenness in tone, and faint shadows beneath her eyes and lower lip. Her most distinguishing features were her oval glasses, a small dark freckle or mole high on her left cheekbone (viewer's right), and small dark stud earrings."
)
## You don't need to alter the following prompt just add clinical description
text_prompt = (
    f"Amateur photo, DMV camera, (extreme skin texture:1.4), (visible pores:1.3), (wrinkles, eye bags, nasolabial folds:1.3), hyper-detailed unairbrushed skin,"\
      f"uneven skin tone, harsh flash lighting, ugly, documentary photography, rough skin.{clinical_description}"
)
negative_prompt = (
    "deformed iris, deformed pupils, semi-realistic, cgi, 3d,render, sketch, cartoon, drawing, anime:1.4), "
    "text, close up, cropped, out of frame, worst quality, jpeg artifacts, ugly, duplicate, "
    " airbrushed, retouched, smooth skin, perfect skin, makeup, foundation, instagram, glamour shot,ultra-HD,4K"
    " studio lighting, soft focus, plastic, doll, 3d render, beauty filter, symmetrical face,childish, idealized"
)
final_photo = pipe(
    prompt=text_prompt,
    negative_prompt=negative_prompt,   # Crucial for killing the beauty prior
    image=inverted_sketch, 
    controlnet_conditioning_scale=1.0, 
    guidance_scale=5.0,              
    num_inference_steps=30,          
).images[0]

final_photo = final_photo.resize(actual_size)
final_photo.save(f"media/{path.split('.',1)[0]}_generated_realistic_unflattering.jpg")

sketch_image = load_image(f"test_dataset/test_sketches/{path}").convert("RGB")
original_image = load_image(f"test_dataset/test_photos/{path}").convert("RGB")
result = Image.new('RGB', (sketch_image.width*3, sketch_image.height))

result.paste(sketch_image, (0, 0))
result.paste(original_image, (sketch_image.width, 0))
result.paste(final_photo, (sketch_image.width * 2, 0))

result.show("Sketch,Photo and Inferred Images")
result.save(f"results/{path.split('.',1)[0]}_output.jpg")
