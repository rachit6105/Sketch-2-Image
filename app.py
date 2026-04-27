import gradio as gr
import torch
import PIL.ImageOps
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverSDEScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file, save_file
from PIL import Image
import os

# ==========================================
# 1. INITIALIZE MODELS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.float16
print(f"Loading models to {device} with {dtype}...")

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype)

longclip_id = "zer0int/LongCLIP-L-Diffusers"
text_encoder = CLIPTextModel.from_pretrained(longclip_id, torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(longclip_id, model_max_length=248)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    torch_dtype=dtype,
    safety_checker=None,
).to(device)

pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

# ==========================================
# 2. LOAD ALL LoRAs
# ==========================================

# --- Forensic LoRA (Kohya) ---
FORENSIC_LORA_LOADED = False
try:
    pipe.load_lora_weights("modelweights/forensic_kohya.safetensors", adapter_name="forensic")
    FORENSIC_LORA_LOADED = True
    print("Forensic LoRA loaded!")
except Exception as e:
    print(f"WARNING: Forensic LoRA not loaded. Error: {e}")

# --- Skin Tone Slider LoRA ---
SKIN_LORA_LOADED = False
try:
    pipe.load_lora_weights("modelweights/skintone2.safetensors", adapter_name="skin_tone")
    SKIN_LORA_LOADED = True
    print("Skin tone LoRA loaded!")
except Exception as e:
    print(f"WARNING: Skin tone LoRA not loaded. Error: {e}")

# --- Age Slider LoRA ---
AGE_LORA_LOADED = False
try:
    pipe.load_lora_weights("modelweights/age.safetensors", adapter_name="age")
    AGE_LORA_LOADED = True
    print("Age LoRA loaded!")
except Exception as e:
    print(f"WARNING: Age LoRA not loaded. Error: {e}")

print(f"\nLoRA Status — Forensic: {FORENSIC_LORA_LOADED} | Skin: {SKIN_LORA_LOADED} | Age: {AGE_LORA_LOADED}")


# ==========================================
# 3. GENERATION FUNCTION
# ==========================================
def generate_composite(sketch_image, description, forensic_weight, skin_tone_weight, age_weight):
    if sketch_image is None:
        return None

    print(f"Generating on {device} | Forensic: {forensic_weight} | Skin: {skin_tone_weight} | Age: {age_weight}")

    # Save original size to restore after generation
    original_size = sketch_image.size

    sketch_resized = sketch_image.convert("RGB").resize((512, 512))
    inverted_sketch = PIL.ImageOps.invert(sketch_resized)

    text_prompt = (
        f"{description}. "
        f"Raw amateur photo, DMV camera mugshot, (extreme skin texture:1.4), (visible pores:1.3), "
        f"hyper-detailed unairbrushed skin, harsh flash lighting, documentary photography."
    )
    negative_prompt = (
        "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4, "
        "text, close up, cropped, out of frame, worst quality, jpeg artifacts, "
        "airbrushed, retouched, makeup, foundation, instagram, glamour shot, ultra-HD, 4K, "
        "studio lighting, soft focus, plastic, doll, 3d render, beauty filter"
    )

    # Build active adapter list dynamically based on what loaded
    adapter_names = []
    adapter_weights = []

    if FORENSIC_LORA_LOADED:
        adapter_names.append("forensic")
        adapter_weights.append(forensic_weight)
    if SKIN_LORA_LOADED:
        adapter_names.append("skin_tone")
        adapter_weights.append(skin_tone_weight)
    if AGE_LORA_LOADED:
        adapter_names.append("age")
        adapter_weights.append(age_weight)

    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    steps = 5 if device == "cpu" else 25

    final_photo = pipe(
        prompt=text_prompt,
        negative_prompt=negative_prompt,
        image=inverted_sketch,
        controlnet_conditioning_scale=0.85,
        guidance_scale=8.0,
        num_inference_steps=steps,
    ).images[0]

    # Restore original size
    final_photo = final_photo.resize(original_size)

    if device == "cuda":
        torch.cuda.empty_cache()

    return final_photo


# ==========================================
# 4. GRADIO UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        Upload a suspect sketch, provide a clinical description, and adjust the sliders to refine the output.
        """
    )

    with gr.Row():
        # LEFT — Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 1. Inputs")
            input_image = gr.Image(type="pil", label="Upload Police Sketch (Lineart)")
            input_text = gr.Textbox(
                lines=5,
                label="Clinical Description",
                placeholder="e.g., Female, late 20s, Mediterranean descent, oval face, dark red hair..."
            )

            gr.Markdown("### 2. LoRA Sliders")
            slider_forensic = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.6, step=0.1,
                label="Forensic Style Strength",
                interactive=FORENSIC_LORA_LOADED
            )
            slider_skin = gr.Slider(
                minimum=-5.0, maximum=5.0, value=0.0, step=0.1,
                label="Skin Tone (Darker ➔ Lighter)",
                interactive=SKIN_LORA_LOADED
            )
            slider_age = gr.Slider(
                minimum=-5.0, maximum=5.0, value=0.0, step=0.1,
                label="Age (Younger ➔ Older)",
                interactive=AGE_LORA_LOADED
            )

            # Show which LoRAs are active
            lora_status = []
            if FORENSIC_LORA_LOADED: lora_status.append(" Forensic")
            if SKIN_LORA_LOADED: lora_status.append(" Skin Tone")
            if AGE_LORA_LOADED: lora_status.append(" Age")
            if not lora_status: lora_status.append(" No LoRAs loaded")
            gr.Markdown(f"**Active LoRAs:** {' | '.join(lora_status)}")

            generate_btn = gr.Button("Generate Realistic Composite", variant="primary", size="lg")

        # RIGHT — Output
        with gr.Column(scale=1):
            gr.Markdown("### 3. Generated Result")
            output_image = gr.Image(label="Final Photorealistic Output", type="pil", interactive=False)

    generate_btn.click(
        fn=generate_composite,
        inputs=[input_image, input_text, slider_forensic, slider_skin, slider_age],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)