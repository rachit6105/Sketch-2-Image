import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import ast

print("Loading Qwen2-VL model (this may take a minute on the first run to download weights)...")

# 1. Load the model and processor
# Using bfloat16 and device_map="auto" puts it on your RTX 3070 perfectly
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 2. Prepare the prompt and image
image_path = "297.jpg"
prompt_text = (
    "Analyze this face and extract the following facial attributes. "
    "You MUST output ONLY a valid JSON object without any markdown formatting, conversational filler, or explanations. "
    "Use exactly these keys: 'eye_color', 'race', 'skin_tone', 'gender', 'hair_type', "
    "'hair_color', 'lip_shape', 'face_shape','moles_present','moles_location_if_present','freckles', 'distinguishing_features'."
)

# Qwen expects messages in a specific chat template format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }
]

# 3. Process inputs for the model
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda") # Send the data to your GPU

# 4. Generate the description
print(f"\nAnalyzing {image_path}...")
generated_ids = model.generate(**inputs, max_new_tokens=200)

# Trim the prompt out of the output so we only get the new text
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# 5. Clean and parse the JSON output
raw_output = output_text.strip()
if "```json" in raw_output:
    raw_output = raw_output.split("```json")[1].split("```")[0].strip()
elif "```" in raw_output:
    raw_output = raw_output.split("```")[1].strip()

try:
    parsed_json = json.loads(raw_output)
    print("\n--- Successfully Extracted JSON ---")
    print(json.dumps(parsed_json, indent=4))
except json.JSONDecodeError:
    try:
        parsed_json = ast.literal_eval(raw_output)
        print("\n--- Successfully Extracted Dictionary ---")
        print(json.dumps(parsed_json, indent=4))
    except:
        print("\nCould not cleanly parse JSON. Here is the raw output:")
        print(raw_output)
