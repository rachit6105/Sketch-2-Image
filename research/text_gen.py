import subprocess
import json
import ast
import os

prompt = (
    "Analyze this face and provide a highly accurate description.Identify the visible features of the face in the image. "
    "You MUST output ONLY a valid JSON object. Do not include any conversational text, "
    "markdown formatting, or explanations. Use exactly these keys: "
    "'eye_color', 'race', 'skin_tone', 'gender', 'hair_type', 'hair_color', "
    "'lip_shape', 'face_shape', 'distinguishing_features'."
)

image_path = r"/home/tichar/Documents/ugp_vlm/diffuse/00084fb011d_931230.jpg"

# Resolve absolute paths BEFORE we change the working directory in subprocess
absolute_image_path = os.path.abspath(image_path)
absolute_model_path = os.path.abspath("./checkpoints/FaceLLaVA")

command = [
    "python", "inference.py",
    "--model_path", absolute_model_path,
    "--file_path", absolute_image_path,
    "--prompt", prompt
]

print(f"Analyzing {absolute_image_path}...\n")
raw_output = "Failed to get output from Face-LLaVA." 
try:
    # Run the command from inside the Face-LLaVA directory
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        cwd="./Face-LLaVA"
    )
    
    # Grab the standard output from the terminal
    raw_output = result.stdout.strip()
    
    # Clean up any potential markdown formatting the model might hallucinate
    if "```json" in raw_output:
        raw_output = raw_output.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_output:
        raw_output = raw_output.split("```")[1].strip()
        
    # Parse the output into a Python dictionary
    try:
        parsed_json = json.loads(raw_output)
        print("--- Successfully Extracted JSON ---")
        print(json.dumps(parsed_json, indent=4))
    except json.JSONDecodeError:
        # Fallback in case the model used single quotes instead of double quotes
        parsed_json = ast.literal_eval(raw_output)
        print("--- Successfully Extracted Dictionary ---")
        print(json.dumps(parsed_json, indent=4))

except subprocess.CalledProcessError as e:
    print(f"Error running Face-LLaVA inference:\n{e.stderr}")
except Exception as e:
    print(f"Failed to parse output: {e}\nRaw Output:\n{raw_output}")