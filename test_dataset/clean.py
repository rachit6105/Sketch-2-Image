import re
from pathlib import Path

INPUT_DIR = Path("./clinical_desc")
OUTPUT_DIR = Path("./LORA_clinical_desc")
TRIGGER_TOKEN = "forensic_mugshot_style"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    text = text.replace("original_mugshot_photo", TRIGGER_TOKEN)

    replacements = {
        "Approximately ": "",
        " years old": "",
        " ethnicity": "",
        " skin tone": " skin",
        " face shape": " face",
        " softly rounded jawline": " rounded jawline",
        " neutral color": "",
        " small pores visible": " visible pores",
        " no noticeable wrinkles": " no wrinkles",
        "Not wearing glasses or any eyewear.": "no glasses.",
        "No scars or facial hair observed.": "no scars, no facial hair.",
        "barely visible": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    label_map = {
        "Hair:": "hair",
        "Eyebrows:": "eyebrows",
        "Eyes:": "eyes",
        "Nose:": "nose",
        "Lips:": "lips",
        "Skin:": "skin",
    }
    for old, new in label_map.items():
        text = text.replace(old, new)

    text = text.replace(". ", ", ")
    text = text.replace(".", "")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")

    return text

for file in INPUT_DIR.glob("*.txt"):
    original = file.read_text(encoding="utf-8")
    cleaned = clean_caption(original)
    (OUTPUT_DIR / file.name).write_text(cleaned, encoding="utf-8")

print(f"Saved cleaned captions to: {OUTPUT_DIR}")