import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1
import random

device ="cpu"
print(f"Using device: {device}")

class SketchToFaceIDEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        # UPGRADED: A non-linear, deep translator
        self.translator = nn.Sequential(
                    nn.Dropout(p=0.25),
                    nn.Linear(512, 512)
        )

    def forward(self, x):
        face_features = self.backbone(x)
        faceid_math = self.translator(face_features)
        return F.normalize(faceid_math, p=2, dim=1)

print("Loading Custom Sketch Translator...")
model = SketchToFaceIDEncoder().to(device)
model.load_state_dict(torch.load("best_vggface_translator.pth", map_location=device))
model.eval() # CRITICAL: Turns off Dropout for testing!

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Loading InsightFace...")
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_sketch_embedding(image_path):
    sketch = Image.open(image_path).convert('RGB')
    sketch_tensor = inference_transform(sketch).unsqueeze(0).to(device)
    
    with torch.no_grad(): # Don't calculate gradients during testing
        embedding = model(sketch_tensor)
    return embedding[0] 

def get_photo_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError(f"InsightFace could not find a face in {image_path}")
    
    return torch.from_numpy(faces[0].normed_embedding).to(device)

def check_match(emb1, emb2, label):
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    match_status = "MATCH" if similarity > 0.40 else "NO MATCH"
    print(f"{label:.<40} Score: {similarity:+.4f} [{match_status}]")

if __name__ == "__main__":
    n1 = random.randint(1, 338)
    n2 = random.randint(1, 338)
    sketch_A_path = f"WildSketch_test_photos_3.jpg"
    photo_A_path  = f"WildSketch_test_photos_3.jpg"

    sketch_B_path = f"test_dataset/test_photos/{n2}.jpg"
    photo_B_path  = f"test_dataset/test_photos/{n2}.jpg"
    
    print("\nExtracting embeddings...")
    try:
        sketch_A_emb = get_sketch_embedding(sketch_A_path)
        photo_A_emb  = get_photo_embedding(photo_A_path)
        
        sketch_B_emb = get_sketch_embedding(sketch_B_path)
        photo_B_emb  = get_photo_embedding(photo_B_path)
        
        print("\n--- BIOMETRIC VERIFICATION RESULTS ---")
        
        # 1. True Positives (The Ultimate Test)
        check_match(sketch_A_emb, photo_A_emb, "Sketch A vs Photo A (Should Match)")
        check_match(sketch_B_emb, photo_B_emb, "Sketch B vs Photo B (Should Match)")
        
        print("-" * 40)
        
        # 2. True Negatives (Ensuring it didn't collapse into guessing 'average')
        check_match(sketch_A_emb, photo_B_emb, "Sketch A vs Photo B (Should NOT Match)")
        check_match(sketch_B_emb, photo_A_emb, "Sketch B vs Photo A (Should NOT Match)")
        
        # 3. Same Domain Checks
        check_match(sketch_A_emb, sketch_B_emb, "Sketch A vs Sketch B (Should NOT Match)")
        check_match(photo_A_emb, photo_B_emb, "Photo A vs Photo B (Should NOT Match)")
        
    except Exception as e:
        print(f"\nError occurred: {e}")