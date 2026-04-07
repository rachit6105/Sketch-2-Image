import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Load Your Trained Sketch Encoder
# ==========================================
class SketchToFaceIDEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50() 
        num_ftrs = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(num_ftrs, 512)
        )

    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)

model = SketchToFaceIDEncoder().to(device)

# LOAD YOUR SAVED WEIGHTS HERE (Change the name if yours is different)
model.load_state_dict(torch.load("sketch_faceid_encoder_v3.pth", map_location=device))
model.eval() # Set to evaluation mode (turns off dropout/batchnorm updates)

# The EXACT transform used in training (BUT WITHOUT the random jitter/rotation!)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. Load InsightFace (For the Photos)
# ==========================================
# Using CPU provider so you can safely run this test on the login node!
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider','CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# 3. Helper Extraction Functions
# ==========================================
def get_sketch_embedding(image_path):
    sketch = Image.open(image_path).convert('RGB')
    sketch_tensor = inference_transform(sketch).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(sketch_tensor)
    return embedding[0] # Return the 1D vector

def get_photo_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError(f"InsightFace could not find a face in {image_path}")
    # Convert numpy array to PyTorch tensor
    return torch.from_numpy(faces[0].normed_embedding).to(device)

def check_match(emb1, emb2, label):
    # Calculate Cosine Similarity
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    match_status = "MATCH" if similarity > 0.45 else "NO MATCH"
    print(f"{label:.<40} Score: {similarity:.4f} [{match_status}]")

# ==========================================
# 4. The Test Area
# ==========================================
if __name__ == "__main__":
    n1 = random.randint(1, 338)
    n2 = random.randint(1, 338)
    sketch_A_path = f"WildSketch_test_sketches_3.png"
    photo_A_path  = f"WildSketch_test_photos_3.jpg"

    sketch_B_path = f"test_sketches/{n2}.jpg"
    photo_B_path  = f"test_photos/{n2}.jpg"
    
    print("Extracting embeddings...\n")
    try:
        sketch_A_emb = get_sketch_embedding(sketch_A_path)
        photo_A_emb  = get_photo_embedding(photo_A_path)
        
        sketch_B_emb = get_sketch_embedding(sketch_B_path)
        photo_B_emb  = get_photo_embedding(photo_B_path)
        
        print("--- BIOMETRIC VERIFICATION RESULTS ---\n")
        print(f"Selected Pairs : {n1} & {n2}\n")
        # 1. The Ultimate Test (Cross-Modal Matching)
        # Does the math extracted from the sketch match the math from the real photo?
        check_match(sketch_A_emb, photo_A_emb, "Sketch A vs Photo A")
        check_match(sketch_B_emb, photo_B_emb, "Sketch B vs Photo B")
        
        # 2. Negative Control (Should NOT match)
        check_match(sketch_A_emb, photo_B_emb, "Sketch A vs Photo B")
        check_match(sketch_B_emb, photo_A_emb, "Sketch B vs Photo A")
        
        # 3. Same Domain Control
        check_match(sketch_A_emb, sketch_B_emb, "Sketch A vs Sketch B")
        check_match(photo_A_emb, photo_B_emb, "Photo A vs Photo B")
        
    except Exception as e:
        print(f"Error: {e}")