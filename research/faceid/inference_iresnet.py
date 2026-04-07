import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import random
import argparse
from encoder_iresnet import SketchToFaceEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ── Load models ────────────────────────────────────────────────────────────────

model = SketchToFaceEncoder("insightface_r50_weights.pth").to(device)
model.load_state_dict(torch.load("best_sketch_encoder.pth", map_location=device), strict=True)
model.eval()

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider','CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

inference_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ── Embedding extractors ───────────────────────────────────────────────────────

def get_sketch_embedding(image_path):
    tensor = inference_transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor)[0]

def get_photo_embedding(image_path):
    faces = app.get(cv2.imread(image_path))
    if not faces:
        raise ValueError(f"No face detected in {image_path}")
    return torch.from_numpy(faces[0].normed_embedding).to(device)

def check_match(emb1, emb2, label, threshold=0.40):
    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    print(f"  {label:.<50} {sim:+.4f}  [{'MATCH' if sim > threshold else 'NO MATCH'}]")

# ── Inference ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare sketch/photo embeddings")
    parser.add_argument("n", nargs="?", help="image index or filename prefix")
    args = parser.parse_args()
    n = str(args.n)

    sketch_A_path = f"/home/tichar/Documents/ugp_vlm/diffuse/test_dataset/test_sketches/{n}.jpg"
    photo_A_path  = f"/home/tichar/Documents/ugp_vlm/diffuse/output/{n}_generated_realistic_unflattering.jpg"

    sketch_B_path = f"/home/tichar/Documents/ugp_vlm/diffuse/test_dataset/generated_photos/{n}.jpg"
    photo_B_path  = f"/home/tichar/Documents/ugp_vlm/diffuse/test_dataset/test_photos/{n}.jpg"

    sketch_A_emb = get_sketch_embedding(sketch_A_path)
    photo_A_emb  = get_photo_embedding(photo_A_path)
    sketch_B_emb = get_sketch_embedding(sketch_B_path)
    photo_B_emb  = get_photo_embedding(photo_B_path)

    print("\n── True positives ──")
    check_match(sketch_A_emb, photo_A_emb, "Sketch A vs Photo A")
    check_match(sketch_B_emb, photo_B_emb, "Sketch B vs Photo B")

    print("\n── Cross-identity ──")
    check_match(sketch_A_emb, photo_B_emb, "Sketch A vs Photo B")
    check_match(sketch_B_emb, photo_A_emb, "Sketch B vs Photo A")

    print("\n── Same domain ──")
    check_match(sketch_A_emb, sketch_B_emb, "Sketch A vs Sketch B")
    check_match(photo_A_emb,  photo_B_emb,  "Photo A  vs Photo B")

    gap_A = F.cosine_similarity(sketch_A_emb.unsqueeze(0), photo_A_emb.unsqueeze(0)).item()
    gap_B = F.cosine_similarity(sketch_B_emb.unsqueeze(0), photo_B_emb.unsqueeze(0)).item()
    print(f"\n── Embedding gap (higher=better) ──")
    print(f"  A: {gap_A:.4f}  B: {gap_B:.4f}  Avg: {(gap_A+gap_B)/2:.4f}")
