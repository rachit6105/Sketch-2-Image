import os
import cv2
import torch
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Directories
photo_dir = "dataset/real_photos/"
output_dir = "dataset/embeddings/"
os.makedirs(output_dir, exist_ok=True)

print("Starting extraction...")

for filename in os.listdir(photo_dir):
    if filename.endswith((".jpg", ".png")):
        img_path = os.path.join(photo_dir, filename)
        img = cv2.imread(img_path)
        faces = app.get(img)
        
        if len(faces) > 0:
            embedding = torch.from_numpy(faces[0].normed_embedding)
            
            base_name = os.path.splitext(filename)[0]
            torch.save(embedding, os.path.join(output_dir, f"{base_name}.pt"))
        else:
            print(f"WARNING: No face detected in {filename}. Please remove this pair.")

print("Extraction complete! You can now delete InsightFace from your GPU memory.")