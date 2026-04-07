import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# 1. The Custom Dataset Loader
class SketchFaceIDDataset(Dataset):
    def __init__(self, sketch_dir, embed_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.embed_dir = embed_dir
        self.transform = transform
        self.filenames = []
        for f in os.listdir(sketch_dir):
            if f.endswith(('.jpg', '.png')):
                base_name = os.path.splitext(f)[0]
                
                embed_base_name = base_name.replace("sketches", "photos")
                embed_path = os.path.join(embed_dir, f"{embed_base_name}.pt")
                if os.path.exists(embed_path):
                    self.filenames.append(f)
                else:
                    print(f"Skipping {f}: No matching embedding found at {embed_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]
        
        # Translate the sketch filename to the photo filename
        embed_base_name = base_name.replace("sketches", "photos")
        
        sketch_path = os.path.join(self.sketch_dir, filename)
        sketch = Image.open(sketch_path).convert('RGB')
        if self.transform:
            sketch = self.transform(sketch)
        embed_path = os.path.join(self.embed_dir, f"{embed_base_name}.pt")
        target_embedding = torch.load(embed_path)
        return sketch, target_embedding
# 2. Aggressive Data Augmentation (Prevents memorizing the 2.6k sketches)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet expects 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Ignore scan quality
    transforms.RandomRotation(5),  # Slight tilts
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet standard
])

dataset = SketchFaceIDDataset("dataset/trainA/", "dataset/embeddings/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

class SketchToFaceIDEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 256)
        )

    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)

model = SketchToFaceIDEncoder().to(device)
for param in model.parameters():
    param.requires_grad = True

base_params = [p for name, p in model.named_parameters() if 'fc' not in name]
fc_params = model.backbone.fc.parameters()

optimizer = optim.Adam([
    {'params': base_params, 'lr': 1e-5}, 
    {'params': fc_params, 'lr': 1e-4}    
], weight_decay=1e-4) 
criterion = nn.MSELoss()
model.train()
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    
    for sketches, target_embeddings in dataloader:
        sketches = sketches.to(device)
        target_embeddings = target_embeddings.to(device)
        
        # The Student guesses based on the sketch
        predicted_embeddings = model(sketches)
        target_ones = torch.ones(sketches.size(0)).to(device)
        # We want the Cosine Similarity to be exactly 1.0
        
        loss = criterion(predicted_embeddings, target_embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "sketch_faceid_encoder_v4.pth")
print("Training Complete!")
