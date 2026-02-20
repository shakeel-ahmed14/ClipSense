import os
from PIL import Image
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cpu"
model.to(device)

captions = []

frames_root = "frames"

for root, dirs, files in os.walk(frames_root):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(root, file)
            
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(**inputs)

            caption = processor.decode(out[0], skip_special_tokens=True)

            captions.append({
                "video_folder": os.path.basename(root),
                "frame": file,
                "caption": caption
            })

print("Captions generated:", len(captions))

with open("captions.json", "w") as f:
    json.dump(captions, f, indent=4)
