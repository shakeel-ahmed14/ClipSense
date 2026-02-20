import json
import joblib
from sentence_transformers import SentenceTransformer

# Load BGE-M3 model
model = SentenceTransformer("BAAI/bge-m3")

# Load captions
with open("captions.json", "r") as f:
    captions = json.load(f)

visual_embeddings = []

for item in captions:
    caption_text = item["caption"]

    embedding = model.encode(
        caption_text,
        normalize_embeddings=True
    )

    visual_embeddings.append({
        "video_folder": item["video_folder"],
        "frame": item["frame"],
        "caption": caption_text,
        "embedding": embedding
    })

print("Total caption embeddings:", len(visual_embeddings))

joblib.dump(visual_embeddings, "visual_embeddings.joblib")
