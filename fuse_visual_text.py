import joblib
import pandas as pd
import numpy as np

# Load transcript embeddings
df = joblib.load("embeddings.joblib")

# Load visual embeddings
visual_data = joblib.load("visual_embeddings.joblib")

# Convert visual_data to dataframe
visual_df = pd.DataFrame(visual_data)

# Extract frame timestamp from filename
visual_df["timestamp"] = visual_df["frame"].str.extract(r'(\d+)').astype(int)

# assuming 1 frame per second
# if fps different, adjust accordingly

def get_visual_for_chunk(chunk):
    start = int(chunk["start"])
    end = int(chunk["end"])
    video = chunk["title"]

    matches = visual_df[
        (visual_df["video_folder"] == video) &
        (visual_df["timestamp"] >= start) &
        (visual_df["timestamp"] <= end)
    ]

    return matches["caption"].tolist(), matches["embedding"].tolist()


# Attach visual info to transcript chunks
df["visual_captions"] = None
df["visual_embeddings"] = None

for idx, row in df.iterrows():

    captions, embeddings = get_visual_for_chunk(row)

    df.at[idx, "visual_captions"] = captions
    df.at[idx, "visual_embeddings"] = embeddings


joblib.dump(df, "multimodal_embeddings.joblib")

print("Multimodal fusion complete.")
