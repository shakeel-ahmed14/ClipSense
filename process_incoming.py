import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42
            }
        }
    )
    return r.json()
    # return response

df = joblib.load('multimodal_embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
text_similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [question_embedding]
).flatten()

visual_similarities = []

for visual_emb_list in df['visual_embeddings']:

    if visual_emb_list is None or len(visual_emb_list) == 0:
        visual_similarities.append(0)

    else:
        try:
            visual_emb_array = np.array(visual_emb_list)

            sim = cosine_similarity(
                visual_emb_array,
                [question_embedding]
            ).max()

            visual_similarities.append(sim)

        except:
            visual_similarities.append(0)


visual_similarities = np.array(visual_similarities)

# Combine both
final_similarities = 0.7 * text_similarities + 0.3 * visual_similarities


top_results = 5
max_indx = final_similarities.argsort()[::-1][0:top_results]

new_df = df.loc[max_indx].reset_index(drop=True)




prompt = f'''
You are an AI assistant helping users find events in cooking tutorial videos.

Below are relevant video segments containing:

- video title
- timestamps
- transcript
- visual descriptions of frames

VIDEO SEGMENTS:
{new_df[['title','number','start','end','text','visual_captions']].to_json(orient='records', indent=2)}

--------------------------------

USER QUESTION:
"{incoming_query}"

INSTRUCTIONS:
- Identify the most relevant video segment
- Use transcript AND visual descriptions
- Mention exact timestamp range
- Mention video title
- Explain clearly and naturally
- If not found, say "This event was not found in the video"
- ONLY use the provided video data
- DO NOT guess or assume anything
- If exact event not present, say "This event was not found in the video"
- DO NOT invent timestamps
'''



with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])