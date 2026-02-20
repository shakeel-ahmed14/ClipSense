import os
import math
import json

n = 5

for filename in os.listdir('jsons'):
    if filename.endswith('.json'):
        file_path = os.path.join('jsons', filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            new_chunk = []
            num_chunks = len(data['chunks'])
            num_groups = math.ceil(num_chunks / n)
        
        
        for i in range(num_groups):
            start_index = i * n
            end_index = min((i + 1) * n, num_chunks)

            chunk_group = data['chunks'][start_index:end_index]
            new_chunk.append({
                "number": data['chunks'][0]['number'],
                "title": chunk_group[0]['title'],
                "start": chunk_group[0]['start'],
                "end": chunk_group[-1]['end'],
                "text": " ".join([chunk['text'] for chunk in chunk_group])
            })
        
        os.makedirs('merged_jsons', exist_ok=True)
        with open(os.path.join('merged_jsons', filename), 'w', encoding='utf-8') as json_file:
            json.dump({"chunks": new_chunk, "text": data['text']}, json_file, indent=4)