# import os
# import subprocess

# files = os.listdir('videos')
# for file in files:
#     tutorial_number = file.split(' -')[0].split('#')[1]
#     file_name = file.split(' v')[0]
#     print(tutorial_number, file_name)
#     subprocess.run(['ffmpeg', '-i', f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])



import os
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")

for file in files:
    if not file.lower().endswith((".mp4", ".mkv", ".webm", ".mov")):
        continue

    try:
        tutorial_number = file.split(" -")[0].split("#")[1]
        file_name = file.split(" v")[0]

        input_path = os.path.join("videos", file)
        output_name = f"{tutorial_number}_{file_name}.mp3"
        output_path = os.path.join("audios", output_name)

        print(f"🎧 Extracting: {tutorial_number} | {file_name}")

        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-t", "30", "-vn", "-acodec", "libmp3lame", output_path],
            check=True
        )

    except Exception as e:
        print(f"⚠️ Skipping file: {file} → {e}")
