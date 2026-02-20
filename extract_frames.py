import cv2
import os

video_folder = "videos"
frame_output = "frames"

os.makedirs(frame_output, exist_ok=True)

for video in os.listdir(video_folder):

    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 2)   # every 2 seconds

    frame_count = 0
    saved_count = 0

    video_name = video.rsplit(".", 1)[0]
    video_frame_folder = os.path.join(frame_output, video_name)
    os.makedirs(video_frame_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            timestamp = frame_count / fps
            filename = f"{video_frame_folder}/frame_{saved_count}_{timestamp:.2f}.jpg"
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

print("Frame extraction completed.")
