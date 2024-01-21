import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'pure3.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
out = None  # Initialize the VideoWriter outside the loop

model_path = os.path.join("./runs/detect/train2/weights/best.pt")

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

class_name_dict = {
    0: 'male-external-genital',
    1: 'female-vulva',
    2: 'female-breast',
    3: 'mouth and male-external-genital'
}

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Exit the loop if there are no more frames

    H, W, _ = frame.shape

    if out is None:
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA))

    out.write(frame)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()