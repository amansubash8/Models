from inference_sdk import InferenceHTTPClient
from paddleocr import PaddleOCR
import cv2
import time
import numpy as np

FRAME_SKIP = 20         # Process every 2nd frame for speed
DETECTION_THRESHOLD = 0.7  # Only accept plate detections above this confidence

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="eH8jjiNqGKLLNTHW26Zl"
)

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # For rotated/angled plates

video_path = "/Users/amansubash/Downloads/large.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video at {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = '/Users/amansubash/Downloads/traffic_annotated_paddleocr.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def get_sharpness(image):
    # Use variance of Laplacian as a sharpness metric
    return cv2.Laplacian(image, cv2.CV_64F).var()

def iou(boxA, boxB):
    # Compute intersection over union for two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

plate_tracks = {}  # {plate_id: {'max_area': 0, 'frame': None, 'coords': None, 'sharpness': 0, 'frame_num': 0}}
plate_id_counter = 0

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"End of video or cannot read frame at frame {frame_num}.")
        break

    frame_num += 1
    if frame_num % FRAME_SKIP != 0:
        out.write(frame)
        cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    print(f"Processing frame {frame_num}")

    temp_path = "/tmp/temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    result = CLIENT.infer(temp_path, model_id="licenceplatedetector/1")
    print(f"Frame {frame_num} result:", result)

    if 'predictions' not in result or not result['predictions']:
        out.write(frame)
        continue

    detected_boxes = []
    for prediction in result['predictions']:
        # Only accept high-confidence detections
        if prediction.get('confidence', 0.9) < DETECTION_THRESHOLD:
            continue
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)
        detected_boxes.append((x1, y1, x2, y2))

    for box in detected_boxes:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        sharpness = get_sharpness(crop)
        matched_id = None
        for pid, track in plate_tracks.items():
            if iou(track['coords'], box) > 0.5:
                matched_id = pid
                break
        if matched_id is None:
            matched_id = plate_id_counter
            plate_tracks[matched_id] = {
                'max_area': 0,
                'frame': None,
                'coords': box,
                'sharpness': 0,
                'frame_num': frame_num
            }
            plate_id_counter += 1
        if area > plate_tracks[matched_id]['max_area'] and sharpness > plate_tracks[matched_id]['sharpness']:
            plate_tracks[matched_id].update({
                'max_area': area,
                'frame': crop.copy(),
                'coords': box,
                'sharpness': sharpness,
                'frame_num': frame_num
            })

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Running OCR on best crops...")
final_results = []
for plate_id, track in plate_tracks.items():
    crop = track['frame']
    if crop is not None and crop.shape[0] > 10 and crop.shape[1] > 10:
        crop_zoomed = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        ocr_result = ocr.ocr(crop_zoomed, cls=True)
        if ocr_result and ocr_result[0]:
            best_text, conf = ocr_result[0][0][1][0], ocr_result[0][0][1][1]
            if conf >= 0.90:
                print(f"Plate {plate_id}: {best_text} (confidence: {conf})")
                final_results.append((plate_id, best_text, conf, track['coords'], track['frame_num']))
            else:
                print(f"Plate {plate_id}: Low confidence ({conf}), skipping.")
        else:
            print(f"Plate {plate_id}: No text detected.")
    else:
        print(f"Plate {plate_id}: Crop too small or empty.")

print(f"✅ Saved annotated video to {output_path}")