import cv2
from inference_sdk import InferenceHTTPClient
import time

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="eH8jjiNqGKLLNTHW26Zl"
)

# Model IDs
HELMET_MODEL = "wright-mechs-isg/3"
MOTORCYCLE_MODEL = "vehicles-dzogt/2"
NO_HELMET_MODEL = "rider-plate-headcls3/4"

# Configuration
video_path = "/Users/amansubash/Downloads/hhhh.mp4"
output_path = "/Users/amansubash/Downloads/output_all_models_detection.mp4"
FRAME_SKIP = 3
HELMET_CONFIDENCE_THRESHOLD = 0.6
MOTORCYCLE_CONFIDENCE_THRESHOLD = 0.4
NO_HELMET_CONFIDENCE_THRESHOLD = 0.4

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video at {video_path}")

# Output settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_num = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ Finished processing all frames.")
            break

        # Only run inference on selected frames
        if frame_num % FRAME_SKIP == 0:
            try:
                # Run all model inferences
                helmet_result = CLIENT.infer(frame, model_id=HELMET_MODEL)
                motorcycle_result = CLIENT.infer(frame, model_id=MOTORCYCLE_MODEL)
                no_helmet_result = CLIENT.infer(frame, model_id=NO_HELMET_MODEL)

                # Draw helmet detections (green)
                for pred in helmet_result.get('predictions', []):
                    if pred['class'] == 'helmet' and pred['confidence'] >= HELMET_CONFIDENCE_THRESHOLD:
                        x1 = int(pred['x'] - pred['width'] / 2)
                        y1 = int(pred['y'] - pred['height'] / 2)
                        x2 = int(pred['x'] + pred['width'] / 2)
                        y2 = int(pred['y'] + pred['height'] / 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"helmet: {pred['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw motorcycle detections (blue)
                for pred in motorcycle_result.get('predictions', []):
                    if pred['class'] == 'motorcycle' and pred['confidence'] >= MOTORCYCLE_CONFIDENCE_THRESHOLD:
                        x1 = int(pred['x'] - pred['width'] / 2)
                        y1 = int(pred['y'] - pred['height'] / 2)
                        x2 = int(pred['x'] + pred['width'] / 2)
                        y2 = int(pred['y'] + pred['height'] / 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"motorcycle: {pred['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw no-helmet detections (red)
                for pred in no_helmet_result.get('predictions', []):
                    if pred['class'] == 'no-helmet' and pred['confidence'] >= NO_HELMET_CONFIDENCE_THRESHOLD:
                        x1 = int(pred['x'] - pred['width'] / 2)
                        y1 = int(pred['y'] - pred['height'] / 2)
                        x2 = int(pred['x'] + pred['width'] / 2)
                        y2 = int(pred['y'] + pred['height'] / 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"no-helmet: {pred['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            except Exception as e:
                print(f"⚠️ Inference failed at frame {frame_num}: {e}")

        # Always write the frame (processed or not)
        out.write(frame)

        # Display the frame
        cv2.imshow("Helmet / No Helmet / Motorcycle Detection", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("🛑 Interrupted by user. Saving and exiting...")
            break

        frame_num += 1

finally:
    cap.release()
    out.release()
    print(f"✅ Output video saved to {output_path}")
    time.sleep(1)  # Ensure file is flushed

cv2.destroyAllWindows()  # Call this after the try-finally block
