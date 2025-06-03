import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)

# Roboflow model client setup with primary API endpoint
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="eH8jjiNqGKLLNTHW26Zl"
)

# Model and tracking parameters
CROWD_MODEL = "crowd-counting-8hvzc/1"
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP = 30  # Process every 2nd frame

# Video configuration
video_path = "/Users/amansubash/Downloads/1temple.mov"
output_path = "/Users/amansubash/Downloads/crowd_counting_output.mp4"

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30,
                  n_init=3,
                  nms_max_overlap=0.7,
                  max_cosine_distance=0.3,
                  nn_budget=None,
                  override_track_class=None,
                  embedder="mobilenet",
                  half=True,
                  bgr=True)

unique_ids = set()

def convert_predictions(predictions, frame):
    """Convert Roboflow predictions to DeepSORT format"""
    if not predictions:
        return []
    
    detections = []
    for pred in predictions:
        if pred["confidence"] >= CONFIDENCE_THRESHOLD:
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            conf = pred["confidence"]
            
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # DeepSORT format: [left, top, width, height, confidence]
            detections.append(([x1, y1, w, h], conf, "person"))
    
    return detections

def draw_info_overlay(frame, info_dict):
    """Draw a semi-transparent overlay with tracking information"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for the top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    
    # Create gradient effect for better visibility
    gradient = np.linspace(0.8, 0.2, 80)
    for i in range(80):
        cv2.line(overlay, (0, i), (width, i), (0, 0, 0), 1)
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add text with enhanced styling
    font = cv2.FONT_HERSHEY_DUPLEX
    total_count = info_dict['total_unique']
    
    # Draw total count with large font and green color
    count_text = str(total_count)
    text_size = cv2.getTextSize(count_text, font, 2.5, 3)[0]
    text_x = width - text_size[0] - 20  # 20 pixels padding from right
    
    # Draw total count number (large)
    cv2.putText(frame, count_text, (text_x, 55), 
                font, 2.5, (0, 255, 0), 3)    # Green for total count
    
    # Draw label (smaller)
    cv2.putText(frame, "Total People:", (20, 55), 
                font, 1.2, (255, 255, 255), 2)
    
    return frame

def get_detections(frame, skip_frame=False):
        if skip_frame:
            return []
            
        try:
            # Convert frame to format required by Roboflow
            _, img_encoded = cv2.imencode('.jpg', frame)
            response = CLIENT.infer(img_encoded.tobytes(), model_id="yolov8n/1")
            
            detections = []
            if response.get("predictions"):
                for pred in response["predictions"]:
                    if pred["class"] == "person" and pred["confidence"] > 0.5:
                        x1 = pred["x"] - pred["width"] / 2
                        y1 = pred["y"] - pred["height"] / 2
                        x2 = x1 + pred["width"]
                        y2 = y1 + pred["height"]
                        conf = pred["confidence"]
                        detections.append(([x1, y1, x2, y2], conf, "person"))
            return detections
            
        except requests.exceptions.RequestException as e:
            print(f"Network error during detection: {e}")
            return []
        except Exception as e:
            print(f"Error during detection: {e}")
            return []

def main():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    current_count = 0
    tracks = []
    
    try:
        print("🚀 Starting crowd counting with DeepSORT...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("✅ Finished processing video.")
                break

            process_frame = frame_id % FRAME_SKIP == 0
            if process_frame:
                try:
                    # Convert frame to RGB as Roboflow expects RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Get predictions from Roboflow
                    result = CLIENT.infer(rgb_frame, model_id=CROWD_MODEL)
                    predictions = result.get("predictions", [])
                    
                    # Convert detections to DeepSORT format
                    detections = convert_predictions(predictions, frame)
                    
                    # Update tracker
                    tracks = tracker.update_tracks(detections, frame=frame)

                    # Update current count and process tracks
                    current_count = 0
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        current_count += 1
                        track_id = track.track_id
                        if track_id not in unique_ids:
                            unique_ids.add(track_id)

                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = map(int, ltrb)

                        # Draw tracking box
                        color = (0, 255, 0) if track.age > 3 else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw ID
                        cv2.putText(frame, f"ID: {track_id}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, color, 2)

                except Exception as e:
                    print(f"⚠️ Error processing frame {frame_id}: {str(e)}")
                    continue

            # Draw info overlay on every frame
            info_dict = {
                'current_count': current_count,
                'total_unique': len(unique_ids),
                'frame': frame_id
            }
            frame = draw_info_overlay(frame, info_dict)

            # Write frame and show
            out.write(frame)
            cv2.imshow("DeepSORT Crowd Counting", frame)
            cv2.setWindowProperty("DeepSORT Crowd Counting", cv2.WND_PROP_TOPMOST, 1)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("\n📊 Final Results")
        print(f"Frames Processed: {frame_id}")
        print(f"Total Unique Persons: {len(unique_ids)}")
        print(f"📁 Output saved at: {output_path}")

if __name__ == "__main__":
    main()
