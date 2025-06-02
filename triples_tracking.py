import cv2
from inference_sdk import InferenceHTTPClient
import time
import numpy as np

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="eH8jjiNqGKLLNTHW26Zl"
)

# Model configuration
TRIPLE_RIDING_MODEL = "traffic-violation-2/2"
TRIPLE_RIDING_CONFIDENCE_THRESHOLD = 0.4

# Video configuration
video_path = "/Users/amansubash/Downloads/triple.mp4"
output_path = "/Users/amansubash/Downloads/output_triple_tracking.mp4"

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

# Initialize tracking variables
trackers = {}
track_id = 0
frame_id = 0

# Custom tracker class
class CustomTracker:
    def __init__(self, frame, bbox, confidence):
        self.tracker = cv2.TrackerKCF_create()
        self.bbox = bbox
        x, y, w, h = bbox
        self.confidence = confidence
        self.lost_count = 0
        self.history = [(x, y)]  # Track center points
        self.tracker.init(frame, bbox)
    
    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            self.bbox = (x, y, w, h)
            self.history.append((x + w//2, y + h//2))  # Store center point
            if len(self.history) > 30:  # Keep last 30 points
                self.history.pop(0)
        return success, bbox

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ Finished processing video.")
            break

        # First, update existing trackers
        trackers_to_remove = []
        active_trackers = {}

        try:
            # Update and draw existing trackers
            for tid, tracker in trackers.items():
                try:
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        
                        # Validate tracker output
                        if (
                            x >= 0 and y >= 0 and
                            x + w < width and y + h < height and
                            w > 20 and h > 20  # Minimum size check
                        ):
                            # Draw tracking box in purple
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                            
                            # Draw tracking path
                            points = tracker.history
                            if len(points) >= 2:
                                for i in range(1, len(points)):
                                    pt1 = points[i - 1]
                                    pt2 = points[i]
                                    cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
                            
                            # Draw ID and confidence
                            label = f"Triple Riding #{tid} ({tracker.confidence:.2f})"
                            (text_w, text_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                            )
                            y_text = max(y - 10, text_h + 10)
                            
                            # Draw label background
                            cv2.rectangle(
                                frame,
                                (x, y_text - text_h - 4),
                                (x + text_w, y_text + 4),
                                (255, 0, 255),
                                -1
                            )
                            cv2.putText(
                                frame,
                                label,
                                (x, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                3
                            )
                            
                            active_trackers[tid] = tracker
                        else:
                            tracker.lost_count += 1
                    else:
                        tracker.lost_count += 1
                    
                    # Keep tracker if not lost for too long (30 frames)
                    if tracker.lost_count < 30:
                        active_trackers[tid] = tracker
                    else:
                        trackers_to_remove.append(tid)
                        
                except Exception as e:
                    print(f"⚠️ Tracker #{tid} error: {str(e)}")
                    trackers_to_remove.append(tid)
            
            # Remove failed trackers
            for tid in trackers_to_remove:
                trackers.pop(tid)
            
            # Update trackers dictionary
            trackers = active_trackers

            # Run detection every 5 frames or when no active trackers
            if frame_id % 5 == 0 or len(trackers) == 0:
                result = CLIENT.infer(frame, model_id=TRIPLE_RIDING_MODEL)
                
                for pred in result.get('predictions', []):
                    if pred['class'] == 'more-than-2-person-on-2-wheeler' and pred['confidence'] >= TRIPLE_RIDING_CONFIDENCE_THRESHOLD:
                        try:
                            x1 = int(pred['x'] - pred['width'] / 2)
                            y1 = int(pred['y'] - pred['height'] / 2)
                            x2 = int(pred['x'] + pred['width'] / 2)
                            y2 = int(pred['y'] + pred['height'] / 2)
                            w = x2 - x1
                            h = y2 - y1
                            confidence = pred['confidence']
                            
                            # Add padding for better tracking
                            pad_x = int(0.1 * w)
                            pad_y = int(0.1 * h)
                            det_box = (
                                max(0, x1 - pad_x),
                                max(0, y1 - pad_y),
                                min(width - 1, x2 + pad_x) - max(0, x1 - pad_x),
                                min(height - 1, y2 + pad_y) - max(0, y1 - pad_y)
                            )
                            
                            # Check for overlap with existing trackers
                            new_detection = True
                            for tracker in trackers.values():
                                tx, ty, tw, th = tracker.bbox
                                if (
                                    x1 < tx + tw and x2 > tx and
                                    y1 < ty + th and y2 > ty
                                ):
                                    new_detection = False
                                    break
                            
                            if new_detection:
                                # Create new tracker
                                tracker = CustomTracker(frame, det_box, confidence)
                                trackers[track_id] = tracker
                                print(f"⚠️ New triple riding detected! ID #{track_id} (Confidence: {confidence:.2f})")
                                track_id += 1
                            
                            # Draw detection box in green
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            label = f"Triple Riding ({confidence:.2f})"
                            (text_w, text_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                            )
                            y_text = max(y1 - 10, text_h + 10)
                            
                            # Draw detection label
                            cv2.rectangle(
                                frame,
                                (x1, y_text - text_h - 4),
                                (x1 + text_w, y_text + 4),
                                (0, 255, 0),
                                -1
                            )
                            cv2.putText(
                                frame,
                                label,
                                (x1, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                3
                            )
                        except Exception as e:
                            print(f"⚠️ Detection processing error: {str(e)}")
                
        except Exception as e:
            print(f"⚠️ Main loop error at frame {frame_id}: {str(e)}")

        # Add debug information
        debug_text = f"Frame: {frame_id} | Active Trackers: {len(trackers)}"
        cv2.putText(
            frame,
            debug_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Write and display frame
        out.write(frame)
        cv2.imshow("Triple Riding Detection + Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 Interrupted by user.")
            break

        frame_id += 1

except Exception as e:
    print(f"❌ Critical error: {str(e)}")

finally:
    cap.release()
    out.release()
    print(f"📁 Output saved to: {output_path}")
    time.sleep(1)
    cv2.destroyAllWindows()