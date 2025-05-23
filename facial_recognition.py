import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
import json
import requests
import logging
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReferenceFace:
    def __init__(self, name: str, embedding: np.ndarray):
        self.name = name
        self.embedding = embedding

class FaceComparer:
    def __init__(self, threshold: float = 0.4):
        """
        Initialize the FaceComparer with a detection threshold
        Args:
            threshold (float): Similarity threshold for face matching (0.0 to 1.0)
        """
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = threshold
        self.reference_faces: List[ReferenceFace] = []
        self.active_detections = {}  # To track currently active faces
        self.last_frames = {}  # To store the last frame for each person

    def frame_to_base64(self, frame, person_name: str, event_type: str, bbox: List[int] = None) -> str:
        """
        Convert frame to base64 string with optional bounding box
        Args:
            frame: The frame to convert
            person_name: Name of the detected person
            event_type: Type of event (entry/exit)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        Returns:
            str: Base64 encoded image string
        """
        try:
            # Create a copy of the frame to draw on
            frame_with_box = frame.copy()
            
            # If bbox is provided, draw it and the person's name
            if bbox is not None:
                # Draw bounding box in green
                cv2.rectangle(frame_with_box, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Add person's name above the box
                cv2.putText(
                    frame_with_box,
                    f"{person_name}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_box)
            # Convert to base64
            base64_image = base64.b64encode(buffer).decode('utf-8')
            return base64_image
            
        except Exception as e:
            logger.error(f"Error converting frame to base64: {str(e)}")
            return ""

    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image
        Args:
            image: Input image in RGB format
        Returns:
            Face embedding if face is detected, None otherwise
        """
        try:
            faces = self.app.get(image)
            if not faces:
                return None
            return faces[0].embedding
        except Exception as e:
            logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        Args:
            embedding1, embedding2: Face embeddings to compare
        Returns:
            Similarity score between 0 and 1
        """
        try:
            similarity = np.dot(embedding1, embedding2) / \
                        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return 0.0
    
    def add_reference_face(self, image_path: str, name: str) -> bool:
        """
        Add a reference face to the comparison set
        Args:
            image_path: Path to the reference image
            name: Identifier for this reference face
        Returns:
            bool: True if face was successfully added, False otherwise
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image from {image_path}")
                return False
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = self.get_face_embedding(img)
            
            if embedding is None:
                logger.error(f"No face detected in reference image: {name}")
                return False
                
            self.reference_faces.append(ReferenceFace(name, embedding))
            logger.info(f"Added reference face: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding reference face: {str(e)}")
            return False
    
    def send_webhook(self, webhook_url: str, data: Dict) -> bool:
        """
        Send webhook notification
        Args:
            webhook_url: URL to send webhook to
            data: Dictionary containing webhook data
        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        try:
            response = requests.post(
                webhook_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"Webhook sent successfully for {data.get('event')} - {data.get('person_name')}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook: {str(e)}")
            return False
        
    def process_video_stream(self, 
                           video_source: str, 
                           webhook_url: str,
                           min_notification_interval: int = 30,
                           display_output: bool = True) -> None:
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {video_source}")
            
            frame_count = 0
            active_faces = {}
            
            logger.info("Starting video stream processing...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Handle end of video
                    current_time = time.time()
                    for name, data in active_faces.items():
                        # Convert last frame to base64
                        base64_image = self.frame_to_base64(
                            data['last_frame'], 
                            name, 
                            "exit",
                            data.get('last_bbox')
                        )
                        end_data = {
                            "event": "Person Exit",
                            "person_name": name,
                            "start_timestamp": datetime.fromtimestamp(data['start_time']).isoformat(),
                            "end_timestamp": datetime.fromtimestamp(data['last_seen']).isoformat(),
                            "start_frame": data['start_frame'],
                            "end_frame": frame_count,
                            "duration_seconds": data['last_seen'] - data['start_time'],
                            "image_data": base64_image
                        }
                        self.send_webhook(webhook_url, end_data)
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.app.get(frame_rgb)
                
                current_time = time.time()
                detected_this_frame = set()
                
                for face in faces:
                    bbox = face.bbox.astype(int)
                    current_embedding = face.embedding
                    
                    for ref_face in self.reference_faces:
                        similarity = self.compare_embeddings(current_embedding, ref_face.embedding)
                        
                        if similarity >= self.threshold:
                            detected_this_frame.add(ref_face.name)
                            
                            # If this is a new detection
                            if ref_face.name not in active_faces:
                                # Convert entry frame to base64
                                base64_image = self.frame_to_base64(
                                    frame, 
                                    ref_face.name, 
                                    "entry",
                                    bbox.tolist()
                                )
                                # Send entry webhook with base64 image
                                start_data = {
                                    "event": "Person Enter",
                                    "person_name": ref_face.name,
                                    "timestamp": datetime.fromtimestamp(current_time).isoformat(),
                                    "frame_number": frame_count,
                                    "similarity_score": float(similarity),
                                    "bbox": bbox.tolist(),
                                    "image_data": base64_image
                                }
                                self.send_webhook(webhook_url, start_data)
                                
                                # Record start time, frame, and bbox
                                active_faces[ref_face.name] = {
                                    'start_time': current_time,
                                    'last_seen': current_time,
                                    'start_frame': frame_count,
                                    'last_frame': frame.copy(),
                                    'last_bbox': bbox.tolist()
                                }
                            else:
                                # Update last seen time, frame, and bbox
                                active_faces[ref_face.name]['last_seen'] = current_time
                                active_faces[ref_face.name]['last_frame'] = frame.copy()
                                active_faces[ref_face.name]['last_bbox'] = bbox.tolist()
                            
                            if display_output:
                                # Draw bounding box and info on frame
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    f"{ref_face.name}: {similarity:.2f}",
                                    (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2
                                )
                
                # Check for people who are no longer detected
                for name in list(active_faces.keys()):
                    if name not in detected_this_frame:
                        if current_time - active_faces[name]['last_seen'] >= min_notification_interval:
                            # Convert exit frame to base64
                            base64_image = self.frame_to_base64(
                                active_faces[name]['last_frame'],
                                name,
                                "exit",
                                active_faces[name].get('last_bbox')
                            )
                            # Send exit webhook with base64 image
                            end_data = {
                                "event": "Person Exit",
                                "person_name": name,
                                "start_timestamp": datetime.fromtimestamp(active_faces[name]['start_time']).isoformat(),
                                "end_timestamp": datetime.fromtimestamp(active_faces[name]['last_seen']).isoformat(),
                                "start_frame": active_faces[name]['start_frame'],
                                "end_frame": frame_count,
                                "duration_seconds": active_faces[name]['last_seen'] - active_faces[name]['start_time'],
                                "image_data": base64_image
                            }
                            self.send_webhook(webhook_url, end_data)
                            del active_faces[name]
                
                if display_output:
                    # Display frame info
                    cv2.putText(
                        frame,
                        f"Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow('Video Stream', frame)
                
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Send final notifications with base64 images
                    for name, data in active_faces.items():
                        # Convert exit frame to base64
                        base64_image = self.frame_to_base64(
                            data['last_frame'],
                            name,
                            "exit",
                            data.get('last_bbox')
                        )
                        end_data = {
                            "event": "Person Exit",
                            "person_name": name,
                            "start_timestamp": datetime.fromtimestamp(data['start_time']).isoformat(),
                            "end_timestamp": datetime.fromtimestamp(data['last_seen']).isoformat(),
                            "start_frame": data['start_frame'],
                            "end_frame": frame_count,
                            "duration_seconds": data['last_seen'] - data['start_time'],
                            "image_data": base64_image
                        }
                        self.send_webhook(webhook_url, end_data)
                    break

        except Exception as e:
            logger.error(f"Error in video stream processing: {str(e)}")
        finally:
            cap.release()
            if display_output:
                cv2.destroyAllWindows()
            logger.info("Video stream processing ended")

def main():
    # Configuration
    CONFIG = {
        'REFERENCE_DIR': '/Users/amansubash/Downloads/face_images',  # Directory containing reference images
        'WEBHOOK_URL': 'https://webhook.site/1b1d0b96-8e4b-45f7-bc29-1df77a878c74',    # Replace with your webhook URL
        'RTSP_URL': '/Users/amansubash/Downloads/F1.mp4',         # Replace with your video file path
        'THRESHOLD': 0.3,                     # Face matching threshold (30%)
        'MIN_NOTIFICATION_INTERVAL': 0,      # Minimum seconds between notifications
        'DISPLAY_OUTPUT': True                # Whether to show video output
    }
    
    try:
        # Initialize face comparer
        face_comparer = FaceComparer(threshold=CONFIG['THRESHOLD'])
        
        # Check if reference directory exists
        if not os.path.exists(CONFIG['REFERENCE_DIR']):
            logger.error(f"Reference directory not found: {CONFIG['REFERENCE_DIR']}")
            return
        
        # Load all reference images
        reference_faces_loaded = 0
        for filename in os.listdir(CONFIG['REFERENCE_DIR']):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(CONFIG['REFERENCE_DIR'], filename)
                name = os.path.splitext(filename)[0]  # Use filename as person's name
                if face_comparer.add_reference_face(image_path, name):
                    reference_faces_loaded += 1
        
        if reference_faces_loaded == 0:
            logger.error("No reference faces were loaded successfully!")
            return
        
        logger.info(f"Successfully loaded {reference_faces_loaded} reference faces")
        
        # Process video stream
        logger.info("Starting video stream processing...")
        face_comparer.process_video_stream(
            CONFIG['RTSP_URL'],
            CONFIG['WEBHOOK_URL'],
            CONFIG['MIN_NOTIFICATION_INTERVAL'],
            CONFIG['DISPLAY_OUTPUT']
        )
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()