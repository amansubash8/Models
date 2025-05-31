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
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = threshold
        self.reference_faces: List[ReferenceFace] = []

    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = self.app.get(image)
            if not faces:
                return None
            return faces[0].embedding
        except Exception as e:
            logger.error(f"Error getting face embedding: {str(e)}")
            return None

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            similarity = np.dot(embedding1, embedding2) / \
                         (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return 0.0

    def add_reference_face(self, image_path: str, name: str) -> bool:
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

    def process_video_stream(self, video_source: str, save_output: bool = True, display_output: bool = True) -> None:
        if not video_source.lower().startswith('rtsp://'):
            raise ValueError("URL must start with 'rtsp://'")

        # Set RTSP transport protocol options
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

        # Setup video capture with RTSP-specific settings
        cap = cv2.VideoCapture(video_source)
        
        # Configure RTSP buffer size and timeout
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer size for more real-time processing
        
        # Try different transport protocols if initial connection fails
        if not cap.isOpened():
            logger.info("Trying UDP transport...")
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                logger.info("Trying HTTP tunnel...")
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;http'
                cap = cv2.VideoCapture(video_source)
                
                if not cap.isOpened():
                    raise ValueError(f"Could not open RTSP stream after trying all transport protocols: {video_source}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Webcams and some RTSP streams might not provide FPS
            fps = 30

        # Setup video writer if saving is enabled
        output_writer = None
        if save_output:
            # Create output directory
            output_dir = os.path.join(os.path.dirname(__file__), 'output_videos')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'video_stream_{timestamp}.mp4')
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            logger.info(f"Saving output to: {output_path}")

        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10

        logger.info(f"Starting video stream processing from: {video_source}")

        try:
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_failures += 1
                        logger.warning(f"Failed to read frame (failure {consecutive_failures}/{max_consecutive_failures})")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.info("Attempting to reconnect to video stream...")
                            cap.release()
                            time.sleep(2)
                            cap = cv2.VideoCapture(video_source)
                            if not cap.isOpened():
                                logger.error("Failed to reconnect to video stream")
                                break
                            consecutive_failures = 0
                        continue

                    consecutive_failures = 0

                    # Process frame for face detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(frame_rgb)

                    # Draw detection results on frame
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        current_embedding = face.embedding

                        # Compare with reference faces
                        for ref_face in self.reference_faces:
                            similarity = self.compare_embeddings(current_embedding, ref_face.embedding)
                            if similarity >= self.threshold:
                                # Draw bounding box and label
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                                cv2.putText(frame, f"{ref_face.name}: {similarity:.2f}",
                                          (bbox[0], bbox[1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Add frame counter and stream info
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Video Stream", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save frame if enabled
                    if output_writer is not None:
                        output_writer.write(frame)

                    # Display frame if enabled
                    if display_output:
                        cv2.imshow('Video Stream Monitor', frame)

                    frame_count += 1

                    # Break on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            if output_writer is not None:
                output_writer.release()
            if display_output:
                cv2.destroyAllWindows()
            logger.info("Video stream processing ended")

def main():
    # Configuration
    CONFIG = {
        'REFERENCE_DIR': '/Users/amansubash/Downloads/face_images',
        'RTSP_URL': 'rtsp://your-camera-ip:554/stream',  # Replace with your RTSP camera URL
        'THRESHOLD': 0.3,
        'DISPLAY_OUTPUT': True,
        'SAVE_OUTPUT': True
    }

    try:
        # Initialize face detector
        face_comparer = FaceComparer(threshold=CONFIG['THRESHOLD'])

        # Load reference faces
        if not os.path.exists(CONFIG['REFERENCE_DIR']):
            logger.error(f"Reference directory not found: {CONFIG['REFERENCE_DIR']}")
            return

        # Load reference faces
        reference_faces_loaded = 0
        for filename in os.listdir(CONFIG['REFERENCE_DIR']):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(CONFIG['REFERENCE_DIR'], filename)
                name = os.path.splitext(filename)[0]
                if face_comparer.add_reference_face(image_path, name):
                    reference_faces_loaded += 1

        if reference_faces_loaded == 0:
            logger.error("No reference faces were loaded!")
            return

        logger.info(f"Successfully loaded {reference_faces_loaded} reference faces")

        # Start processing video stream
        face_comparer.process_video_stream(
            CONFIG['RTSP_URL'],
            save_output=CONFIG['SAVE_OUTPUT'],
            display_output=CONFIG['DISPLAY_OUTPUT']
        )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
