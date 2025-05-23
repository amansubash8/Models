import cv2
import numpy as np
from ultralytics import YOLO
import requests
import base64
from datetime import datetime
from dateutil.tz import tzlocal
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhoneDetector:
    def __init__(self, rtsp_url, webhook_url, confidence_threshold=0.5, display_output=True):
        """
        Initialize the phone detector
        Args:
            rtsp_url (str): RTSP stream URL
            webhook_url (str): Webhook URL to send detection results
            confidence_threshold (float): Confidence threshold for detections
            display_output (bool): Whether to display the video output
        """
        self.rtsp_url = rtsp_url
        self.webhook_url = webhook_url
        self.confidence_threshold = confidence_threshold
        self.display_output = display_output
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8l.pt')  # Using the nano model
        self.frame_count = 0

    def send_to_webhook(self, data):
        """
        Send detection results to webhook with improved error handling and retries
        """
        try:
            # Compress the image before sending
            if 'image_base64' in data:
                # Reduce image quality to decrease payload size
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(data['image_base64']), np.uint8), cv2.IMREAD_COLOR)
                # Resize image to reduce size (adjust dimensions as needed)
                frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                data['image_base64'] = base64.b64encode(buffer).decode('utf-8')

            # Add retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = requests.post(
                        self.webhook_url, 
                        json=data,
                        headers={
                            'Content-Type': 'application/json',
                            'User-Agent': 'PhoneDetector/1.0'
                        },
                        timeout=10  # Increased timeout
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Successfully sent data to webhook. Status code: {response.status_code}")
                        # Print the response content for debugging
                        logger.info(f"Webhook response: {response.text[:200]}...")  # Print first 200 chars
                        return True
                    else:
                        logger.warning(f"Webhook returned status code: {response.status_code}")
                        logger.warning(f"Response content: {response.text[:200]}...")
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Retry {retry_count + 1}/{max_retries} failed: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)  # Wait 1 second before retrying
                    continue
                
                break  # Break if request was successful
                
            if retry_count == max_retries:
                logger.error("Failed to send webhook after maximum retries")
                return False
                
        except Exception as e:
            logger.error(f"Error preparing webhook data: {str(e)}")
            return False

    def frame_to_base64(self, frame) -> str:
        """
        Convert OpenCV frame to base64 string with compression
        """
        try:
            # Resize the frame to reduce size
            frame = cv2.resize(frame, (640, 480))
            # Compress image with reduced quality
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return img_str
        except Exception as e:
            logger.error(f"Error converting frame to base64: {str(e)}")
            return ""

    def process_frame(self, frame):
        """
        Process a single frame and detect phones
        Args:
            frame: Video frame to process
        Returns:
            Processed frame and detection results
        """
        try:
            start_time = datetime.now(tzlocal())
            
            # Run YOLOv8 inference
            results = self.model(frame)
            
            end_time = datetime.now(tzlocal())
            
            # Get phone detections (class 67 in COCO dataset is cell phone)
            phone_detections = []
            phones_in_current_frame = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if detection is a phone (class 67) and meets confidence threshold
                    if cls == 67 and conf >= self.confidence_threshold:
                        phones_in_current_frame += 1
                        xyxy = box.xyxy[0].cpu().numpy()  # get box coordinates
                        phone_detections.append({
                            'confidence': conf,
                            'bbox': xyxy.tolist()
                        })
                        
                        # Draw bounding box on frame
                        cv2.rectangle(frame, 
                                    (int(xyxy[0]), int(xyxy[1])), 
                                    (int(xyxy[2]), int(xyxy[3])), 
                                    (0, 255, 0), 2)
                        
                        # Add confidence score text
                        cv2.putText(frame, 
                                  f'Phone: {conf:.2f}', 
                                  (int(xyxy[0]), int(xyxy[1] - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  (0, 255, 0), 
                                  2)

            return frame, phone_detections, start_time, end_time, phones_in_current_frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, [], start_time, datetime.now(tzlocal()), 0

    def run(self):
        """
        Main method to run the detection pipeline
        """
        try:
            logger.info("Starting video stream processing...")
            cap = cv2.VideoCapture(self.rtsp_url)
            
            if not cap.isOpened():
                logger.error(f"Error: Could not open video source: {self.rtsp_url}")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error: Could not read frame from stream")
                    break

                self.frame_count += 1
                
                # Process the frame
                processed_frame, detections, start_time, end_time, phones_in_frame = self.process_frame(frame)
                
                # Add frame counter and phones in current frame to the display
                cv2.putText(
                    processed_frame,
                    f"Frame: {self.frame_count} | Phones in frame: {phones_in_frame}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # If phones were detected, send to webhook
                if phones_in_frame > 0:
                    # Prepare webhook data
                    webhook_data = {
                        'frame_number': self.frame_count,
                        'start_timestamp': start_time.isoformat(),
                        'end_timestamp': end_time.isoformat(),
                        'detections': detections,
                        'phones_in_frame': phones_in_frame,
                        'image_base64': self.frame_to_base64(processed_frame)
                    }
                    
                    # Send to webhook
                    self.send_to_webhook(webhook_data)

                # Display the output if enabled
                if self.display_output:
                    cv2.imshow('Phone Detection Stream', processed_frame)
                    
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested stop")
                        break

        except KeyboardInterrupt:
            logger.info("Stopping detection...")
        except Exception as e:
            logger.error(f"Error in video stream processing: {str(e)}")
        finally:
            cap.release()
            if self.display_output:
                cv2.destroyAllWindows()
            logger.info("Video stream processing ended")

def main():
    # Configuration
    CONFIG = {
        'RTSP_URL': '/Users/amansubash/Downloads/phone.mp4',
        'WEBHOOK_URL': 'https://webhook.site/1b1d0b96-8e4b-45f7-bc29-1df77a878c74',
        'CONFIDENCE_THRESHOLD': 0.2,
        'DISPLAY_OUTPUT': True
    }
    
    try:
        # Test webhook connection before starting
        test_data = {
            "test": "Connection test",
            "timestamp": datetime.now(tzlocal()).isoformat()
        }
        
        test_response = requests.post(
            CONFIG['WEBHOOK_URL'],
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if test_response.status_code != 200:
            logger.error(f"Webhook test failed with status code: {test_response.status_code}")
            logger.error(f"Response: {test_response.text[:200]}...")
            return
            
        logger.info("Webhook test successful, starting detection...")
        
        # Initialize phone detector
        detector = PhoneDetector(
            CONFIG['RTSP_URL'],
            CONFIG['WEBHOOK_URL'],
            CONFIG['CONFIDENCE_THRESHOLD'],
            CONFIG['DISPLAY_OUTPUT']
        )
        
        # Start detection
        detector.run()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Webhook test failed with error: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()