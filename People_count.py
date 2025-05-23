import cv2
import numpy as np
from collections import deque

class MotionBasedCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self.total_count = 0
        self.frame_count = 0
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=50, 
            detectShadows=True
        )
        
        # Motion tracking
        self.motion_history = deque(maxlen=10)
        self.last_motion_frame = None
        
        # Line crossing detection
        self.crossing_buffer = deque(maxlen=30)  # Store recent crossings
        self.min_crossing_interval = 15  # Minimum frames between counts
        self.last_crossing_frame = 0
        
        # Calibration
        self.calibration_frames = 50
        self.is_calibrated = False
        
    def get_line_mask(self, frame_shape):
        """Create a mask for the counting line area"""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        
        # Create a thick line mask
        cv2.line(mask, self.line_start, self.line_end, 255, 30)
        
        return mask
    
    def detect_line_crossings(self, motion_mask, frame_shape):
        """Detect crossings based on motion across the line"""
        line_mask = self.get_line_mask(frame_shape)
        
        # Get motion only in the line area
        line_motion = cv2.bitwise_and(motion_mask, line_mask)
        
        # Calculate motion intensity
        motion_pixels = cv2.countNonZero(line_motion)
        motion_intensity = motion_pixels / cv2.countNonZero(line_mask) if cv2.countNonZero(line_mask) > 0 else 0
        
        return motion_intensity, line_motion
    
    def analyze_motion_direction(self, current_frame, line_motion):
        """Analyze motion direction across the line"""
        if self.last_motion_frame is None:
            self.last_motion_frame = current_frame
            return 0
        
        # Calculate optical flow in line region
        flow = cv2.calcOpticalFlowPyrLK(
            self.last_motion_frame, 
            current_frame,
            None, None,
            winSize=(15, 15),
            maxLevel=2
        )
        
        self.last_motion_frame = current_frame
        return 0
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Detect motion in line area
        motion_intensity, line_motion = self.detect_line_crossings(fg_mask, frame.shape)
        
        # Store motion history
        self.motion_history.append(motion_intensity)
        
        # Skip counting during calibration
        if self.frame_count < self.calibration_frames:
            status_text = f"Calibrating... {self.frame_count}/{self.calibration_frames}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.draw_overlay(frame, motion_intensity, line_motion)
            return frame
        
        if not self.is_calibrated:
            self.is_calibrated = True
            print("✅ Calibration complete! Starting counting...")
        
        # Detect significant motion spikes (potential crossings)
        if len(self.motion_history) >= 5:
            recent_avg = np.mean(list(self.motion_history)[-5:])
            overall_avg = np.mean(list(self.motion_history)[-10:])
            
            # Crossing detection logic
            crossing_threshold = 0.15  # Adjust based on your scene
            
            if (recent_avg > crossing_threshold and 
                recent_avg > overall_avg * 1.5 and
                self.frame_count - self.last_crossing_frame > self.min_crossing_interval):
                
                # Additional validation
                if self.validate_crossing(line_motion):
                    self.total_count += 1
                    self.last_crossing_frame = self.frame_count
                    print(f"🚶 Motion crossing detected! Count: {self.total_count} (Intensity: {recent_avg:.3f})")
                    
                    # Visual feedback
                    cv2.putText(frame, "CROSSING!", (200, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        self.draw_overlay(frame, motion_intensity, line_motion)
        return frame
    
    def validate_crossing(self, line_motion):
        """Additional validation for crossing detection"""
        # Check if motion is distributed across the line (not just noise)
        contours, _ = cv2.findContours(line_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        # Check for reasonable contour sizes
        valid_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Reasonable size range
                valid_contours += 1
        
        return valid_contours > 0
    
    def draw_overlay(self, frame, motion_intensity, line_motion):
        """Draw visualization overlay"""
        # Draw counting line
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 0), 4)
        
        # Draw line endpoints
        cv2.circle(frame, self.line_start, 8, (0, 255, 0), -1)
        cv2.circle(frame, self.line_end, 8, (0, 0, 255), -1)
        
        # Count display (CHANGED TO BLACK)
        count_text = f"Count: {self.total_count}"
        cv2.rectangle(frame, (10, 60), (300, 120), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # Motion intensity
        intensity_text = f"Motion: {motion_intensity:.3f}"
        cv2.putText(frame, intensity_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show motion history as a simple graph
        if len(self.motion_history) > 1:
            history_array = np.array(list(self.motion_history))
            max_val = max(0.5, np.max(history_array))
            
            for i in range(1, len(history_array)):
                pt1 = (frame.shape[1] - 200 + i * 10, frame.shape[0] - 100 - int(history_array[i-1] * 50 / max_val))
                pt2 = (frame.shape[1] - 200 + (i+1) * 10, frame.shape[0] - 100 - int(history_array[i] * 50 / max_val))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
        
        # Show motion in line area (small preview)
        if line_motion is not None:
            line_motion_colored = cv2.applyColorMap(line_motion, cv2.COLORMAP_HOT)
            line_motion_small = cv2.resize(line_motion_colored, (100, 60))
            frame[frame.shape[0]-70:frame.shape[0]-10, 10:110] = line_motion_small

class OpticalFlowCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self.total_count = 0
        self.frame_count = 0
        self.prev_gray = None
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.tracks = []
        self.track_id = 0
        self.crossing_history = deque(maxlen=30)
        
    def process_frame(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # Calculate optical flow
        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            
            # Update tracks
            good_new = p1[_st == 1]
            good_old = p0[_st == 1]
            
            new_tracks = []
            for i, (tr, (x, y)) in enumerate(zip(self.tracks, p1.reshape(-1, 2))):
                if _st[i] == 1:
                    tr.append((x, y))
                    if len(tr) > 10:
                        del tr[0]
                    new_tracks.append(tr)
                    
                    # Check for line crossing
                    if len(tr) >= 2:
                        self.check_crossing(tr[-2], tr[-1])
            
            self.tracks = new_tracks
            
            # Draw tracks
            for tr in self.tracks:
                cv2.polylines(frame, [np.int32(tr)], False, (0, 255, 0), 2)
        
        # Detect new features
        mask = np.zeros_like(gray)
        mask[:] = 255
        
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        
        p = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])
        
        self.prev_gray = gray
        
        # Draw line and count (CHANGED TO BLACK)
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 0), 3)
        cv2.putText(frame, f"Count: {self.total_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return frame
    
    def check_crossing(self, p1, p2):
        """Check if a track crossed the line"""
        # Simple line crossing detection
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        if intersect(p1, p2, (x1, y1), (x2, y2)):
            self.total_count += 1
            print(f"Track crossing detected! Count: {self.total_count}")

def main():
    cap = cv2.VideoCapture("/Users/amansubash/Downloads/video32.mov")
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Could not read video. Check the file path.")
        return

    height, width = frame.shape[:2]
    print(f"📹 Video dimensions: {width}x{height}")
    
    # Try horizontal line across the middle
    line_start = (int(width * 0.2), int(height * 0.5))
    line_end = (int(width * 0.8), int(height * 0.5))
    
    print("Choose counting method:")
    print("1. Motion-based counting (recommended for crowds)")
    print("2. Optical flow tracking")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        counter = OpticalFlowCounter(line_start, line_end)
        print("🔄 Using Optical Flow Counter")
    else:
        counter = MotionBasedCounter(line_start, line_end)
        print("🔄 Using Motion-Based Counter")

    print("🚀 Counter Started")
    print("Press 'q' to quit")
    print("Press 'r' to reset counter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        frame = counter.process_frame(frame)
        
        # Resize for display if needed
        display_frame = frame
        if width > 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imshow("Crowd Counter", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counter.total_count = 0
            if hasattr(counter, 'tracked_ids'):
                counter.tracked_ids.clear()
            print("🔄 Counter reset!")

    print(f"📊 Final count: {counter.total_count}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
