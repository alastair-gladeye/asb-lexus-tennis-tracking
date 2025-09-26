"""
Robust Tennis Ball Tracker with Object Tracking
===============================================

This version uses proper object tracking instead of frame-by-frame detection.
Addresses the false positive issue by:
1. Manual ball initialization on first frame
2. OpenCV object trackers (CSRT, KCF, etc.)
3. Temporal consistency enforcement
4. ROI-based tracking with fallback detection

Usage:
1. First frame: Click on the tennis ball to initialize tracking
2. Tracker follows the ball through the video
3. Automatic re-detection if tracking is lost
"""

import cv2
import numpy as np
import json
from pathlib import Path

class RobustTennisBallTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.homography = None
        self.camera_matrix = None
        
        # Tennis court dimensions (meters)
        self.court_3d = np.array([
            [0, 0, 0],
            [23.77, 0, 0],
            [23.77, 10.97, 0],
            [0, 10.97, 0]
        ], dtype=np.float32)
        
        # Ball parameters
        self.BALL_DIAMETER = 0.067  # meters
        
        # Tracking state
        self.tracker = None
        self.tracker_initialized = False
        self.last_good_position = None
        self.tracking_confidence = 0.0
        self.lost_frames = 0
        self.max_lost_frames = 10
        
        # Available tracker types (compatible with different OpenCV versions)
        self.tracker_types = {
            'CSRT': self.create_csrt_tracker,      # Best for accuracy
            'KCF': self.create_kcf_tracker,        # Good balance
            'MOSSE': self.create_mosse_tracker,    # Fastest
            'MIL': self.create_mil_tracker,        # Good for tennis balls
        }
        
        self.current_tracker_type = 'CSRT'  # Default to most accurate
    
    def create_csrt_tracker(self):
        """Create CSRT tracker with version compatibility"""
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                print("CSRT tracker not available, falling back to KCF")
                return self.create_kcf_tracker()
    
    def create_kcf_tracker(self):
        """Create KCF tracker with version compatibility"""
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerKCF_create()
            except AttributeError:
                print("KCF tracker not available, falling back to MOSSE")
                return self.create_mosse_tracker()
    
    def create_mosse_tracker(self):
        """Create MOSSE tracker with version compatibility"""
        try:
            return cv2.TrackerMOSSE_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerMOSSE_create()
            except AttributeError:
                print("MOSSE tracker not available, falling back to MIL")
                return self.create_mil_tracker()
    
    def create_mil_tracker(self):
        """Create MIL tracker with version compatibility"""
        try:
            return cv2.TrackerMIL_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerMIL_create()
            except AttributeError:
                print("❌ No trackers available in this OpenCV version")
                return None
    
    def calibrate_court(self, frame=None):
        """Step 1: Calibrate court (manual for now)"""
        if frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        print("COURT CALIBRATION")
        print("Click on 4 court corners in order:")
        print("1. Baseline left (near camera)")
        print("2. Baseline right (near camera)")
        print("3. Far baseline right")
        print("4. Far baseline left")
        print("Press 'r' to reset points, 'c' to continue")
        
        points = []
        temp_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                temp_frame = frame.copy()
                
                # Draw all points
                for i, point in enumerate(points):
                    cv2.circle(temp_frame, tuple(point), 8, (0, 255, 0), -1)
                    cv2.putText(temp_frame, f"{i+1}", (point[0]+15, point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw lines between points
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(temp_frame, tuple(points[i]), tuple(points[i+1]), (255, 0, 0), 2)
                
                cv2.imshow('Court Calibration', temp_frame)
        
        cv2.namedWindow('Court Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Court Calibration', 1280, 720)
        cv2.setMouseCallback('Court Calibration', mouse_callback)
        cv2.imshow('Court Calibration', frame)
        
        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Reset points
                points = []
                temp_frame = frame.copy()
                cv2.imshow('Court Calibration', temp_frame)
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Calculate homography
        points = np.array(points, dtype=np.float32)
        court_2d = self.court_3d[:, :2]
        self.homography, _ = cv2.findHomography(points, court_2d)
        
        # Estimate camera parameters
        h, w = frame.shape[:2]
        focal = w
        self.camera_matrix = np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ])
        
        print("✅ Court calibration complete!")
        return self.homography
    
    def initialize_ball_tracking(self, frame):
        """Initialize ball tracking by manual selection"""
        print("\nBALL INITIALIZATION")
        print("Click and drag to select the tennis ball")
        print("Press 'r' to reset selection, ENTER to confirm, 'q' to quit")
        
        bbox = None
        selecting = False
        start_point = None
        temp_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal bbox, selecting, start_point, temp_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_point = (x, y)
                bbox = None
            
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Ball Initialization', temp_frame)
            
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                if start_point:
                    # Calculate bounding box
                    x1, y1 = start_point
                    x2, y2 = x, y
                    
                    # Ensure proper ordering
                    bbox = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                    
                    # Draw final selection
                    temp_frame = frame.copy()
                    cv2.rectangle(temp_frame, (bbox[0], bbox[1]), 
                                 (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                    cv2.putText(temp_frame, "Press ENTER to confirm, 'r' to reset", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Ball Initialization', temp_frame)
        
        cv2.namedWindow('Ball Initialization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ball Initialization', 1280, 720)
        cv2.setMouseCallback('Ball Initialization', mouse_callback)
        cv2.imshow('Ball Initialization', frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset selection
                bbox = None
                temp_frame = frame.copy()
                cv2.imshow('Ball Initialization', temp_frame)
            
            elif key == 13:  # ENTER key
                if bbox and bbox[2] > 10 and bbox[3] > 10:  # Valid selection
                    break
                else:
                    print("Please select a valid region for the ball")
            
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Initialize tracker
        self.tracker = self.tracker_types[self.current_tracker_type]()
        success = self.tracker.init(frame, bbox)
        
        if success:
            self.tracker_initialized = True
            self.last_good_position = self.bbox_to_center_radius(bbox)
            print(f"✅ Ball tracking initialized with {self.current_tracker_type} tracker")
            return bbox
        else:
            print("❌ Failed to initialize tracker")
            return None
    
    def bbox_to_center_radius(self, bbox):
        """Convert bounding box to center point and radius"""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2
        return (center_x, center_y, radius)
    
    def track_ball_in_frame(self, frame):
        """Track ball in current frame"""
        if not self.tracker_initialized:
            return None
        
        # Update tracker
        success, bbox = self.tracker.update(frame)
        
        if success:
            # Convert bbox to center and radius
            ball_position = self.bbox_to_center_radius(bbox)
            
            # Validate tracking result
            if self.validate_tracking_result(ball_position, frame):
                self.last_good_position = ball_position
                self.tracking_confidence = 0.8  # High confidence for successful tracking
                self.lost_frames = 0
                return ball_position, bbox
            else:
                success = False
        
        if not success:
            self.lost_frames += 1
            self.tracking_confidence = max(0.0, self.tracking_confidence - 0.1)
            
            # Try to re-detect ball near last known position
            if self.last_good_position and self.lost_frames < self.max_lost_frames:
                detected_ball = self.search_for_ball_near_position(frame, self.last_good_position)
                if detected_ball:
                    # Reinitialize tracker with new detection
                    center_x, center_y, radius = detected_ball
                    new_bbox = (center_x - radius, center_y - radius, radius * 2, radius * 2)
                    
                    self.tracker = self.tracker_types[self.current_tracker_type]()
                    if self.tracker.init(frame, new_bbox):
                        self.last_good_position = detected_ball
                        self.tracking_confidence = 0.6  # Medium confidence for re-detection
                        self.lost_frames = 0
                        print(f"🔄 Re-initialized tracking at frame")
                        return detected_ball, new_bbox
            
            # If we've lost tracking for too long, return None
            if self.lost_frames >= self.max_lost_frames:
                print(f"⚠️ Lost tracking for {self.lost_frames} frames")
                return None
        
        return None
    
    def validate_tracking_result(self, position, frame):
        """Validate that tracking result makes sense"""
        if not position:
            return False
        
        center_x, center_y, radius = position
        
        # Check if position is within frame bounds
        if (center_x < radius or center_x >= frame.shape[1] - radius or
            center_y < radius or center_y >= frame.shape[0] - radius):
            return False
        
        # Check for reasonable movement from last position
        if self.last_good_position:
            last_x, last_y, last_r = self.last_good_position
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            
            # Reasonable movement threshold (pixels per frame)
            max_movement = 100  # Adjust based on your video
            if distance > max_movement:
                return False
        
        # Check if the tracked region looks like a ball
        color_score = self.analyze_region_color(frame, center_x, center_y, radius)
        return color_score > 0.3  # Minimum color similarity threshold
    
    def analyze_region_color(self, frame, center_x, center_y, radius):
        """Analyze if region looks like a tennis ball"""
        try:
            # Extract region
            y1 = max(0, center_y - radius)
            y2 = min(frame.shape[0], center_y + radius)
            x1 = max(0, center_x - radius)
            x2 = min(frame.shape[1], center_x + radius)
            
            if y2 <= y1 or x2 <= x1:
                return 0.0
            
            region = frame[y1:y2, x1:x2]
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Check for tennis ball colors
            yellow_mask = cv2.inRange(hsv_region, np.array([15, 50, 50]), np.array([45, 255, 255]))
            bright_mask = cv2.inRange(hsv_region, np.array([0, 0, 150]), np.array([180, 50, 255]))
            
            total_pixels = region.shape[0] * region.shape[1]
            yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
            bright_ratio = np.sum(bright_mask > 0) / total_pixels
            
            return min(1.0, yellow_ratio * 2 + bright_ratio)
            
        except Exception:
            return 0.0
    
    def search_for_ball_near_position(self, frame, last_position):
        """Search for ball near last known position"""
        center_x, center_y, radius = last_position
        search_radius = radius * 3  # Search in larger area
        
        # Define search region
        y1 = max(0, center_y - search_radius)
        y2 = min(frame.shape[0], center_y + search_radius)
        x1 = max(0, center_x - search_radius)
        x2 = min(frame.shape[1], center_x + search_radius)
        
        search_region = frame[y1:y2, x1:x2]
        
        # Simple ball detection in search region
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Yellow tennis ball mask
        lower = np.array([15, 80, 80])
        upper = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 1000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity > 0.5:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            # Convert back to full frame coordinates
                            rel_cx = int(M["m10"] / M["m00"])
                            rel_cy = int(M["m01"] / M["m00"])
                            abs_cx = x1 + rel_cx
                            abs_cy = y1 + rel_cy
                            candidate_radius = int(np.sqrt(area / np.pi))
                            
                            # Score based on distance from last position and circularity
                            distance = np.sqrt((abs_cx - center_x)**2 + (abs_cy - center_y)**2)
                            score = circularity * (1.0 - distance / search_radius)
                            
                            if score > best_score:
                                best_score = score
                                best_candidate = (abs_cx, abs_cy, candidate_radius)
        
        return best_candidate if best_score > 0.4 else None
    
    def track_video_robust(self):
        """Track ball through video using robust tracking"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get first frame for ball initialization
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Could not read first frame")
            return []
        
        # Initialize ball tracking
        bbox = self.initialize_ball_tracking(first_frame)
        if not bbox:
            print("❌ Ball tracking initialization failed")
            return []
        
        # Reset to beginning for tracking
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        detections = []
        frame_num = 0
        
        print("🎾 Starting robust ball tracking...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num == 0:
                # Re-initialize on first frame
                self.tracker = self.tracker_types[self.current_tracker_type]()
                self.tracker.init(frame, bbox)
                ball_position = self.bbox_to_center_radius(bbox)
            else:
                # Track ball
                result = self.track_ball_in_frame(frame)
                ball_position = result[0] if result else None
                bbox = result[1] if result else None
            
            # Visualize tracking
            display_frame = frame.copy()
            
            if ball_position:
                center_x, center_y, radius = ball_position
                
                # Color based on confidence
                if self.tracking_confidence > 0.7:
                    color = (0, 255, 0)  # Green = high confidence
                elif self.tracking_confidence > 0.4:
                    color = (0, 255, 255)  # Yellow = medium confidence
                else:
                    color = (0, 0, 255)  # Red = low confidence
                
                cv2.circle(display_frame, (center_x, center_y), radius, color, 2)
                cv2.circle(display_frame, (center_x, center_y), 3, color, -1)
                
                # Draw bounding box if available
                if bbox:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 1)
                
                detections.append({
                    'frame': frame_num,
                    'x': center_x,
                    'y': center_y,
                    'radius': radius,
                    'confidence': self.tracking_confidence
                })
            
            # Status display
            status = f"Frame: {frame_num} | Tracked: {len(detections)} | Conf: {self.tracking_confidence:.2f}"
            if self.lost_frames > 0:
                status += f" | Lost: {self.lost_frames}"
            
            cv2.putText(display_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize for display
            if display_frame.shape[1] > 1280:
                scale = 1280 / display_frame.shape[1]
                new_width = int(display_frame.shape[1] * scale)
                new_height = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            cv2.imshow('Robust Tennis Ball Tracking', display_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reinitialize tracking
                print("🔄 Reinitializing tracker...")
                new_bbox = self.initialize_ball_tracking(frame)
                if new_bbox:
                    bbox = new_bbox
                    self.tracker = self.tracker_types[self.current_tracker_type]()
                    self.tracker.init(frame, bbox)
                    self.tracker_initialized = True
                    self.lost_frames = 0
                    self.tracking_confidence = 0.8
            
            frame_num += 1
        
        cv2.destroyAllWindows()
        print(f"✅ Robust tracking complete! Tracked {len(detections)} frames")
        return detections
    
    def reconstruct_3d(self, detections):
        """Convert 2D tracking to 3D positions"""
        positions_3d = []
        
        for det in detections:
            # Get ground position using homography
            point = np.array([[det['x'], det['y']]], dtype=np.float32)
            point = np.array([point])
            court_pos = cv2.perspectiveTransform(point, self.homography)[0][0]
            
            # Height estimation based on ball size
            max_height = 4.0
            min_height = 0.067
            
            normalized_radius = np.clip(det['radius'] / 25.0, 0.2, 3.0)
            height = max_height / normalized_radius
            height = np.clip(height, min_height, max_height)
            
            positions_3d.append({
                'frame': det['frame'],
                'time': det['frame'] / self.fps,
                'x': float(court_pos[0]),
                'y': float(court_pos[1]),
                'z': float(height),
                'confidence': det.get('confidence', 1.0)
            })
        
        return positions_3d
    
    def smooth_trajectory(self, positions_3d):
        """Apply trajectory smoothing"""
        if len(positions_3d) < 3:
            return positions_3d
        
        smoothed = []
        window = 3  # Smaller window to preserve sharp movements
        
        for i, pos in enumerate(positions_3d):
            start = max(0, i - window // 2)
            end = min(len(positions_3d), i + window // 2 + 1)
            
            window_positions = positions_3d[start:end]
            
            # Weight by confidence
            weights = [p.get('confidence', 1.0) for p in window_positions]
            weights = np.array(weights)
            weights /= np.sum(weights)
            
            avg_x = np.average([p['x'] for p in window_positions], weights=weights)
            avg_y = np.average([p['y'] for p in window_positions], weights=weights)
            avg_z = np.average([p['z'] for p in window_positions], weights=weights)
            
            smoothed.append({
                'frame': pos['frame'],
                'time': pos['time'],
                'x': avg_x,
                'y': avg_y,
                'z': avg_z,
                'confidence': pos.get('confidence', 1.0)
            })
        
        return smoothed
    
    def export_data(self, positions_3d, output_path):
        """Export tracking data"""
        data = {
            'metadata': {
                'video': str(self.video_path),
                'fps': self.fps,
                'court_dimensions': [23.77, 10.97],
                'units': 'meters',
                'total_frames': len(positions_3d),
                'tracking_method': f'robust_object_tracking_{self.current_tracker_type}',
                'tracker_type': self.current_tracker_type
            },
            'tracking_data': positions_3d
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(positions_3d)} frames to {output_path}")
    
    def run_full_pipeline(self, output_path='tennis_tracking_robust.json'):
        """Run complete robust tracking pipeline"""
        print("🎾 ROBUST TENNIS BALL TRACKING")
        print("=" * 50)
        
        print("\nStep 1: Court Calibration")
        homography = self.calibrate_court()
        if homography is None:
            return []
        
        print("\nStep 2: Robust Ball Tracking")
        detections = self.track_video_robust()
        if not detections:
            print("❌ No ball tracking data captured")
            return []
        
        print(f"✅ Tracked ball in {len(detections)} frames")
        
        print("\nStep 3: 3D Reconstruction")
        positions_3d = self.reconstruct_3d(detections)
        
        print("\nStep 4: Trajectory Smoothing")
        positions_3d = self.smooth_trajectory(positions_3d)
        
        print("\nStep 5: Exporting Data")
        self.export_data(positions_3d, output_path)
        
        self.cap.release()
        return positions_3d

# Usage
if __name__ == "__main__":
    tracker = RobustTennisBallTracker('assets/tennis.mp4')
    positions = tracker.run_full_pipeline('tennis_tracking_robust.json')
    print(f"\n🎾 Robust tracking complete! Tracked {len(positions)} frames")
    print("This should have much better consistency than the previous method!")
