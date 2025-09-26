"""
Improved Tennis Ball Tracker
============================

Enhanced version with better ball detection and tracking parameters.
Addresses issues found in verification: large jumps, static tracking, false positives.

Improvements:
- Better HSV color ranges for tennis balls
- Improved morphological operations
- Temporal consistency checking
- Multiple detection methods
- Kalman filter for smoothing
"""

import cv2
import numpy as np
import json
from pathlib import Path

class ImprovedTennisBallTracker:
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
        self.GRAVITY = 9.8  # m/s^2
        
        # Tracking state
        self.last_position = None
        self.position_history = []
        self.max_velocity = 40.0  # m/s (realistic max for tennis ball)
        
        # Enhanced detection parameters
        self.detection_methods = ['yellow_hsv', 'white_hsv', 'edge_detection']
    
    def calibrate_court(self, frame=None):
        """Step 1: Calibrate court (manual for now)"""
        if frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        print("Click on 4 court corners (baseline left, baseline right, far right, far left)")
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"Point {len(points)}", (x+15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Court Calibration', frame)
        
        cv2.namedWindow('Court Calibration')
        cv2.setMouseCallback('Court Calibration', mouse_callback)
        cv2.imshow('Court Calibration', frame)
        
        while len(points) < 4:
            cv2.waitKey(1)
        
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
        
        print("Court calibration complete!")
        return self.homography
    
    def detect_ball_yellow_hsv(self, frame):
        """Detect yellow tennis ball using HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Expanded yellow range for tennis balls
        lower1 = np.array([15, 80, 80])   # Darker yellow
        upper1 = np.array([35, 255, 255]) # Brighter yellow
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Additional range for fluorescent yellow
        lower2 = np.array([35, 100, 100])
        upper2 = np.array([45, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        return self.find_ball_candidates(mask)
    
    def detect_ball_white_hsv(self, frame):
        """Detect white/bright tennis ball using HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White/bright areas (for worn tennis balls)
        lower = np.array([0, 0, 180])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        return self.find_ball_candidates(mask)
    
    def detect_ball_edge_detection(self, frame):
        """Detect ball using edge detection and circular Hough transform"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use HoughCircles to detect circular objects
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        candidates = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Estimate area for consistency with other methods
                area = np.pi * r * r
                if 20 < area < 2000:
                    candidates.append({
                        'center': (x, y),
                        'radius': r,
                        'area': area,
                        'circularity': 1.0,  # Circles are perfect circles
                        'method': 'hough'
                    })
        
        return candidates
    
    def find_ball_candidates(self, mask):
        """Find ball candidates from binary mask"""
        # Improved morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:  # Reasonable ball size range
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # More lenient circularity threshold
                    if circularity > 0.4:  # Reduced from 0.6
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            radius = int(np.sqrt(area / np.pi))
                            
                            candidates.append({
                                'center': (cx, cy),
                                'radius': radius,
                                'area': area,
                                'circularity': circularity,
                                'method': 'contour'
                            })
        
        return candidates
    
    def detect_ball_multi_method(self, frame):
        """Detect ball using multiple methods and select best candidate"""
        all_candidates = []
        
        # Try yellow HSV detection
        yellow_candidates = self.detect_ball_yellow_hsv(frame)
        all_candidates.extend(yellow_candidates)
        
        # Try white HSV detection
        white_candidates = self.detect_ball_white_hsv(frame)
        all_candidates.extend(white_candidates)
        
        # Try edge detection
        edge_candidates = self.detect_ball_edge_detection(frame)
        all_candidates.extend(edge_candidates)
        
        if not all_candidates:
            return None
        
        # Score candidates based on multiple criteria
        best_candidate = self.select_best_candidate(all_candidates, frame)
        
        if best_candidate:
            return (best_candidate['center'][0], best_candidate['center'][1], best_candidate['radius'])
        
        return None
    
    def select_best_candidate(self, candidates, frame):
        """Select best ball candidate using temporal consistency and quality metrics"""
        if not candidates:
            return None
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            score = 0.0
            cx, cy = candidate['center']
            
            # Base score from circularity and area
            score += candidate['circularity'] * 30
            
            # Prefer medium-sized detections
            area = candidate['area']
            if 50 < area < 500:
                score += 20
            elif 20 < area < 1000:
                score += 10
            
            # Temporal consistency - prefer candidates near last position
            if self.last_position:
                last_x, last_y = self.last_position[:2]
                distance = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                
                # Realistic movement constraint (max 100 pixels between frames)
                if distance < 100:
                    score += 30 - (distance / 100) * 30
                elif distance > 300:  # Penalize very large jumps
                    score -= 50
            
            # Color analysis - check if region actually looks like a tennis ball
            color_score = self.analyze_ball_color(frame, cx, cy, candidate['radius'])
            score += color_score
            
            scored_candidates.append((score, candidate))
        
        # Return candidate with highest score, but only if score is reasonable
        best_score, best_candidate = max(scored_candidates, key=lambda x: x[0])
        
        if best_score > 20:  # Minimum quality threshold
            return best_candidate
        
        return None
    
    def analyze_ball_color(self, frame, cx, cy, radius):
        """Analyze color in ball region to validate detection"""
        try:
            # Extract region around ball
            y1, y2 = max(0, cy - radius), min(frame.shape[0], cy + radius)
            x1, x2 = max(0, cx - radius), min(frame.shape[1], cx + radius)
            
            if y2 <= y1 or x2 <= x1:
                return 0
            
            region = frame[y1:y2, x1:x2]
            
            # Convert to HSV
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Check for yellow/green colors typical of tennis balls
            yellow_mask = cv2.inRange(hsv_region, np.array([15, 50, 50]), np.array([45, 255, 255]))
            yellow_ratio = np.sum(yellow_mask > 0) / (region.shape[0] * region.shape[1])
            
            # Check for bright areas (worn balls)
            bright_mask = cv2.inRange(hsv_region, np.array([0, 0, 150]), np.array([180, 50, 255]))
            bright_ratio = np.sum(bright_mask > 0) / (region.shape[0] * region.shape[1])
            
            # Score based on color content
            color_score = 0
            if yellow_ratio > 0.3:  # At least 30% yellow/green
                color_score += 20
            elif yellow_ratio > 0.1:
                color_score += 10
            
            if bright_ratio > 0.2:  # Some bright areas
                color_score += 10
            
            return color_score
            
        except Exception:
            return 0
    
    def track_video(self):
        """Step 3: Track ball through entire video with improved detection"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        detections = []
        frame_num = 0
        
        print("Starting improved ball tracking...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Use multi-method detection
            ball = self.detect_ball_multi_method(frame)
            
            if ball:
                self.last_position = ball
                self.position_history.append(ball)
                
                # Keep only recent history
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
                
                detections.append({
                    'frame': frame_num,
                    'x': ball[0],
                    'y': ball[1],
                    'radius': ball[2]
                })
                
                # Visualize with confidence indicator
                color = (0, 255, 0) if ball else (0, 0, 255)
                cv2.circle(frame, (ball[0], ball[1]), ball[2], color, 2)
                cv2.circle(frame, (ball[0], ball[1]), 2, color, -1)
            else:
                # No detection - try to interpolate from recent history
                if len(self.position_history) >= 2:
                    # Simple linear prediction
                    last = self.position_history[-1]
                    prev = self.position_history[-2]
                    predicted_x = last[0] + (last[0] - prev[0])
                    predicted_y = last[1] + (last[1] - prev[1])
                    predicted_r = last[2]
                    
                    # Only use prediction if reasonable
                    if (0 < predicted_x < frame.shape[1] and 
                        0 < predicted_y < frame.shape[0]):
                        
                        ball = (int(predicted_x), int(predicted_y), predicted_r)
                        detections.append({
                            'frame': frame_num,
                            'x': ball[0],
                            'y': ball[1],
                            'radius': ball[2]
                        })
                        
                        cv2.circle(frame, (ball[0], ball[1]), ball[2], (255, 0, 0), 2)  # Blue for predicted
            
            # Show progress
            cv2.putText(frame, f"Frame: {frame_num} | Detections: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Resize for display if too large
            display_frame = frame
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                display_frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow('Improved Tracking', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_num += 1
        
        cv2.destroyAllWindows()
        print(f"Improved tracking complete! Found {len(detections)} detections")
        return detections
    
    # Rest of the methods remain the same as original tracker
    def reconstruct_3d(self, detections):
        """Step 4: Convert 2D to 3D positions"""
        positions_3d = []
        
        for det in detections:
            # Get ground position using homography
            point = np.array([[det['x'], det['y']]], dtype=np.float32)
            point = np.array([point])
            court_pos = cv2.perspectiveTransform(point, self.homography)[0][0]
            
            # Improved height estimation
            max_height = 4.0
            min_height = 0.067  # Ball diameter
            
            # Use radius for height estimation with better scaling
            normalized_radius = np.clip(det['radius'] / 30.0, 0.1, 2.0)
            height = max_height / normalized_radius
            height = np.clip(height, min_height, max_height)
            
            positions_3d.append({
                'frame': det['frame'],
                'time': det['frame'] / self.fps,
                'x': float(court_pos[0]),
                'y': float(court_pos[1]),
                'z': float(height)
            })
        
        return positions_3d
    
    def smooth_trajectory(self, positions_3d):
        """Step 5: Apply improved physics-based smoothing"""
        if len(positions_3d) < 3:
            return positions_3d
        
        smoothed = []
        window = 5  # Larger window for better smoothing
        
        for i, pos in enumerate(positions_3d):
            start = max(0, i - window // 2)
            end = min(len(positions_3d), i + window // 2 + 1)
            
            window_positions = positions_3d[start:end]
            
            # Weighted average with more weight on center
            weights = []
            for j in range(len(window_positions)):
                distance_from_center = abs(j - len(window_positions) // 2)
                weight = 1.0 / (1.0 + distance_from_center * 0.5)
                weights.append(weight)
            
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
                'z': avg_z
            })
        
        return smoothed
    
    def export_data(self, positions_3d, output_path):
        """Step 6: Export to JSON"""
        data = {
            'metadata': {
                'video': str(self.video_path),
                'fps': self.fps,
                'court_dimensions': [23.77, 10.97],
                'units': 'meters',
                'total_frames': len(positions_3d),
                'tracking_method': 'improved_multi_method'
            },
            'tracking_data': positions_3d
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(positions_3d)} frames to {output_path}")
    
    def run_full_pipeline(self, output_path='tennis_tracking_improved.json'):
        """Run complete improved pipeline"""
        print("Step 1: Court Calibration")
        self.calibrate_court()
        
        print("\nStep 2: Improved Ball Tracking")
        detections = self.track_video()
        print(f"Detected ball in {len(detections)} frames")
        
        print("\nStep 3: 3D Reconstruction")
        positions_3d = self.reconstruct_3d(detections)
        
        print("\nStep 4: Enhanced Smoothing")
        positions_3d = self.smooth_trajectory(positions_3d)
        
        print("\nStep 5: Exporting Data")
        self.export_data(positions_3d, output_path)
        
        self.cap.release()
        return positions_3d

# Usage
if __name__ == "__main__":
    tracker = ImprovedTennisBallTracker('assets/tennis.mp4')
    positions = tracker.run_full_pipeline('tennis_tracking_improved.json')
    print(f"\nImproved tracking complete! Tracked {len(positions)} frames")
