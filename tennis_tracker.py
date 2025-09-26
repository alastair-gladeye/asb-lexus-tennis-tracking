import cv2
import numpy as np
import json
from pathlib import Path

class TennisBallTracker:
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
    
    def detect_ball(self, frame):
        """Step 2: Detect ball in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow tennis ball
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphology to clean up
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_circularity = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:  # Size filter
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity > max_circularity and circularity > 0.6:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            radius = int(np.sqrt(area / np.pi))
                            best_ball = (cx, cy, radius)
                            max_circularity = circularity
        
        return best_ball
    
    def track_video(self):
        """Step 3: Track ball through entire video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        detections = []
        frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            ball = self.detect_ball(frame)
            
            if ball:
                detections.append({
                    'frame': frame_num,
                    'x': ball[0],
                    'y': ball[1],
                    'radius': ball[2]
                })
                
                # Visualize
                cv2.circle(frame, (ball[0], ball[1]), ball[2], (0, 255, 0), 2)
            
            # Show progress
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_num += 1
        
        cv2.destroyAllWindows()
        return detections
    
    def reconstruct_3d(self, detections):
        """Step 4: Convert 2D to 3D positions"""
        positions_3d = []
        
        for det in detections:
            # Get ground position using homography
            point = np.array([[det['x'], det['y']]], dtype=np.float32)
            point = np.array([point])
            court_pos = cv2.perspectiveTransform(point, self.homography)[0][0]
            
            # Estimate height from ball size
            # Larger radius = closer = lower height
            # This is simplified - you may want more sophisticated method
            max_height = 4.0
            normalized_radius = det['radius'] / 50.0  # Adjust scale
            height = max_height * (1.0 - normalized_radius)
            height = np.clip(height, 0.1, max_height)
            
            positions_3d.append({
                'frame': det['frame'],
                'time': det['frame'] / self.fps,
                'x': float(court_pos[0]),
                'y': float(court_pos[1]),
                'z': float(height)
            })
        
        return positions_3d
    
    def smooth_trajectory(self, positions_3d):
        """Step 5: Apply physics-based smoothing"""
        # Simple moving average for now
        # You can implement more sophisticated physics here
        window = 3
        smoothed = []
        
        for i, pos in enumerate(positions_3d):
            start = max(0, i - window // 2)
            end = min(len(positions_3d), i + window // 2 + 1)
            
            window_positions = positions_3d[start:end]
            avg_x = np.mean([p['x'] for p in window_positions])
            avg_y = np.mean([p['y'] for p in window_positions])
            avg_z = np.mean([p['z'] for p in window_positions])
            
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
                'total_frames': len(positions_3d)
            },
            'tracking_data': positions_3d
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(positions_3d)} frames to {output_path}")
    
    def run_full_pipeline(self, output_path='tracking_data.json'):
        """Run complete pipeline"""
        print("Step 1: Court Calibration")
        self.calibrate_court()
        
        print("\nStep 2: Tracking Ball")
        detections = self.track_video()
        print(f"Detected ball in {len(detections)} frames")
        
        print("\nStep 3: 3D Reconstruction")
        positions_3d = self.reconstruct_3d(detections)
        
        print("\nStep 4: Smoothing Trajectory")
        positions_3d = self.smooth_trajectory(positions_3d)
        
        print("\nStep 5: Exporting Data")
        self.export_data(positions_3d, output_path)
        
        self.cap.release()
        return positions_3d

# Usage
if __name__ == "__main__":
    tracker = TennisBallTracker('assets/tennis.mp4')
    positions = tracker.run_full_pipeline('tennis_tracking.json')
    print(f"\nComplete! Tracked {len(positions)} frames")