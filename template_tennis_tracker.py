"""
Template-based Tennis Ball Tracker
==================================

Uses template matching for robust tennis ball tracking.
This approach works with any OpenCV version and avoids false positives.

Key features:
- Manual ball selection on first frame
- Template matching for consistent tracking
- Search region optimization
- Temporal consistency validation
- No dependency on specific OpenCV tracker modules
"""

import cv2
import numpy as np
import json
from pathlib import Path

class TemplateTennisBallTracker:
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
        
        # Template tracking state
        self.ball_template = None
        self.template_size = (40, 40)  # Template size
        self.last_position = None
        self.search_radius = 80  # Search radius around last position
        self.confidence_threshold = 0.6
    
    def calibrate_court(self, frame=None):
        """Court calibration with improved UI"""
        if frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        print("🏟️ COURT CALIBRATION")
        print("Click 4 court corners in order:")
        print("1. Baseline left (near camera)")
        print("2. Baseline right (near camera)")  
        print("3. Far baseline right")
        print("4. Far baseline left")
        
        points = []
        temp_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                temp_frame = frame.copy()
                
                # Draw all points and connections
                for i, point in enumerate(points):
                    cv2.circle(temp_frame, tuple(point), 8, (0, 255, 0), -1)
                    cv2.putText(temp_frame, f"{i+1}", (point[0]+15, point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw court outline
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(temp_frame, tuple(points[i]), tuple(points[i+1]), (255, 0, 0), 2)
                    if len(points) == 4:
                        cv2.line(temp_frame, tuple(points[3]), tuple(points[0]), (255, 0, 0), 2)
                
                cv2.imshow('Court Calibration', temp_frame)
        
        cv2.namedWindow('Court Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Court Calibration', 1280, 720)
        cv2.setMouseCallback('Court Calibration', mouse_callback)
        cv2.imshow('Court Calibration', frame)
        
        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Calculate homography
        points = np.array(points, dtype=np.float32)
        court_2d = self.court_3d[:, :2]
        self.homography, _ = cv2.findHomography(points, court_2d)
        
        print("✅ Court calibration complete!")
        return self.homography
    
    def select_ball_template(self, frame):
        """Select tennis ball and create template"""
        print("\n🎾 BALL TEMPLATE SELECTION")
        print("Click and drag to select the tennis ball")
        print("Make the selection as tight as possible around the ball")
        print("Press ENTER to confirm, 'r' to reset, 'q' to quit")
        
        selecting = False
        start_point = None
        end_point = None
        temp_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selecting, start_point, end_point, temp_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_point = (x, y)
                end_point = None
            
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
                cv2.putText(temp_frame, "Selecting ball template", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Ball Template Selection', temp_frame)
            
            elif event == cv2.EVENT_LBUTTONUP and selecting:
                selecting = False
                end_point = (x, y)
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(temp_frame, "Press ENTER to confirm, 'r' to reset", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Ball Template Selection', temp_frame)
        
        cv2.namedWindow('Ball Template Selection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ball Template Selection', 1280, 720)
        cv2.setMouseCallback('Ball Template Selection', mouse_callback)
        
        # Add instructions to the frame
        instruction_frame = frame.copy()
        cv2.putText(instruction_frame, "BALL SELECTION: Click and drag around the tennis ball", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(instruction_frame, "Press ENTER to confirm, 'r' to reset, 'q' to quit", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(instruction_frame, "Make sure this window is active/focused!", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Ball Template Selection', instruction_frame)
        
        print("🖱️ Make sure the 'Ball Template Selection' window is active")
        print("👆 Click on the window first, then drag around the tennis ball")
        print("⌨️ Press ENTER when done, 'r' to reset, 'q' to quit")
        
        # Wait a moment for user to see instructions
        cv2.waitKey(2000)  # 2 second delay
        
        while True:
            key = cv2.waitKey(30) & 0xFF  # Longer wait time
            
            if key == ord('r'):  # Reset
                start_point = None
                end_point = None
                temp_frame = frame.copy()
                cv2.imshow('Ball Template Selection', temp_frame)
            
            elif key == 13:  # ENTER
                if start_point and end_point:
                    break
                else:
                    print("Please select a region first")
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None, None
        
        cv2.destroyAllWindows()
        
        # Extract template
        x1, y1 = start_point
        x2, y2 = end_point
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            print("❌ Template too small, please select a larger region")
            return None, None
        
        # Extract and resize template
        template = frame[y1:y2, x1:x2]
        self.ball_template = cv2.resize(template, self.template_size)
        
        # Calculate initial position
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = max(x2 - x1, y2 - y1) // 2
        
        self.last_position = (center_x, center_y, radius)
        
        print(f"✅ Ball template created: {self.template_size[0]}x{self.template_size[1]} pixels")
        return self.ball_template, self.last_position
    
    def find_ball_template_match(self, frame):
        """Find ball using template matching"""
        if self.ball_template is None:
            return None
        
        # Define search region around last known position
        if self.last_position:
            center_x, center_y, radius = self.last_position
            
            # Calculate search bounds
            search_x1 = max(0, center_x - self.search_radius)
            search_y1 = max(0, center_y - self.search_radius)
            search_x2 = min(frame.shape[1], center_x + self.search_radius)
            search_y2 = min(frame.shape[0], center_y + self.search_radius)
            
            # Extract search region
            search_region = frame[search_y1:search_y2, search_x1:search_x2]
            
            if search_region.shape[0] < self.template_size[1] or search_region.shape[1] < self.template_size[0]:
                # Search region too small, use full frame
                search_region = frame
                search_x1, search_y1 = 0, 0
        else:
            # No previous position, search full frame
            search_region = frame
            search_x1, search_y1 = 0, 0
        
        # Perform template matching
        result = cv2.matchTemplate(search_region, self.ball_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if match is good enough
        if max_val < self.confidence_threshold:
            return None
        
        # Convert match location to full frame coordinates
        match_x = search_x1 + max_loc[0] + self.template_size[0] // 2
        match_y = search_y1 + max_loc[1] + self.template_size[1] // 2
        
        # Estimate radius (use template size as reference)
        estimated_radius = max(self.template_size) // 2
        
        return (match_x, match_y, estimated_radius, max_val)
    
    def track_ball_through_video(self):
        """Track ball through entire video using template matching"""
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get first frame for template selection
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Could not read first frame")
            return []
        
        # Select ball template
        template, initial_pos = self.select_ball_template(first_frame)
        if template is None:
            print("❌ Ball template selection failed")
            return []
        
        # Reset to beginning for tracking
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        detections = []
        frame_num = 0
        lost_frames = 0
        max_lost_frames = 5
        
        print("\n🎬 Starting template-based tracking...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Find ball using template matching
            if frame_num == 0:
                # Use initial position for first frame
                ball_result = (initial_pos[0], initial_pos[1], initial_pos[2], 1.0)
            else:
                ball_result = self.find_ball_template_match(frame)
            
            # Create visualization
            display_frame = frame.copy()
            
            if ball_result:
                center_x, center_y, radius, confidence = ball_result
                
                # Update tracking state
                self.last_position = (center_x, center_y, radius)
                lost_frames = 0
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green = high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow = medium confidence
                else:
                    color = (0, 165, 255)  # Orange = low confidence
                
                # Draw detection
                cv2.circle(display_frame, (center_x, center_y), radius, color, 2)
                cv2.circle(display_frame, (center_x, center_y), 3, color, -1)
                
                # Draw search region
                if self.last_position:
                    lx, ly, lr = self.last_position
                    search_rect = (
                        max(0, lx - self.search_radius),
                        max(0, ly - self.search_radius),
                        min(frame.shape[1], lx + self.search_radius),
                        min(frame.shape[0], ly + self.search_radius)
                    )
                    cv2.rectangle(display_frame, (search_rect[0], search_rect[1]),
                                 (search_rect[2], search_rect[3]), (255, 0, 0), 1)
                
                # Store detection
                detections.append({
                    'frame': frame_num,
                    'x': center_x,
                    'y': center_y,
                    'radius': radius,
                    'confidence': confidence
                })
                
                # Display confidence
                cv2.putText(display_frame, f"Conf: {confidence:.2f}", 
                           (center_x + radius + 5, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            else:
                lost_frames += 1
                color = (0, 0, 255)  # Red for lost tracking
                
                # Try to extrapolate position for a few frames
                if self.last_position and lost_frames <= max_lost_frames:
                    lx, ly, lr = self.last_position
                    cv2.circle(display_frame, (lx, ly), lr, color, 2)
                    cv2.putText(display_frame, f"LOST ({lost_frames})", (lx + lr + 5, ly),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # If lost for too long, expand search
                if lost_frames > max_lost_frames:
                    self.search_radius = min(200, self.search_radius + 10)
            
            # Status display
            status = f"Frame: {frame_num} | Tracked: {len(detections)} | Lost: {lost_frames}"
            cv2.putText(display_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save template", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize for display if needed
            if display_frame.shape[1] > 1280:
                scale = 1280 / display_frame.shape[1]
                new_width = int(display_frame.shape[1] * scale)
                new_height = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            cv2.imshow('Template Tennis Ball Tracking', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Save current template
                if ball_result:
                    x, y, r, _ = ball_result
                    y1, y2 = max(0, y - r), min(frame.shape[0], y + r)
                    x1, x2 = max(0, x - r), min(frame.shape[1], x + r)
                    new_template = frame[y1:y2, x1:x2]
                    if new_template.shape[0] > 0 and new_template.shape[1] > 0:
                        self.ball_template = cv2.resize(new_template, self.template_size)
                        print(f"📷 Template updated at frame {frame_num}")
            
            frame_num += 1
        
        cv2.destroyAllWindows()
        
        print(f"✅ Template tracking complete!")
        print(f"📊 Tracked {len(detections)} frames out of {frame_num}")
        print(f"📈 Success rate: {len(detections)/frame_num*100:.1f}%")
        
        return detections
    
    def reconstruct_3d(self, detections):
        """Convert 2D detections to 3D coordinates"""
        positions_3d = []
        
        for det in detections:
            # Get ground position using homography
            point = np.array([[det['x'], det['y']]], dtype=np.float32)
            point = np.array([point])
            court_pos = cv2.perspectiveTransform(point, self.homography)[0][0]
            
            # Height estimation
            max_height = 4.0
            min_height = 0.067
            
            # Use radius and confidence for height estimation
            normalized_radius = np.clip(det['radius'] / 25.0, 0.3, 2.0)
            confidence_factor = det.get('confidence', 1.0)
            
            height = (max_height / normalized_radius) * confidence_factor
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
        """Smooth trajectory using confidence-weighted averaging"""
        if len(positions_3d) < 3:
            return positions_3d
        
        smoothed = []
        window = 3
        
        for i, pos in enumerate(positions_3d):
            start = max(0, i - window // 2)
            end = min(len(positions_3d), i + window // 2 + 1)
            
            window_positions = positions_3d[start:end]
            
            # Weight by confidence and distance from center
            weights = []
            for j, wp in enumerate(window_positions):
                confidence_weight = wp.get('confidence', 1.0)
                distance_weight = 1.0 / (1.0 + abs(j - len(window_positions) // 2))
                weights.append(confidence_weight * distance_weight)
            
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
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
        """Export tracking data to JSON"""
        data = {
            'metadata': {
                'video': str(self.video_path),
                'fps': self.fps,
                'court_dimensions': [23.77, 10.97],
                'units': 'meters',
                'total_frames': len(positions_3d),
                'tracking_method': 'template_matching',
                'template_size': self.template_size
            },
            'tracking_data': positions_3d
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Exported {len(positions_3d)} frames to {output_path}")
    
    def run_full_pipeline(self, output_path='tennis_tracking_template.json'):
        """Run complete template-based tracking pipeline"""
        print("🎾 TEMPLATE-BASED TENNIS BALL TRACKING")
        print("=" * 50)
        
        print("\nStep 1: Court Calibration")
        homography = self.calibrate_court()
        if homography is None:
            print("❌ Court calibration failed")
            return []
        
        print("\nStep 2: Template-Based Ball Tracking")
        detections = self.track_ball_through_video()
        if not detections:
            print("❌ No tracking data captured")
            return []
        
        print(f"\n✅ Tracked {len(detections)} frames")
        
        print("\nStep 3: 3D Reconstruction")
        positions_3d = self.reconstruct_3d(detections)
        
        print("\nStep 4: Trajectory Smoothing")
        positions_3d = self.smooth_trajectory(positions_3d)
        
        print("\nStep 5: Export Data")
        self.export_data(positions_3d, output_path)
        
        self.cap.release()
        
        print(f"\n🎾 Template tracking complete!")
        print(f"📊 Successfully tracked {len(positions_3d)} frames")
        print(f"📁 Data saved to: {output_path}")
        
        return positions_3d

# Usage
if __name__ == "__main__":
    tracker = TemplateTennisBallTracker('assets/tennis.mp4')
    positions = tracker.run_full_pipeline('tennis_tracking_template.json')
    
    if positions:
        print(f"\n✅ SUCCESS: {len(positions)} frames tracked!")
        print("This method should eliminate false positives by tracking the specific ball you selected.")
    else:
        print("\n❌ Tracking failed. Please try again with a clearer ball selection.")
