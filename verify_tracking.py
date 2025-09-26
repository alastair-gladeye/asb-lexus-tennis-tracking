"""
Tennis Ball Tracking Verification Script
========================================

This script creates a video overlay showing the tracking results on top of the original video.
Use this to verify tracking accuracy and identify issues before importing to Blender.

Features:
- Shows tracked ball position as colored circle
- Displays frame number and tracking confidence
- Creates output video for easy review
- Highlights tracking issues and jumps
"""

import cv2
import numpy as np
import json
from pathlib import Path

class TrackingVerifier:
    def __init__(self, video_path, tracking_json_path):
        self.video_path = video_path
        self.tracking_json_path = tracking_json_path
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Load tracking data
        with open(tracking_json_path, 'r') as f:
            self.tracking_data = json.load(f)
        
        self.positions = {pos['frame']: pos for pos in self.tracking_data['tracking_data']}
        
        print(f"Video: {video_path}")
        print(f"Tracking data: {len(self.tracking_data['tracking_data'])} points")
    
    def detect_tracking_issues(self):
        """Analyze tracking data for potential issues"""
        positions = self.tracking_data['tracking_data']
        issues = []
        
        # Check for large jumps between frames
        for i in range(1, len(positions)):
            prev = positions[i-1]
            curr = positions[i]
            
            # Calculate 2D distance jump (we'll need to reverse the homography)
            # For now, just check 3D position jumps
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dz = curr['z'] - prev['z']
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Flag large jumps (tennis ball shouldn't move >5m between frames)
            if distance > 5.0:
                issues.append({
                    'frame': curr['frame'],
                    'type': 'large_jump',
                    'distance': distance,
                    'from': (prev['x'], prev['y'], prev['z']),
                    'to': (curr['x'], curr['y'], curr['z'])
                })
        
        # Check for static positions (ball stuck in same place)
        static_threshold = 0.1  # meters
        static_count = 0
        for i in range(1, len(positions)):
            prev = positions[i-1]
            curr = positions[i]
            
            dx = abs(curr['x'] - prev['x'])
            dy = abs(curr['y'] - prev['y'])
            dz = abs(curr['z'] - prev['z'])
            
            if dx < static_threshold and dy < static_threshold and dz < static_threshold:
                static_count += 1
            else:
                if static_count > 10:  # More than 10 frames stuck
                    issues.append({
                        'frame': positions[i-static_count]['frame'],
                        'type': 'static_tracking',
                        'duration': static_count
                    })
                static_count = 0
        
        return issues
    
    def create_verification_video(self, output_path='tracking_verification.mp4'):
        """Create video with tracking overlay"""
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating verification video: {output_path}")
        print(f"Video properties: {width}x{height} at {fps} FPS")
        
        # Detect issues first
        issues = self.detect_tracking_issues()
        issue_frames = {issue['frame']: issue for issue in issues}
        
        if issues:
            print(f"⚠️ Found {len(issues)} potential tracking issues:")
            for issue in issues[:5]:  # Show first 5 issues
                if issue['type'] == 'large_jump':
                    print(f"  Frame {issue['frame']}: Large jump ({issue['distance']:.1f}m)")
                elif issue['type'] == 'static_tracking':
                    print(f"  Frame {issue['frame']}: Static for {issue['duration']} frames")
        
        frame_num = 0
        trail_positions = []  # Store recent positions for trail effect
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Create overlay
            overlay = frame.copy()
            
            # Get tracking data for this frame
            if frame_num in self.positions:
                pos_data = self.positions[frame_num]
                
                # We need to reverse the 3D->2D projection to get pixel coordinates
                # For now, let's run the ball detection on this frame to get 2D position
                ball_2d = self.detect_ball_in_frame(frame)
                
                if ball_2d:
                    x, y, radius = ball_2d
                    
                    # Add to trail
                    trail_positions.append((x, y))
                    if len(trail_positions) > 20:  # Keep last 20 positions
                        trail_positions.pop(0)
                    
                    # Draw trail
                    for i, (tx, ty) in enumerate(trail_positions[:-1]):
                        alpha = i / len(trail_positions)
                        color = (0, int(255 * alpha), int(255 * (1-alpha)))  # Blue to red gradient
                        cv2.circle(overlay, (int(tx), int(ty)), 3, color, -1)
                    
                    # Check if this frame has issues
                    color = (0, 255, 0)  # Green = good tracking
                    thickness = 2
                    
                    if frame_num in issue_frames:
                        issue = issue_frames[frame_num]
                        if issue['type'] == 'large_jump':
                            color = (0, 0, 255)  # Red = large jump
                            thickness = 4
                        elif issue['type'] == 'static_tracking':
                            color = (0, 255, 255)  # Yellow = static
                            thickness = 3
                    
                    # Draw ball detection
                    cv2.circle(overlay, (x, y), radius, color, thickness)
                    cv2.circle(overlay, (x, y), 3, color, -1)  # Center dot
                    
                    # Draw 3D position info
                    info_text = f"3D: ({pos_data['x']:.1f}, {pos_data['y']:.1f}, {pos_data['z']:.1f})"
                    cv2.putText(overlay, info_text, (x + radius + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                else:
                    # No ball detected in this frame, but we have tracking data
                    cv2.putText(overlay, "TRACKING WITHOUT DETECTION", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Frame info
            cv2.putText(overlay, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Legend
            cv2.putText(overlay, "Green=Good, Red=Jump, Yellow=Static", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(overlay)
            
            # Progress
            if frame_num % 50 == 0:
                print(f"Processed frame {frame_num}")
            
            frame_num += 1
        
        # Cleanup
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Verification video created: {output_path}")
        return issues
    
    def detect_ball_in_frame(self, frame):
        """Detect ball in current frame (same as main tracker)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow tennis ball - you may need to adjust these values
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphology
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_circularity = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:
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
    
    def generate_detection_report(self):
        """Generate a detailed report of tracking quality"""
        issues = self.detect_tracking_issues()
        
        print("\n" + "="*50)
        print("📊 TRACKING QUALITY REPORT")
        print("="*50)
        
        total_frames = len(self.tracking_data['tracking_data'])
        print(f"Total tracked frames: {total_frames}")
        print(f"Issues found: {len(issues)}")
        print(f"Quality score: {((total_frames - len(issues)) / total_frames * 100):.1f}%")
        
        if issues:
            print(f"\n⚠️ ISSUES DETECTED:")
            jump_issues = [i for i in issues if i['type'] == 'large_jump']
            static_issues = [i for i in issues if i['type'] == 'static_tracking']
            
            if jump_issues:
                print(f"• {len(jump_issues)} large jumps detected")
                print("  Largest jumps:")
                sorted_jumps = sorted(jump_issues, key=lambda x: x['distance'], reverse=True)
                for jump in sorted_jumps[:3]:
                    print(f"    Frame {jump['frame']}: {jump['distance']:.1f}m jump")
            
            if static_issues:
                print(f"• {len(static_issues)} static tracking periods")
                print("  Longest static periods:")
                sorted_static = sorted(static_issues, key=lambda x: x['duration'], reverse=True)
                for static in sorted_static[:3]:
                    print(f"    Frame {static['frame']}: {static['duration']} frames static")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if len(issues) > total_frames * 0.1:  # More than 10% issues
            print("• Consider adjusting ball detection parameters")
            print("• Check lighting and ball color in problematic frames")
            print("• May need manual tracking correction")
        else:
            print("• Tracking quality is good")
            print("• Minor smoothing may improve results")
        
        return issues

def main():
    """Main verification function"""
    print("🎾 Tennis Ball Tracking Verification")
    print("="*40)
    
    # Check for required files
    video_path = "assets/tennis.mp4"
    tracking_path = "tennis_tracking.json"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not Path(tracking_path).exists():
        print(f"❌ Tracking data not found: {tracking_path}")
        return
    
    # Create verifier
    verifier = TrackingVerifier(video_path, tracking_path)
    
    # Generate quality report
    issues = verifier.generate_detection_report()
    
    # Create verification video
    print(f"\n🎬 Creating verification video...")
    verifier.create_verification_video("tracking_verification.mp4")
    
    print(f"\n✅ Verification complete!")
    print(f"📁 Check 'tracking_verification.mp4' to see tracking overlay")
    print(f"🔍 Look for:")
    print(f"  • Green circles = good tracking")
    print(f"  • Red circles = large jumps")
    print(f"  • Yellow circles = static tracking")
    print(f"  • Blue-to-red trails = ball path")

if __name__ == "__main__":
    main()
