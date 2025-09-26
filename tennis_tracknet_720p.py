"""
TrackNet Tennis Ball Tracker - Optimized for 720p
===============================================

Optimized TrackNet implementation for 1280x720 video resolution,
matching the training data dimensions for maximum accuracy.
"""

import sys
import os
import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm

# Add TrackNet to path
sys.path.append('TrackNet')

try:
    from model import BallTrackerNet
    from general import postprocess
except ImportError:
    print("❌ TrackNet modules not found. Make sure TrackNet is cloned and in the right directory.")
    sys.exit(1)

class TrackNet720pTracker:
    def __init__(self, model_path='models/tracknet_model.pt', video_path='assets/tennis.mp4'):
        self.model_path = model_path
        self.video_path = video_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TrackNet model input dimensions (as used in training)
        self.model_height = 360
        self.model_width = 640
        
        # Visual settings
        self.ball_color = (0, 255, 0)      # Green for current position
        self.trace_color = (0, 255, 255)   # Yellow for trajectory trace
        self.text_color = (255, 255, 255)  # White for text
        self.trace_length = 20              # Number of previous positions to show
        
        # Tracking history
        self.ball_positions = []
        self.confidence_scores = []
        
        print(f"🎾 TrackNet 720p Tracker initialized")
        print(f"📱 Device: {self.device}")
        print(f"📐 Model Input: {self.model_width}x{self.model_height}")
        print(f"📺 Video optimized for: 1280x720 (TrackNet training resolution)")
        
        # Initialize model
        self.model = BallTrackerNet()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ TrackNet model loaded from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
    
    def preprocess_frames_720p(self, frames):
        """Preprocess 3 consecutive frames for TrackNet - optimized for 720p input"""
        
        # Resize all frames to model input size (640x360)
        resized_frames = []
        for frame in frames:
            # Since input is 1280x720, we can resize more precisely
            # 1280x720 -> 640x360 is exactly 2:1 scaling
            resized = cv2.resize(frame, (self.model_width, self.model_height))
            resized_frames.append(resized)
        
        # Concatenate frames as TrackNet expects: current, previous, pre-previous
        imgs = np.concatenate(resized_frames, axis=2)  # Stack along channel dimension
        imgs = imgs.astype(np.float32) / 255.0         # Normalize to [0,1]
        imgs = np.rollaxis(imgs, 2, 0)                 # HWC -> CHW
        input_tensor = np.expand_dims(imgs, axis=0)    # Add batch dimension
        
        return torch.from_numpy(input_tensor).float().to(self.device)
    
    def detect_ball_720p(self, input_tensor, original_frame_shape):
        """Detect ball using TrackNet with proper 720p scaling"""
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Use argmax to get the predicted class for each pixel
            feature_map = output.argmax(dim=1).detach().cpu().numpy()
            
            # Use the original postprocessing function with proper scaling
            x_pred, y_pred = postprocess(feature_map[0], scale=2)
            
            if x_pred is not None and y_pred is not None:
                # Scale coordinates from model output (640x360) back to original frame (1280x720)
                # The postprocess function already applies scale=2, so we get coordinates for 1280x720
                x_final = int(x_pred)
                y_final = int(y_pred)
                
                # Validate coordinates are within frame bounds
                if 0 <= x_final < original_frame_shape[1] and 0 <= y_final < original_frame_shape[0]:
                    return (x_final, y_final), 1.0
        
        return None, 0.0
    
    def draw_720p_overlay(self, frame, current_pos, confidence, frame_num):
        """Draw enhanced tracking visualization optimized for 720p"""
        height, width = frame.shape[:2]
        
        # Draw trajectory trace with smooth fading
        if len(self.ball_positions) > 1:
            for i in range(max(0, len(self.ball_positions) - self.trace_length), len(self.ball_positions) - 1):
                if self.ball_positions[i] and self.ball_positions[i + 1]:
                    age = len(self.ball_positions) - i - 1
                    alpha = max(0.2, 1.0 - (age / self.trace_length))
                    thickness = max(2, int(6 * alpha))
                    trace_color = tuple(int(c * alpha) for c in self.trace_color)
                    cv2.line(frame, self.ball_positions[i], self.ball_positions[i + 1], trace_color, thickness)
        
        # Draw current ball position with enhanced visibility
        if current_pos:
            # Main detection circle (scaled for 720p)
            cv2.circle(frame, current_pos, 15, self.ball_color, 3)
            cv2.circle(frame, current_pos, 8, self.ball_color, -1)
            
            # Outer ring for visibility
            cv2.circle(frame, current_pos, 20, self.ball_color, 2)
            
            # Crosshair for precision
            cv2.line(frame, (current_pos[0] - 25, current_pos[1]), 
                    (current_pos[0] + 25, current_pos[1]), self.ball_color, 2)
            cv2.line(frame, (current_pos[0], current_pos[1] - 25), 
                    (current_pos[0], current_pos[1] + 25), self.ball_color, 2)
        
        # Info panel (scaled for 720p)
        panel_height = 160
        panel_width = 350
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
        
        # Text information (smaller font for 720p)
        y_offset = 30
        font_scale = 0.6
        cv2.putText(frame, f"Frame: {frame_num}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, 2)
        
        y_offset += 25
        if current_pos:
            cv2.putText(frame, f"Ball: ({current_pos[0]}, {current_pos[1]})", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, 2)
        else:
            cv2.putText(frame, "Ball: Not Detected", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, 2)
        
        # Detection statistics
        if len(self.ball_positions) > 0:
            detections = sum(1 for pos in self.ball_positions if pos is not None)
            rate = detections / len(self.ball_positions)
            y_offset += 25
            cv2.putText(frame, f"Detection Rate: {rate:.1%}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, 2)
        
        # Model info
        y_offset += 25
        cv2.putText(frame, "TrackNet 720p Optimized", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        
        # Status indicator
        status = "TRACKING" if current_pos else "SEARCHING"
        status_color = self.ball_color if current_pos else (0, 0, 255)
        cv2.putText(frame, status, (width - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        return frame
    
    def process_video_720p(self, output_video_path='tracking_verification_720p.mp4',
                          output_json_path='tennis_tracking_720p.json',
                          display_video=False):
        """Process 720p video with optimized TrackNet"""
        
        # Open video and load all frames
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {self.video_path}")
            return False
        
        # Verify resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Processing {total_frames} frames at {fps} FPS")
        print(f"📐 Video Resolution: {width}x{height}")
        
        if width != 1280 or height != 720:
            print(f"⚠️ Warning: Video is not 1280x720! TrackNet was trained on this resolution.")
        
        # Load all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"✅ Loaded {len(frames)} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        detections = []
        start_time = time.time()
        
        # Process frames
        with tqdm(total=len(frames), desc="🎾 TrackNet 720p") as pbar:
            for frame_num in range(len(frames)):
                current_pos = None
                confidence = 0.0
                
                # Need at least 3 frames for TrackNet
                if frame_num >= 2:
                    # Prepare 3 consecutive frames
                    frame_triplet = [
                        frames[frame_num],     # Current frame
                        frames[frame_num-1],   # Previous frame  
                        frames[frame_num-2]    # Pre-previous frame
                    ]
                    
                    # Preprocess frames
                    input_tensor = self.preprocess_frames_720p(frame_triplet)
                    
                    # Detect ball
                    current_pos, confidence = self.detect_ball_720p(input_tensor, frames[frame_num].shape)
                    
                    if current_pos:
                        detections.append({
                            'frame': frame_num,
                            'x': current_pos[0],
                            'y': current_pos[1],
                            'confidence': float(confidence)
                        })
                
                # Update history
                self.ball_positions.append(current_pos)
                self.confidence_scores.append(confidence)
                
                # Create visualization
                frame_with_overlay = self.draw_720p_overlay(
                    frames[frame_num].copy(), current_pos, confidence, frame_num)
                
                out.write(frame_with_overlay)
                
                if display_video:
                    cv2.imshow('TrackNet 720p Tracking', cv2.resize(frame_with_overlay, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
        
        out.release()
        if display_video:
            cv2.destroyAllWindows()
        
        # Save results
        processing_time = time.time() - start_time
        detection_count = len(detections)
        detection_rate = detection_count / max(1, total_frames - 2)
        
        tracking_data = {
            'metadata': {
                'video': str(self.video_path),
                'output_video': str(output_video_path),
                'fps': fps,
                'total_frames': total_frames,
                'resolution': f"{width}x{height}",
                'processing_time_seconds': processing_time,
                'tracking_method': 'TrackNet_720p_optimized',
                'model_input_size': f"{self.model_width}x{self.model_height}",
                'resolution_match': width == 1280 and height == 720
            },
            'performance': {
                'detections': detection_count,
                'detection_rate': detection_rate,
                'processing_fps': total_frames / processing_time if processing_time > 0 else 0
            },
            'tracking_data': detections
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"\n🎉 720P TRACKNET COMPLETE!")
        print(f"📊 Detections: {detection_count}/{total_frames-2} ({detection_rate:.1%})")
        print(f"📁 Video: {output_video_path}")
        print(f"📋 Data: {output_json_path}")
        
        return True

def main():
    print("🎾 TRACKNET 720P OPTIMIZED TRACKER")
    print("=" * 50)
    
    tracker = TrackNet720pTracker()
    success = tracker.process_video_720p()
    
    if success:
        print("\n✅ Processing complete! Check 'tracking_verification_720p.mp4'")
        print("🎯 This should be much more accurate with proper 720p resolution!")
    
    return success

if __name__ == "__main__":
    main()
