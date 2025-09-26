"""
Tennis TrackNet Adapter
======================

Adapts TrackNet for your tennis ball tracking workflow.
Processes your tennis video and outputs data compatible with Blender.
"""

import sys
import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path

# Add TrackNet to path
sys.path.append('TrackNet')

try:
    from model import BallTrackerNet as TrackNet
    # Import general utilities if they exist
    try:
        from general import *
    except ImportError:
        pass  # general.py might not have what we need
except ImportError:
    print("❌ TrackNet modules not found. Make sure TrackNet is cloned and in the right directory.")
    sys.exit(1)

class TennisTrackNetProcessor:
    def __init__(self, model_path='models/tracknet_model.pt', video_path='assets/tennis.mp4'):
        self.model_path = model_path
        self.video_path = video_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = TrackNet()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ TrackNet model loaded from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            print("Please download the pre-trained model first")
    
    def process_video(self, output_path='tennis_tracking_tracknet.json'):
        """Process tennis video with TrackNet"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {self.video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Processing video: {total_frames} frames at {fps} FPS")
        
        detections = []
        frame_buffer = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for TrackNet (typically 512x288)
            input_frame = cv2.resize(frame, (512, 288))
            frame_buffer.append(input_frame)
            
            # TrackNet uses 3 consecutive frames
            if len(frame_buffer) >= 3:
                # Prepare input tensor - concatenate 3 frames into 9 channels
                input_frames = np.stack(frame_buffer[-3:], axis=0)  # (3, H, W, 3)
                input_frames = input_frames.transpose(0, 3, 1, 2)  # (3, 3, H, W)
                input_frames = input_frames.reshape(-1, input_frames.shape[2], input_frames.shape[3])  # (9, H, W)
                input_tensor = torch.FloatTensor(input_frames).unsqueeze(0)  # (1, 9, H, W)
                input_tensor = input_tensor.to(self.device) / 255.0
                
                # Run inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                    heatmap = output.squeeze().cpu().numpy()
                
                # Find ball position from heatmap
                ball_pos = self.extract_ball_position(heatmap, frame.shape)
                
                if ball_pos:
                    detections.append({
                        'frame': int(frame_num),
                        'x': float(ball_pos[0]),
                        'y': float(ball_pos[1]), 
                        'confidence': float(ball_pos[2])
                    })
                
                # Visualize (optional)
                if ball_pos:
                    cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 10, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show progress
                if frame_num % 100 == 0:
                    print(f"⏳ Processed {frame_num}/{total_frames} frames...")
            
            frame_num += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ TrackNet processing complete: {len(detections)} detections")
        
        # Save results in Blender-compatible format
        self.save_blender_format(detections, fps, output_path)
        return detections
    
    def extract_ball_position(self, heatmap, original_shape):
        """Extract ball position from TrackNet heatmap"""
        # Find maximum response in heatmap
        max_val = np.max(heatmap)
        
        if max_val < 0.5:  # Confidence threshold
            return None
        
        # Find peak location
        max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Scale back to original frame coordinates
        scale_x = original_shape[1] / heatmap.shape[1]
        scale_y = original_shape[0] / heatmap.shape[0]
        
        x = max_loc[1] * scale_x
        y = max_loc[0] * scale_y
        
        return (x, y, max_val)
    
    def save_blender_format(self, detections, fps, output_path):
        """Save tracking data in Blender-compatible format"""
        # Note: This still needs court calibration for 3D conversion
        # For now, save 2D positions that can be manually calibrated
        
        data = {
            'metadata': {
                'video': str(self.video_path),
                'fps': float(fps),
                'total_frames': len(detections),
                'tracking_method': 'TrackNet_deep_learning',
                'note': '2D positions - requires court calibration for 3D'
            },
            'tracking_data_2d': detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 TrackNet results saved to: {output_path}")

def main():
    print("🎾 TENNIS TRACKNET PROCESSOR")
    print("=" * 40)
    
    # Check if model exists
    model_path = 'models/tracknet_model.pt'
    if not os.path.exists(model_path):
        print(f"❌ Pre-trained model not found: {model_path}")
        print("Please download the model from Google Drive first")
        print("See models/model_download_info.json for instructions")
        return
    
    # Process video
    processor = TennisTrackNetProcessor(model_path, 'assets/tennis.mp4')
    detections = processor.process_video('tennis_tracking_tracknet.json')
    
    if detections:
        print(f"✅ SUCCESS: TrackNet tracked {len(detections)} frames")
        print("🔄 Next: Integrate with court calibration for 3D positions")
    else:
        print("❌ No detections found")

if __name__ == "__main__":
    main()
