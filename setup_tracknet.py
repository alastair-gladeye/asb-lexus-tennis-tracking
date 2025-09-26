"""
TrackNet Setup and Integration Script
====================================

This script helps you set up TrackNet for professional tennis ball tracking.
TrackNet is a deep learning approach specifically designed for tennis ball tracking
in broadcast videos, offering superior accuracy to template matching.

Steps:
1. Clone TrackNet repository
2. Install PyTorch dependencies
3. Download pre-trained model
4. Adapt for your video processing pipeline
5. Generate tracking data compatible with Blender
"""

import os
import subprocess
import json
import urllib.request
from pathlib import Path

class TrackNetSetup:
    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.tracknet_dir = self.project_dir / "TrackNet"
        self.models_dir = self.project_dir / "models"
        
    def clone_tracknet(self):
        """Clone the TrackNet repository"""
        print("📥 Cloning TrackNet repository...")
        
        if self.tracknet_dir.exists():
            print(f"✅ TrackNet already exists in {self.tracknet_dir}")
            return True
        
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/yastrebksv/TrackNet.git",
                str(self.tracknet_dir)
            ], check=True)
            
            # Remove git repository from TrackNet to avoid nested git repos
            tracknet_git_dir = self.tracknet_dir / ".git"
            if tracknet_git_dir.exists():
                import shutil
                shutil.rmtree(tracknet_git_dir)
                print("🧹 Removed TrackNet git repository (keeping as dependency folder)")
            
            print("✅ TrackNet cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone TrackNet: {e}")
            return False
    
    def install_dependencies(self):
        """Install TrackNet dependencies"""
        print("📦 Installing TrackNet dependencies...")
        
        # Create requirements for TrackNet
        tracknet_requirements = [
            "torch>=1.9.0",
            "torchvision>=0.10.0", 
            "opencv-python>=4.5.0",
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "matplotlib>=3.3.0",
            "Pillow>=8.0.0",
            "tqdm>=4.60.0"
        ]
        
        # Write requirements file
        req_file = self.tracknet_dir / "requirements_extended.txt"
        with open(req_file, 'w') as f:
            f.write('\n'.join(tracknet_requirements))
        
        try:
            # Install in current virtual environment
            subprocess.run([
                "tennis_venv\\Scripts\\pip.exe", "install", "-r", str(req_file)
            ], check=True, cwd=self.project_dir)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def download_pretrained_model(self):
        """Download pre-trained TrackNet model"""
        print("🤖 Setting up pre-trained model download...")
        
        self.models_dir.mkdir(exist_ok=True)
        
        # Note: The actual model download requires Google Drive authentication
        # We'll provide instructions for manual download
        model_info = {
            "model_url": "https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing",
            "model_file": "tracknet_model.pth",
            "instructions": [
                "1. Visit the Google Drive link above",
                "2. Download the model file", 
                "3. Save it as 'tracknet_model.pth' in the models/ directory",
                "4. The model is pre-trained on tennis ball tracking data"
            ]
        }
        
        model_info_file = self.models_dir / "model_download_info.json"
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"📝 Model download instructions saved to: {model_info_file}")
        print("🌐 Manual download required from Google Drive")
        return True
    
    def create_tennis_tracknet_adapter(self):
        """Create adapter script to use TrackNet with your tennis footage"""
        adapter_script = '''"""
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
    from model import TrackNet
    from general import *
except ImportError:
    print("❌ TrackNet modules not found. Make sure TrackNet is cloned and in the right directory.")
    sys.exit(1)

class TennisTrackNetProcessor:
    def __init__(self, model_path='models/tracknet_model.pth', video_path='assets/tennis.mp4'):
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
                # Prepare input tensor
                input_frames = np.stack(frame_buffer[-3:], axis=0)
                input_tensor = torch.FloatTensor(input_frames).permute(0, 3, 1, 2).unsqueeze(0)
                input_tensor = input_tensor.to(self.device) / 255.0
                
                # Run inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                    heatmap = output.squeeze().cpu().numpy()
                
                # Find ball position from heatmap
                ball_pos = self.extract_ball_position(heatmap, frame.shape)
                
                if ball_pos:
                    detections.append({
                        'frame': frame_num,
                        'x': ball_pos[0],
                        'y': ball_pos[1], 
                        'confidence': ball_pos[2]
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
                'fps': fps,
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
    model_path = 'models/tracknet_model.pth'
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
'''
        
        adapter_file = self.project_dir / "tennis_tracknet_adapter.py"
        with open(adapter_file, 'w', encoding='utf-8') as f:
            f.write(adapter_script)
        
        print(f"✅ TrackNet adapter created: {adapter_file}")
        return True
    
    def run_setup(self):
        """Run complete TrackNet setup"""
        print("🎾 SETTING UP TRACKNET FOR TENNIS BALL TRACKING")
        print("=" * 60)
        
        steps = [
            ("Cloning TrackNet Repository", self.clone_tracknet),
            ("Installing Dependencies", self.install_dependencies), 
            ("Setting up Pre-trained Model", self.download_pretrained_model),
            ("Creating Tennis Adapter", self.create_tennis_tracknet_adapter)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 TRACKNET SETUP COMPLETE!")
        print("=" * 60)
        print("\n📋 NEXT STEPS:")
        print("1. Download pre-trained model from Google Drive (see models/model_download_info.json)")
        print("2. Run: python tennis_tracknet_adapter.py")
        print("3. Integrate with your court calibration")
        print("4. Import to Blender with superior tracking quality!")
        print("\n🚀 TrackNet will provide professional-grade tennis ball tracking!")
        
        return True

if __name__ == "__main__":
    setup = TrackNetSetup()
    setup.run_setup()
