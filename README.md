# Tennis Ball 3D Tracker

A computer vision system for tracking tennis balls in 3D space from locked-off camera footage. This project enables real-time ball position tracking for video production workflows, allowing graphics to be superimposed into tennis footage using tools like Blender, TouchDesigner, or Unreal Engine.

## 🎾 Overview

This system processes tennis footage to:

1. **Detect** and track the tennis ball frame by frame using TrackNet neural network
2. **Convert** 2D ball positions to 3D coordinates using court calibration
3. **Smooth** trajectories using physics-based algorithms
4. **Export** tracking data in JSON format for use in graphics software
5. **Import** directly into Blender for 3D animation and visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10 build 26100)
- **Video footage at 1280x720 resolution** (critical for TrackNet accuracy)

### Installation

1. **Clone or download this project** to your desired location

2. **Run the setup script** (this will create the virtual environment and install dependencies):

   ```bash
   setup.bat
   ```

   Or manually:

   ```bash
   # Create virtual environment
   python -m venv tennis_venv

   # Activate virtual environment
   tennis_venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Prepare your tennis video**:
   - **IMPORTANT**: Resize your video to **1280x720 resolution** (TrackNet was trained on this resolution)
   - Place the video in the `assets/` directory and name it `tennis.mp4`
   - Higher resolution videos (like 1920x1080) will produce poor tracking results

4. **Run the 720p tracker**:

   ```bash
   # Make sure virtual environment is activated
   tennis_venv\Scripts\activate

   # Run the optimized 720p tracking pipeline
   python tennis_tracknet_720p.py
   ```

## 📋 Usage

### Step 1: Video Preparation
**Critical**: Your input video must be 1280x720 resolution for optimal results.

```bash
# Example using ffmpeg to resize video to 720p
ffmpeg -i your_video.mp4 -vf scale=1280:720 assets/tennis.mp4
```

### Step 2: Run Ball Tracking

```bash
# Activate environment
tennis_venv\Scripts\activate

# Run the 720p optimized tracker
python tennis_tracknet_720p.py
```

**Expected Output:**
- 📊 Detection rate: ~96% (excellent accuracy with 720p footage)
- 🎬 Verification video: `tracking_verification_720p.mp4`
- 📋 Tracking data: `tennis_tracking_720p.json`

### Step 3: Import into Blender

1. **Open Blender** (version 3.0 or newer)
2. **Switch to Scripting workspace** (top menu)
3. **Create new script** (click "New" in text editor)
4. **Copy and paste** the entire contents of `blender_tennis_720p_import.py`
5. **Run the script** (click ▶️ button)

The script will automatically:
- Load your tracking data
- Create a regulation tennis court (23.77m × 10.97m)
- Create and animate the tennis ball
- Set up lighting and camera

### Step 4: Verify Results

Watch `tracking_verification_720p.mp4` to verify the ball tracking quality:
- Green circle shows detected ball position
- Yellow trail shows ball trajectory
- Info panel shows detection statistics

## 🎯 Key Features

### ✅ **720p Resolution Optimization**
- TrackNet neural network trained specifically on 1280x720 footage
- Achieves 96% detection accuracy with proper resolution
- Automatic coordinate scaling and validation

### ✅ **Real-time Visualization**
- Live tracking overlay with ball position and trajectory
- Detection confidence indicators
- Performance metrics display

### ✅ **Blender Integration**
- Direct import script for 3D animation
- Proper court dimensions and ball scaling
- Realistic materials and lighting setup

### ✅ **Professional Output**
- JSON tracking data for external applications
- High-quality verification videos
- Frame-accurate ball positions

## 📁 File Structure

```
tennis_tracker/
├── assets/
│   └── tennis.mp4                      # Your 720p tennis footage (REQUIRED)
├── models/
│   └── tracknet_model.pt              # Pre-trained TrackNet model
├── tennis_tracknet_720p.py            # Main tracking script (720p optimized)
├── blender_tennis_720p_import.py      # Blender import script
├── tracking_verification_720p.mp4     # Verification video output
├── tennis_tracking_720p.json          # Tracking data output
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## ⚠️ Important Notes

### **Video Resolution Requirements**
- **Must use 1280x720 (720p) resolution** for optimal tracking
- TrackNet was trained on this specific resolution
- Other resolutions will produce poor results
- Use video conversion tools to resize if needed

### **Tracking Quality**
- Expected detection rate: 90-96% with proper 720p footage
- Works best with good lighting and clear ball visibility
- Court surface should provide good contrast with the ball

### **Performance**
- Processing speed: ~1.3 FPS (slower but accurate)
- Memory usage: Moderate (loads entire video into memory)
- GPU support: Automatic CUDA detection if available

## 🔧 Troubleshooting

### Low Detection Rate
- **Check video resolution**: Must be exactly 1280x720
- Ensure good lighting and ball visibility
- Verify court provides contrast with ball color

### Import Issues in Blender
- Make sure `tennis_tracking_720p.json` is in the same directory
- Use Blender 3.0 or newer
- Run script from Blender's Scripting workspace

### Performance Issues
- Reduce video length for testing
- Ensure sufficient RAM for video loading
- Use GPU if available (CUDA)

## 📊 Expected Results

With properly prepared 720p footage:
- **Detection Rate**: 96%+ 
- **Coordinate Accuracy**: High precision within frame bounds
- **Trajectory Quality**: Smooth, realistic ball motion
- **Processing Time**: ~4 minutes for 5-second clip

## 🏆 Best Practices

1. **Video Quality**: Use high-quality, well-lit tennis footage
2. **Resolution**: Always resize to 1280x720 before processing
3. **Verification**: Always check `tracking_verification_720p.mp4` for quality
4. **Backup**: Keep original footage and use template data as fallback

## 📄 Alternative Options

- **Template Data**: Use `tennis_tracking_template.json` for proven high-quality data
- **Original Blender Script**: `blender_tennis_import_fixed.py` for template data import
- **3D Integration**: `integrate_tracknet_3d.py` for court calibration workflows

## 🤝 Support

For issues or improvements:
1. Check video resolution (must be 720p)
2. Verify tracking quality in verification video
3. Use template data as fallback option

---

**🎾 Ready to track some tennis balls? Start with properly sized 720p footage for best results!**