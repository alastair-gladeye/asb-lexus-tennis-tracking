# Tennis Ball 3D Tracker

A computer vision system for tracking tennis balls in 3D space from locked-off camera footage. This project enables real-time ball position tracking for video production workflows, allowing graphics to be superimposed into tennis footage using tools like Blender, TouchDesigner, or Unreal Engine.

## 🎾 Overview

This system processes tennis footage to:

1. **Calibrate** the tennis court in 3D space
2. **Detect** and track the tennis ball frame by frame
3. **Reconstruct** 3D ball positions using computer vision techniques
4. **Smooth** trajectories using physics-based algorithms
5. **Export** tracking data in JSON format for use in graphics software

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10 build 26100)
- Webcam or video files of tennis matches

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

3. **Place your tennis video** in the project directory and name it `tennis_video.mp4` (or modify the script)

4. **Run the tracker**:

   ```bash
   # Make sure virtual environment is activated
   tennis_venv\Scripts\activate

   # Run the tracking pipeline
   python tennis_tracker.py
   ```

## 📋 Usage

### Basic Usage

```python
from tennis_tracker import TennisBallTracker

# Initialize tracker with your video
tracker = TennisBallTracker('path/to/your/tennis_video.mp4')

# Run the complete pipeline
positions = tracker.run_full_pipeline('output_tracking_data.json')
```

### Step-by-Step Usage

```python
# Manual step-by-step process
tracker = TennisBallTracker('tennis_video.mp4')

# Step 1: Calibrate court (manual clicking on corners)
tracker.calibrate_court()

# Step 2: Track ball through video
detections = tracker.track_video()

# Step 3: Convert to 3D positions
positions_3d = tracker.reconstruct_3d(detections)

# Step 4: Apply smoothing
smooth_positions = tracker.smooth_trajectory(positions_3d)

# Step 5: Export data
tracker.export_data(smooth_positions, 'tennis_data.json')
```

## 🎯 Court Calibration

When you run the tracker, you'll need to manually calibrate the court:

1. A window will appear showing the first frame of your video
2. Click on **4 court corners** in this order:
   - Baseline left (near player)
   - Baseline right (near player)
   - Far baseline right (far player)
   - Far baseline left (far player)
3. The system will calculate the court's perspective transformation

**Tips for better calibration:**

- Use a frame where the court lines are clearly visible
- Click precisely on the corner intersections
- Ensure good lighting and contrast

## 📊 Output Data Format

The system exports tracking data as JSON with this structure:

```json
{
  "metadata": {
    "video": "tennis_video.mp4",
    "fps": 30.0,
    "court_dimensions": [23.77, 10.97],
    "units": "meters",
    "total_frames": 1250
  },
  "tracking_data": [
    {
      "frame": 0,
      "time": 0.0,
      "x": 11.885,
      "y": 5.485,
      "z": 1.2
    },
    ...
  ]
}
```

### Coordinate System

- **X**: Court length (0 to 23.77 meters)
- **Y**: Court width (0 to 10.97 meters)
- **Z**: Height above court (0+ meters)
- **Time**: Seconds from video start

## 🎨 Integration with Graphics Software

### Blender

```python
import json
import bpy

# Load tracking data
with open('tennis_tracking.json', 'r') as f:
    data = json.load(f)

# Create ball object and animate
for frame_data in data['tracking_data']:
    frame = frame_data['frame']
    x, y, z = frame_data['x'], frame_data['y'], frame_data['z']

    # Set keyframe for ball position
    bpy.context.scene.frame_set(frame)
    ball.location = (x, y, z)
    ball.keyframe_insert(data_path="location")
```

### TouchDesigner

The JSON data can be imported into TouchDesigner using:

- **File In DAT** → Load JSON file
- **Convert DAT** → Parse tracking data
- **Transform SOPs** → Apply positions to 3D objects

### Unreal Engine

Import via:

- **Data Table** asset with custom structure
- **Sequencer** for keyframe animation
- **Blueprint** scripts for real-time tracking

## ⚙️ Configuration

### Ball Detection Parameters

Modify these in `tennis_tracker.py`:

```python
# HSV color range for yellow tennis ball
lower = np.array([20, 100, 100])  # Lower HSV bound
upper = np.array([40, 255, 255])  # Upper HSV bound

# Size filtering
min_area = 20    # Minimum contour area
max_area = 2000  # Maximum contour area

# Circularity threshold
circularity_threshold = 0.6  # 0.0 to 1.0
```

### Court Dimensions

Standard tennis court (already configured):

- **Length**: 23.77 meters (78 feet)
- **Width**: 10.97 meters (36 feet)

## 🔧 Troubleshooting

### Common Issues

**Ball not detected:**

- Adjust HSV color ranges for your specific ball/lighting
- Check ball size parameters (min/max area)
- Ensure sufficient contrast between ball and background

**Poor 3D reconstruction:**

- Improve court calibration accuracy
- Use a frame with clear court line visibility
- Check camera angle (avoid extreme perspectives)

**Tracking jumps/noise:**

- Increase smoothing window size
- Implement more sophisticated physics models
- Filter out low-confidence detections

**Virtual environment issues:**

```bash
# If activation fails, try:
tennis_venv\Scripts\Activate.ps1  # PowerShell
# or
tennis_venv\Scripts\activate.bat  # Command Prompt
```

## 📁 Project Structure

```
asb_lexus/
├── tennis_tracker.py      # Main tracking script
├── requirements.txt       # Python dependencies
├── setup.bat             # Setup script
├── README.md             # This file
├── tennis_venv/          # Virtual environment (created by setup)
└── output/               # Generated tracking data
    ├── tennis_tracking.json
    └── debug_frames/
```

## 🛠️ Development

### Dependencies

- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific computing (future physics improvements)
- **Matplotlib**: Plotting and visualization (debugging)

### Future Enhancements

- [ ] Machine learning ball detection (YOLO/detectron2)
- [ ] Multi-ball tracking for doubles matches
- [ ] Real-time processing capabilities
- [ ] Automatic court line detection
- [ ] Physics-based trajectory prediction
- [ ] GUI interface for easier use

## 📝 License

This project is developed for ASB Lexus video production workflows. Contact the development team for usage rights and modifications.

## 🤝 Support

For technical support or feature requests:

1. Check the troubleshooting section above
2. Review the configuration options
3. Contact the video production team

---

**Happy Tracking! 🎾📹**
