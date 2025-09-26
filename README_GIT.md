# 🎾 Tennis Ball Tracking System

Professional tennis ball tracking system using computer vision and deep learning for video production workflows.

## 🚀 Overview

This system provides **three levels of tennis ball tracking quality** for superimposing graphics into tennis footage:

| Method | Success Rate | Quality | Use Case |
|--------|-------------|---------|----------|
| **TrackNet Deep Learning** | 99.8% | Professional | **Production Ready** ✨ |
| Template Tracking | 87.6% | Good | Backup method |
| Basic Detection | Variable | Poor | Not recommended |

## 📁 Project Structure

```
tennis-ball-tracking/
├── 📊 TRACKING SCRIPTS
│   ├── tennis_tracker.py                    # Original detection method
│   ├── template_tennis_tracker.py           # Template-based tracking
│   ├── tennis_tracknet_adapter.py           # TrackNet deep learning
│   └── integrate_tracknet_3d.py             # 3D coordinate integration
├── 🎬 BLENDER INTEGRATION
│   ├── blender_tennis_import_fixed.py       # Import script for Blender
│   ├── blender_advanced_setup.py            # Advanced effects
│   └── BLENDER_FINAL_SETUP.md              # Complete guide
├── 🛠️ SETUP & UTILITIES
│   ├── setup_tracknet.py                   # TrackNet installation
│   ├── verify_tracking.py                  # Quality verification
│   ├── setup.bat                          # Environment setup
│   └── requirements.txt                    # Python dependencies
└── 📚 DOCUMENTATION
    ├── README.md                           # Main documentation
    ├── BLENDER_INTEGRATION_GUIDE.md        # Blender workflow
    └── BLENDER_FINAL_SETUP.md             # Final setup guide
```

## 🎯 Quick Start

### 1. Environment Setup
```bash
# Run automated setup
setup.bat

# OR manual setup
python -m venv tennis_venv
tennis_venv\Scripts\activate
pip install -r requirements.txt
```

### 2. TrackNet Deep Learning (Recommended)
```bash
# Install TrackNet
python setup_tracknet.py

# Download model from: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view
# Save as: models/tracknet_model.pt

# Run tracking
python tennis_tracknet_adapter.py
python integrate_tracknet_3d.py
```

### 3. Blender Integration
```bash
# In Blender Scripting workspace:
# Run: blender_tennis_import_fixed.py
# Imports: tennis_tracking_tracknet_3d.json (99.8% accuracy)
```

## 🔧 Dependencies

### Core Requirements
- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+

### TrackNet Deep Learning
- PyTorch 2.8+
- TorchVision 0.23+

### Graphics Integration
- Blender 3.0+ (for animation)

## 📊 Performance Comparison

Based on testing with tennis broadcast footage:

- **TrackNet**: 1,153/1,155 frames (99.8%) - Professional quality
- **Template**: 1,012/1,155 frames (87.6%) - Clean tracking
- **Detection**: 1,155/1,155 frames (100%) - Many false positives

## 🎬 Production Workflow

1. **Capture** → Tennis footage (locked-off camera)
2. **Track** → Run TrackNet deep learning tracking
3. **Calibrate** → 4-point court calibration
4. **Convert** → 2D to 3D coordinate transformation
5. **Import** → Blender animation with 99.8% accuracy
6. **Composite** → Graphics overlay for final delivery

## 🤖 TrackNet Deep Learning

Based on: [TrackNet Research Paper](https://arxiv.org/abs/1907.03698)

**Key Features:**
- Trained on 19,835 tennis ball frames
- Handles small, blurry, and occluded balls
- Generates confidence scores
- Professional broadcast quality

## 🎾 Technical Details

### Court Calibration
- Manual 4-point perspective correction
- Standard tennis court: 23.77m × 10.97m
- Real-world coordinate mapping

### 3D Reconstruction
- Homography-based ground positioning
- Height estimation from ball size
- Physics-based trajectory smoothing

### Blender Output
- Keyframe animation data
- Real-world coordinate system
- Frame-perfect synchronization

## 📈 Quality Assurance

Run verification to check tracking quality:
```bash
python verify_tracking.py
# Generates: tracking_verification.mp4
```

## 🔗 Integration

### Blender
- Automatic keyframe generation
- Material setup for tennis balls
- Camera and lighting configuration

### Other Software
- TouchDesigner: JSON data import
- Unreal Engine: Blueprint integration
- After Effects: Expression-based animation

## 🛡️ Version Control

Project uses Git with comprehensive `.gitignore`:
- Excludes large video files (*.mp4, *.avi)
- Excludes ML models (*.pt, *.pth)
- Excludes generated tracking data
- Includes all source code and documentation

## 📝 License

Developed for ASB Lexus video production workflows.

## 🤝 Contributing

For improvements or issues:
1. Check tracking quality with verification tools
2. Test with different tennis footage
3. Validate Blender integration workflow

---

**🎾 Professional Tennis Ball Tracking for Video Production** 📹✨
