@echo off
echo ========================================
echo Tennis Ball 3D Tracker - Setup Script
echo ========================================
echo.

echo [1/4] Creating Python virtual environment...
python -m venv tennis_venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python 3.8+ is installed and in PATH
    pause
    exit /b 1
)
echo ✓ Virtual environment created successfully

echo.
echo [2/4] Activating virtual environment...
call tennis_venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

echo.
echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and requirements.txt
    pause
    exit /b 1
)
echo ✓ Dependencies installed successfully

echo.
echo [4/4] Verifying installation...
python -c "import cv2, numpy as np; print('OpenCV version:', cv2.__version__); print('NumPy version:', np.__version__)"
if %errorlevel% neq 0 (
    echo ERROR: Installation verification failed
    pause
    exit /b 1
)
echo ✓ Installation verified

echo.
echo ========================================
echo Setup Complete! 🎾
echo ========================================
echo.
echo Next steps:
echo 1. Place your tennis video file in this directory
echo 2. Run: tennis_venv\Scripts\activate.bat
echo 3. Run: python tennis_tracker.py
echo.
echo For detailed usage instructions, see README.md
echo.
pause
