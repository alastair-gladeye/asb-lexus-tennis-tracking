#!/usr/bin/env python3
"""
Example usage script for Tennis Ball 3D Tracker
This demonstrates how to use the tennis_tracker module for different scenarios.
"""

from tennis_tracker import TennisBallTracker
from pathlib import Path

def basic_tracking_example():
    """Basic example: Track a tennis video and export 3D data"""
    print("=== Basic Tennis Ball Tracking ===")
    
    # Check if tennis video exists
    video_path = "assets/tennis.mp4"  # Update this path to your video
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place your tennis video file in the project directory")
        print("Or update the video_path variable in this script")
        return
    
    # Initialize tracker
    tracker = TennisBallTracker(video_path)
    
    # Run complete pipeline
    positions = tracker.run_full_pipeline('tennis_tracking_output.json')
    
    print(f"Tracking complete! Processed {len(positions)} frames")
    print("Output saved to: tennis_tracking_output.json")

def step_by_step_example():
    """Step-by-step example with manual control"""
    print("\n=== Step-by-Step Tracking ===")
    
    video_path = "assets/tennis.mp4"
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return
    
    tracker = TennisBallTracker(video_path)
    
    # Step 1: Court calibration
    print("Step 1: Calibrating court...")
    tracker.calibrate_court()
    
    # Step 2: Track ball through video
    print("Step 2: Tracking ball...")
    detections = tracker.track_video()
    print(f"Found {len(detections)} ball detections")
    
    # Step 3: Convert to 3D
    print("Step 3: Converting to 3D positions...")
    positions_3d = tracker.reconstruct_3d(detections)
    
    # Step 4: Apply smoothing
    print("Step 4: Smoothing trajectory...")
    smoothed_positions = tracker.smooth_trajectory(positions_3d)
    
    # Step 5: Export data
    print("Step 5: Exporting data...")
    tracker.export_data(smoothed_positions, 'detailed_tracking_output.json')
    
    print("Step-by-step tracking complete!")

def analyze_tracking_data():
    """Analyze exported tracking data"""
    import json
    import numpy as np
    
    output_file = 'tennis_tracking_output.json'
    if not Path(output_file).exists():
        print(f"No tracking data found: {output_file}")
        print("Run basic_tracking_example() first")
        return
    
    print("\n=== Tracking Data Analysis ===")
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    tracking_data = data['tracking_data']
    
    if not tracking_data:
        print("No tracking data found in file")
        return
    
    # Calculate some statistics
    positions = np.array([[p['x'], p['y'], p['z']] for p in tracking_data])
    
    print(f"Total tracked frames: {len(tracking_data)}")
    print(f"Video duration: {tracking_data[-1]['time']:.2f} seconds")
    print(f"Average FPS: {data['metadata']['fps']}")
    
    print("\nPosition statistics:")
    print(f"X range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} meters")
    print(f"Y range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} meters")
    print(f"Z range: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f} meters")
    print(f"Max height: {positions[:, 2].max():.2f} meters")
    
    # Calculate approximate ball speed
    if len(tracking_data) > 1:
        distances = []
        for i in range(1, len(tracking_data)):
            prev_pos = np.array([tracking_data[i-1]['x'], tracking_data[i-1]['y'], tracking_data[i-1]['z']])
            curr_pos = np.array([tracking_data[i]['x'], tracking_data[i]['y'], tracking_data[i]['z']])
            dist = np.linalg.norm(curr_pos - prev_pos)
            time_diff = tracking_data[i]['time'] - tracking_data[i-1]['time']
            if time_diff > 0:
                speed = dist / time_diff  # m/s
                distances.append(speed)
        
        if distances:
            avg_speed = np.mean(distances)
            max_speed = np.max(distances)
            print(f"\nBall speed statistics:")
            print(f"Average speed: {avg_speed:.2f} m/s ({avg_speed * 3.6:.1f} km/h)")
            print(f"Maximum speed: {max_speed:.2f} m/s ({max_speed * 3.6:.1f} km/h)")

if __name__ == "__main__":
    print("Tennis Ball 3D Tracker - Example Usage")
    print("=" * 50)
    
    # Check if we have a video file
    video_exists = Path("assets/tennis.mp4").exists()
    
    if video_exists:
        print("Found tennis video file!")
        
        # Run basic example
        basic_tracking_example()
        
        # Analyze the results
        analyze_tracking_data()
        
    else:
        print("No tennis video found.")
        print("\nTo use this tracker:")
        print("1. Place your tennis video file as 'assets/tennis.mp4'")
        print("2. Or update the video_path in the example functions")
        print("3. Run this script again")
        
    print("\nFor more advanced usage, see the README.md file")
