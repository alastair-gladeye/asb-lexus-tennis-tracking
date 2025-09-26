"""
Blender Tennis Ball Import - 720p TrackNet Data
==============================================

Import script specifically designed for the 720p TrackNet tracking data.
This script imports the high-quality ball tracking from tracking_verification_720p.mp4.

Usage in Blender:
1. Open Blender (any version 3.0+)
2. Switch to Scripting workspace (top menu)
3. Click "New" to create new text file
4. Copy and paste this entire script
5. Click "Run Script" ▶️

The script will automatically:
- Load tennis_tracking_720p.json
- Create a tennis ball with realistic material
- Animate the ball using the tracking data
- Set up proper court dimensions and lighting
- Configure the scene for optimal viewing
"""

import bpy
import json
import bmesh
from mathutils import Vector
import os

def ensure_object_mode():
    """Ensure we're in object mode"""
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass

def safe_clear_scene():
    """Safely clear existing mesh objects"""
    ensure_object_mode()
    
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    # Delete them manually
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    print(f"✅ Cleared {len(mesh_objects)} objects from scene")

def load_720p_tracking_data():
    """Load the 720p TrackNet tracking data"""
    
    # List of possible file locations
    possible_files = [
        'tennis_tracking_720p.json',
        '../tennis_tracking_720p.json', 
        'tracking_data/tennis_tracking_720p.json',
        os.path.expanduser('~/tennis_tracking_720p.json')
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded tracking data from: {filepath}")
                return data
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")
                continue
    
    # If not found, show helpful message
    print("❌ Could not find tennis_tracking_720p.json")
    print("📁 Please ensure the file is in one of these locations:")
    for filepath in possible_files:
        print(f"   - {filepath}")
    
    return None

def create_tennis_court_720p():
    """Create tennis court reference scaled for 720p coordinates"""
    
    # Tennis court dimensions (official)
    court_length = 23.77  # meters
    court_width = 10.97   # meters
    
    # Create court base
    try:
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
        court = bpy.context.active_object
        court.name = "TennisCourt_720p"
        
        # Scale to proper court size
        court.scale = (court_length, court_width, 1)
        
        # Apply scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Create court material
        mat = bpy.data.materials.new(name="CourtMaterial_720p")
        mat.use_nodes = True
        
        # Set court color (blue-green)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.2, 0.4, 0.8, 1.0)  # Court blue
        bsdf.inputs[7].default_value = 0.8  # Roughness
        
        court.data.materials.append(mat)
        
        print("✅ Tennis court created (23.77m x 10.97m)")
        return court
        
    except Exception as e:
        print(f"❌ Court creation failed: {e}")
        return None

def create_tennis_ball_720p():
    """Create tennis ball with realistic material"""
    
    try:
        # Create sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.033, location=(0, 0, 1))
        ball = bpy.context.active_object
        ball.name = "TennisBall_720p"
        
        # Create tennis ball material
        mat = bpy.data.materials.new(name="TennisBallMaterial_720p")
        mat.use_nodes = True
        
        # Set tennis ball color (bright yellow-green)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.8, 1.0, 0.2, 1.0)  # Tennis ball yellow
        bsdf.inputs[7].default_value = 0.3  # Roughness
        bsdf.inputs[6].default_value = 0.1  # Metallic
        
        ball.data.materials.append(mat)
        
        print("✅ Tennis ball created (regulation size: 6.6cm diameter)")
        return ball
        
    except Exception as e:
        print(f"❌ Ball creation failed: {e}")
        return None

def animate_ball_720p(ball, tracking_data):
    """Animate ball using 720p tracking data with proper coordinate conversion"""
    
    if not ball or not tracking_data:
        print("❌ Missing ball object or tracking data")
        return False
    
    # Get tracking points
    tracking_points = tracking_data.get('tracking_data', [])
    metadata = tracking_data.get('metadata', {})
    
    if not tracking_points:
        print("❌ No tracking points found in data")
        return False
    
    print(f"📊 Processing {len(tracking_points)} tracking points")
    
    # Video properties from metadata
    video_width = 1280  # 720p width
    video_height = 720  # 720p height
    fps = metadata.get('fps', 60)
    
    # Court dimensions for scaling
    court_length = 23.77  # meters
    court_width = 10.97   # meters
    
    # Clear existing keyframes
    ball.animation_data_clear()
    
    # Set up scene frame rate
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(tracking_points) + 10
    
    print(f"🎬 Animation: {len(tracking_points)} frames at {fps} FPS")
    
    # Convert tracking points to 3D positions
    for i, point in enumerate(tracking_points):
        frame_num = point['frame'] + 1  # Blender frames start at 1
        x_pixel = point['x']
        y_pixel = point['y']
        
        # Convert pixel coordinates to world coordinates
        # Map video coordinates (1280x720) to court coordinates
        x_world = (x_pixel / video_width - 0.5) * court_length
        y_world = ((video_height - y_pixel) / video_height - 0.5) * court_width
        z_world = 1.0  # Ball height above court
        
        # Set ball position
        ball.location = (x_world, y_world, z_world)
        
        # Insert keyframe
        ball.keyframe_insert(data_path="location", frame=frame_num)
        
        # Progress update
        if i % 50 == 0:
            print(f"   ⏳ Processed {i}/{len(tracking_points)} keyframes...")
    
    # Set interpolation to linear for smooth motion
    if ball.animation_data and ball.animation_data.action:
        for fcurve in ball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
    
    print(f"✅ Animation complete: {len(tracking_points)} keyframes")
    return True

def setup_lighting_720p():
    """Set up lighting optimized for 720p scene"""
    
    # Remove existing lights
    lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
    for light in lights:
        bpy.data.objects.remove(light, do_unlink=True)
    
    try:
        # Create main area light
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
        main_light = bpy.context.active_object
        main_light.name = "MainLight_720p"
        main_light.data.energy = 100
        main_light.data.size = 20
        
        # Create fill light
        bpy.ops.object.light_add(type='SUN', location=(10, 10, 5))
        fill_light = bpy.context.active_object
        fill_light.name = "FillLight_720p"
        fill_light.data.energy = 3
        fill_light.rotation_euler = (0.785, 0, 0.785)  # 45-degree angle
        
        print("✅ Lighting setup complete")
        
    except Exception as e:
        print(f"⚠️ Lighting setup issue: {e}")

def setup_camera_720p():
    """Set up camera for optimal 720p scene viewing"""
    
    try:
        # Get default camera or create one
        camera = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA':
                camera = obj
                break
        
        if not camera:
            bpy.ops.object.camera_add(location=(0, -15, 8))
            camera = bpy.context.active_object
        
        camera.name = "Camera_720p"
        camera.location = (0, -15, 8)
        camera.rotation_euler = (1.1, 0, 0)  # Look down at court
        
        # Set as active camera
        bpy.context.scene.camera = camera
        
        print("✅ Camera positioned for court overview")
        
    except Exception as e:
        print(f"⚠️ Camera setup issue: {e}")

def main():
    """Main import function for 720p TrackNet data"""
    
    print("=" * 60)
    print("🎾 BLENDER 720P TRACKNET IMPORT")
    print("=" * 60)
    
    # Step 1: Load tracking data
    print("\n📂 Step 1: Loading 720p tracking data...")
    tracking_data = load_720p_tracking_data()
    
    if not tracking_data:
        print("❌ Cannot continue without tracking data")
        print("💡 Make sure tennis_tracking_720p.json is accessible")
        return False
    
    # Display tracking info
    metadata = tracking_data.get('metadata', {})
    performance = tracking_data.get('performance', {})
    tracking_points = tracking_data.get('tracking_data', [])
    
    print(f"✅ Loaded tracking data:")
    print(f"   📺 Resolution: {metadata.get('resolution', 'Unknown')}")
    print(f"   🎬 Total frames: {metadata.get('total_frames', 'Unknown')}")
    print(f"   🎯 Detection rate: {performance.get('detection_rate', 0):.1%}")
    print(f"   📊 Tracking points: {len(tracking_points)}")
    
    # Step 2: Clear scene
    print("\n🧹 Step 2: Clearing scene...")
    safe_clear_scene()
    
    # Step 3: Create tennis court
    print("\n🏟️ Step 3: Creating tennis court...")
    court = create_tennis_court_720p()
    
    # Step 4: Create tennis ball
    print("\n🎾 Step 4: Creating tennis ball...")
    ball = create_tennis_ball_720p()
    
    # Step 5: Animate ball
    print("\n🎬 Step 5: Setting up ball animation...")
    success = animate_ball_720p(ball, tracking_data)
    
    if not success:
        print("❌ Animation setup failed")
        return False
    
    # Step 6: Setup lighting
    print("\n💡 Step 6: Setting up lighting...")
    setup_lighting_720p()
    
    # Step 7: Setup camera
    print("\n📹 Step 7: Setting up camera...")
    setup_camera_720p()
    
    # Step 8: Final configuration
    print("\n🎯 Step 8: Final configuration...")
    
    # Set viewport shading
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL_PREVIEW'
                        break
        print("✅ Viewport shading set to Material Preview")
    except:
        print("⚠️ Could not set viewport shading")
    
    # Set current frame to 1
    bpy.context.scene.frame_set(1)
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 720P IMPORT COMPLETE!")
    print("=" * 60)
    print(f"📊 Imported: {len(tracking_points)} ball positions")
    print(f"🎬 Animation: Frame 1-{len(tracking_points)}")
    print(f"🎯 Quality: {performance.get('detection_rate', 0):.1%} detection rate")
    print(f"📺 Source: 1280x720 TrackNet tracking")
    print("\n🎮 Controls:")
    print("   • Press SPACEBAR to play animation")
    print("   • Use mouse to orbit around the scene")
    print("   • Mouse wheel to zoom in/out")
    print("=" * 60)
    
    return True

# Run the import
if __name__ == "__main__":
    main()
