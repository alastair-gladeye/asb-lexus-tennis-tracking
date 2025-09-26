"""
Fixed Blender Tennis Ball Tracking Import Script
===============================================

This is a robust version that handles Blender context issues and operator failures.
Use this version if you encountered errors with the original script.

Usage:
1. Open Blender
2. Switch to Scripting workspace  
3. Load this script or paste it into the text editor
4. Make sure tennis_tracking.json is accessible
5. Run the script

Features:
- Robust error handling for different Blender versions
- Manual mesh creation fallbacks
- Context-aware operations
- Improved compatibility
"""

import bpy
import json
import bmesh
from mathutils import Vector
import os

def ensure_object_mode():
    """Ensure we're in object mode before running operators"""
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass  # Some contexts don't allow mode changes

def safe_clear_scene():
    """Safely clear existing mesh objects"""
    ensure_object_mode()
    
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    # Delete them manually (more reliable than operators)
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    print(f"Cleared {len(mesh_objects)} mesh objects")

def create_tennis_ball_robust():
    """Create tennis ball with fallback methods"""
    ensure_object_mode()
    
    # Method 1: Try operator
    try:
        bpy.ops.mesh.uv_sphere_add(radius=0.0335, location=(0, 0, 0))
        ball = bpy.context.active_object
        ball.name = "TennisBall"
        print("✅ Ball created with operator")
    except:
        # Method 2: Manual creation with bmesh
        print("⚠️ Operator failed, creating ball manually...")
        mesh = bpy.data.meshes.new("TennisBall")
        ball = bpy.data.objects.new("TennisBall", mesh)
        bpy.context.collection.objects.link(ball)
        
        # Create sphere with bmesh
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.0335)
        bm.to_mesh(mesh)
        bm.free()
        print("✅ Ball created manually")
    
    # Create material
    mat = bpy.data.materials.new(name="TennisBallMaterial")
    mat.use_nodes = True
    
    # Set up material nodes
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Add principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.8, 0.9, 0.2, 1.0)  # Tennis ball yellow
    principled.inputs['Roughness'].default_value = 0.3
    
    # Add output node
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material
    ball.data.materials.append(mat)
    
    return ball

def create_tennis_court_robust():
    """Create tennis court with fallback methods"""
    ensure_object_mode()
    
    # Method 1: Try operator
    try:
        bpy.ops.mesh.plane_add(size=1, location=(11.885, 5.485, 0))
        court = bpy.context.active_object
        court.name = "TennisCourt"
        court.scale = (23.77, 10.97, 1)
        print("✅ Court created with operator")
    except:
        # Method 2: Manual creation
        print("⚠️ Operator failed, creating court manually...")
        mesh = bpy.data.meshes.new("TennisCourt")
        court = bpy.data.objects.new("TennisCourt", mesh)
        bpy.context.collection.objects.link(court)
        
        # Create plane with bmesh
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1)
        bm.to_mesh(mesh)
        bm.free()
        
        # Position and scale
        court.location = (11.885, 5.485, 0)
        court.scale = (23.77, 10.97, 1)
        print("✅ Court created manually")
    
    # Create court material
    mat = bpy.data.materials.new(name="CourtMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.2, 0.6, 0.2, 1.0)  # Green court
    principled.inputs['Roughness'].default_value = 0.8
    
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    court.data.materials.append(mat)
    return court

def load_tracking_data_robust(filepath=None):
    """Load tracking data with multiple path attempts"""
    possible_paths = []
    
    if filepath:
        possible_paths.append(filepath)
    
    # Try multiple common locations (prioritize TrackNet deep learning)
    possible_paths.extend([
        "tennis_tracking_tracknet_3d.json",  # TrackNet deep learning (BEST quality - 99.8%)
        "tennis_tracking_template.json",     # Template tracking (good quality - 87.6%)
        "tennis_tracking.json",              # Original detection (poor quality)
        bpy.path.abspath("//tennis_tracking_tracknet_3d.json"),
        bpy.path.abspath("//tennis_tracking_template.json"),
        bpy.path.abspath("//tennis_tracking.json"),
        os.path.join(os.path.dirname(bpy.data.filepath), "tennis_tracking_tracknet_3d.json"),
        os.path.join(os.path.dirname(bpy.data.filepath), "tennis_tracking_template.json"),
        os.path.join(os.path.dirname(bpy.data.filepath), "tennis_tracking.json"),
        os.path.join(os.getcwd(), "tennis_tracking_tracknet_3d.json"),
        os.path.join(os.getcwd(), "tennis_tracking_template.json"),
        os.path.join(os.getcwd(), "tennis_tracking.json")
    ])
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded tracking data from: {path}")
                return data
        except Exception as e:
            print(f"⚠️ Failed to load from {path}: {e}")
            continue
    
    print("❌ Could not find tennis_tracking.json in any expected location")
    print("Make sure the file is in the same directory as your .blend file")
    return None

def animate_ball_robust(ball, tracking_data):
    """Animate ball with error handling"""
    if not tracking_data:
        return False
    
    positions = tracking_data['tracking_data']
    metadata = tracking_data['metadata']
    
    # Set scene properties
    scene = bpy.context.scene
    scene.render.fps = int(metadata['fps'])
    scene.frame_start = 1
    scene.frame_end = len(positions)
    
    print(f"Setting up animation for {len(positions)} frames at {metadata['fps']} FPS")
    
    # Clear existing animation
    if ball.animation_data:
        ball.animation_data_clear()
    
    # Create keyframes
    for i, pos in enumerate(positions):
        frame_number = i + 1
        
        # Set position
        ball.location = (pos['x'], pos['y'], pos['z'])
        
        # Insert keyframe
        ball.keyframe_insert(data_path="location", frame=frame_number)
        
        # Progress indicator
        if i % 500 == 0:
            print(f"⏳ Processed {i+1}/{len(positions)} frames...")
    
    # Set interpolation to linear
    if ball.animation_data and ball.animation_data.action:
        for fcurve in ball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
    
    print("✅ Animation setup complete!")
    return True

def setup_lighting_robust():
    """Set up lighting with fallback methods"""
    ensure_object_mode()
    
    # Position existing camera
    camera = bpy.data.objects.get('Camera')
    if camera:
        camera.location = (12, -15, 8)
        camera.rotation_euler = (1.2, 0, 0)
        print("✅ Camera positioned")
    
    # Try to add light with operator
    try:
        bpy.ops.object.light_add(type='AREA', location=(12, 5, 10))
        light = bpy.context.active_object
        light.data.energy = 50
        light.data.size = 5
        print("✅ Light created with operator")
    except:
        # Manual light creation
        print("⚠️ Creating light manually...")
        light_data = bpy.data.lights.new(name="TennisLight", type='AREA')
        light_data.energy = 50
        light_data.size = 5
        light_object = bpy.data.objects.new("TennisLight", light_data)
        light_object.location = (12, 5, 10)
        bpy.context.collection.objects.link(light_object)
        print("✅ Light created manually")

def main():
    """Main execution function"""
    print("=" * 50)
    print("🎾 TENNIS BALL TRACKING - BLENDER IMPORT")
    print("=" * 50)
    
    # Step 1: Load tracking data
    print("\n📂 Step 1: Loading tracking data...")
    tracking_data = load_tracking_data_robust()
    
    if not tracking_data:
        print("❌ Cannot continue without tracking data")
        return False
    
    print(f"✅ Found {len(tracking_data['tracking_data'])} tracking points")
    
    # Step 2: Clear scene
    print("\n🧹 Step 2: Clearing scene...")
    safe_clear_scene()
    
    # Step 3: Create tennis court
    print("\n🏟️ Step 3: Creating tennis court...")
    court = create_tennis_court_robust()
    
    # Step 4: Create tennis ball
    print("\n🎾 Step 4: Creating tennis ball...")
    ball = create_tennis_ball_robust()
    
    # Step 5: Animate ball
    print("\n🎬 Step 5: Setting up animation...")
    success = animate_ball_robust(ball, tracking_data)
    
    if not success:
        print("❌ Animation setup failed")
        return False
    
    # Step 6: Setup lighting
    print("\n💡 Step 6: Setting up lighting...")
    setup_lighting_robust()
    
    # Step 7: Final setup
    print("\n🎯 Step 7: Final configuration...")
    
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
    
    print("\n" + "=" * 50)
    print("🎉 IMPORT COMPLETE!")
    print("=" * 50)
    print(f"📊 Imported: {len(tracking_data['tracking_data'])} tracking points")
    print(f"⏱️ Duration: {tracking_data['tracking_data'][-1]['time']:.1f} seconds")
    print(f"🎬 Frame rate: {tracking_data['metadata']['fps']} FPS")
    print("\n🎮 CONTROLS:")
    print("• Press SPACEBAR to play animation")
    print("• Use mouse wheel to zoom")
    print("• Middle-click drag to rotate view")
    print("• Shift+middle-click drag to pan")
    print("\n📋 NEXT STEPS:")
    print("1. Play animation to see ball movement")
    print("2. Adjust camera position as needed")
    print("3. Add your tennis video as background")
    print("4. Render your final composite!")
    
    return True

# Run the script
if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready to animate! Press SPACEBAR to play.")
    else:
        print("\n❌ Setup failed. Check the console for errors.")
