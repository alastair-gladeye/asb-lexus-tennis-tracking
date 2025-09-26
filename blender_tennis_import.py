"""
Blender Tennis Ball Tracking Import Script
==========================================

This script imports tennis ball tracking data from tennis_tracking.json
and creates an animated ball object in Blender.

Usage:
1. Open Blender
2. Switch to Scripting workspace
3. Load this script or paste it into the text editor
4. Make sure tennis_tracking.json is in the same directory as your .blend file
5. Run the script

The script will:
- Import tracking data
- Create a tennis ball object
- Set up keyframe animation
- Configure proper scaling and timing
"""

import bpy
import json
import bmesh
from mathutils import Vector
import os

def create_ball_manual():
    """Manually create tennis ball using bmesh when operators fail"""
    # Create mesh and object
    mesh = bpy.data.meshes.new("TennisBall")
    ball = bpy.data.objects.new("TennisBall", mesh)
    bpy.context.collection.objects.link(ball)
    
    # Create bmesh sphere
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.0335)
    bm.to_mesh(mesh)
    bm.free()
    
    return ball

def create_court_manual():
    """Manually create tennis court using bmesh when operators fail"""
    # Create mesh and object
    mesh = bpy.data.meshes.new("TennisCourt")
    court = bpy.data.objects.new("TennisCourt", mesh)
    bpy.context.collection.objects.link(court)
    
    # Create bmesh plane
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1)
    bm.to_mesh(mesh)
    bm.free()
    
    # Position and scale court
    court.location = (11.885, 5.485, 0)
    court.scale = (23.77, 10.97, 1)
    
    return court

def clear_scene():
    """Clear existing mesh objects from the scene"""
    # Ensure we're in object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    try:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete(use_global=False)
    except Exception as e:
        print(f"Error clearing scene with operators: {e}")
        # Fallback: manually remove mesh objects
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        for obj in mesh_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

def create_tennis_ball():
    """Create a tennis ball mesh with proper materials"""
    # Ensure we're in object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    try:
        # Create UV sphere for tennis ball
        bpy.ops.mesh.uv_sphere_add(radius=0.0335, location=(0, 0, 0))  # 6.7cm diameter
        ball = bpy.context.active_object
        ball.name = "TennisBall"
    except Exception as e:
        print(f"Error creating ball with operator: {e}")
        # Fallback: create ball manually using bmesh
        ball = create_ball_manual()
        return ball
    
    # Create tennis ball material
    mat = bpy.data.materials.new(name="TennisBallMaterial")
    mat.use_nodes = True
    
    # Set up material nodes for tennis ball appearance
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Add principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.8, 0.9, 0.2, 1.0)  # Tennis ball yellow
    principled.inputs['Roughness'].default_value = 0.3
    principled.inputs['Specular'].default_value = 0.2
    
    # Add output node
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to ball
    ball.data.materials.append(mat)
    
    return ball

def create_court_reference():
    """Create a reference tennis court for scale"""
    # Ensure we're in object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Court dimensions: 23.77m x 10.97m
    try:
        bpy.ops.mesh.plane_add(size=1, location=(11.885, 5.485, 0))
        court = bpy.context.active_object
        court.name = "TennisCourt"
        court.scale = (23.77, 10.97, 1)
    except Exception as e:
        print(f"Error creating court with operator: {e}")
        # Fallback: create court manually using bmesh
        court = create_court_manual()
        return court
    
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

def load_tracking_data(filepath):
    """Load tennis tracking data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        print("Make sure tennis_tracking.json is in the same directory as your .blend file")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return None

def animate_tennis_ball(ball, tracking_data):
    """Animate the tennis ball using tracking data"""
    if not tracking_data:
        return
    
    metadata = tracking_data['metadata']
    positions = tracking_data['tracking_data']
    
    fps = metadata['fps']
    
    # Set Blender's frame rate to match video
    bpy.context.scene.render.fps = int(fps)
    
    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(positions)
    
    print(f"Animating {len(positions)} frames at {fps} FPS")
    
    # Clear existing keyframes
    ball.animation_data_clear()
    
    # Set keyframes for each tracked position
    for i, pos in enumerate(positions):
        frame_number = i + 1  # Blender frames start at 1
        
        # Set ball position (convert from tennis coordinates to Blender coordinates)
        ball.location = (pos['x'], pos['y'], pos['z'])
        
        # Insert keyframe
        ball.keyframe_insert(data_path="location", frame=frame_number)
        
        # Progress indicator
        if i % 100 == 0:
            print(f"Processed frame {i+1}/{len(positions)}")
    
    # Set interpolation to linear for smooth motion
    if ball.animation_data and ball.animation_data.action:
        for fcurve in ball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'
    
    print("Animation complete!")

def setup_camera_and_lighting():
    """Set up camera and lighting for better visualization"""
    # Position camera for good court view
    camera = bpy.data.objects.get('Camera')
    if camera:
        camera.location = (12, -15, 8)
        camera.rotation_euler = (1.2, 0, 0)
    
    # Add area light for better illumination
    try:
        bpy.ops.object.light_add(type='AREA', location=(12, 5, 10))
        light = bpy.context.active_object
        light.data.energy = 50
        light.data.size = 5
    except Exception as e:
        print(f"Error adding light with operator: {e}")
        # Fallback: create light manually
        light_data = bpy.data.lights.new(name="TennisLight", type='AREA')
        light_data.energy = 50
        light_data.size = 5
        light_object = bpy.data.objects.new("TennisLight", light_data)
        light_object.location = (12, 5, 10)
        bpy.context.collection.objects.link(light_object)

def main():
    """Main function to import and set up tennis ball animation"""
    print("=== Tennis Ball Tracking Import ===")
    
    # Get the directory of the current .blend file
    blend_dir = bpy.path.abspath("//")
    if not blend_dir:
        # If .blend file hasn't been saved, use current working directory
        blend_dir = os.getcwd()
    
    json_path = os.path.join(blend_dir, "tennis_tracking.json")
    
    # Load tracking data
    print(f"Loading tracking data from: {json_path}")
    tracking_data = load_tracking_data(json_path)
    
    if not tracking_data:
        print("❌ Failed to load tracking data")
        return
    
    # Clear existing scene
    print("Clearing scene...")
    clear_scene()
    
    # Create tennis court reference
    print("Creating tennis court reference...")
    court = create_court_reference()
    
    # Create tennis ball
    print("Creating tennis ball...")
    ball = create_tennis_ball()
    
    # Animate the ball
    print("Setting up animation...")
    animate_tennis_ball(ball, tracking_data)
    
    # Set up camera and lighting
    print("Configuring camera and lighting...")
    setup_camera_and_lighting()
    
    # Set viewport shading to solid or material preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL_PREVIEW'
    
    # Set current frame to 1
    bpy.context.scene.frame_set(1)
    
    print("✅ Tennis ball animation setup complete!")
    print(f"📊 Imported {len(tracking_data['tracking_data'])} tracking points")
    print("🎬 Press SPACE to play animation")
    print("\nNext steps:")
    print("1. Press SPACE to play the animation")
    print("2. Adjust camera angle as needed")
    print("3. Add your background video/image")
    print("4. Render your composite!")

# Run the main function
if __name__ == "__main__":
    main()
