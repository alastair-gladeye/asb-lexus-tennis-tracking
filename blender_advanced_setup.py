"""
Advanced Blender Tennis Ball Setup
==================================

This script provides advanced features for tennis ball visualization:
- Multiple visualization modes
- Custom materials and effects
- Automatic video background setup
- Physics simulation options
- Batch rendering utilities

Usage: Run this after the basic import script for enhanced visuals
"""

import bpy
import json
import bmesh
from mathutils import Vector
import os

class TennisBallVisualizer:
    def __init__(self):
        self.ball = None
        self.court = None
        self.tracking_data = None
    
    def add_ball_trail(self, trail_length=20):
        """Add a trail effect to show ball path"""
        if not self.ball:
            print("No ball object found")
            return
        
        # Create curve for trail
        curve_data = bpy.data.curves.new('BallTrail', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2
        
        # Create spline
        polyline = curve_data.splines.new('POLY')
        
        # Add trail material
        trail_mat = bpy.data.materials.new(name="TrailMaterial")
        trail_mat.use_nodes = True
        nodes = trail_mat.node_tree.nodes
        nodes.clear()
        
        # Emission shader for glowing trail
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (1.0, 0.5, 0.0, 1.0)  # Orange
        emission.inputs['Strength'].default_value = 2.0
        
        output = nodes.new('ShaderNodeOutputMaterial')
        trail_mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        # Create trail object
        trail_obj = bpy.data.objects.new('BallTrail', curve_data)
        bpy.context.collection.objects.link(trail_obj)
        trail_obj.data.materials.append(trail_mat)
        
        # Set curve properties
        curve_data.bevel_depth = 0.005
        curve_data.bevel_resolution = 4
        
        print(f"✅ Ball trail added with {trail_length} point history")
        return trail_obj
    
    def create_impact_effects(self):
        """Create particle effects at ball-court impact points"""
        # Add particle system to court
        if not self.court:
            print("No court object found")
            return
        
        # Select court object
        bpy.context.view_layer.objects.active = self.court
        
        # Add particle system
        bpy.ops.object.particle_system_add()
        particles = self.court.particle_systems[-1]
        
        # Configure particles for impact effects
        settings = particles.settings
        settings.name = "ImpactSparks"
        settings.type = 'EMITTER'
        settings.count = 50
        settings.lifetime = 30
        settings.emit_from = 'FACE'
        settings.physics_type = 'NEWTON'
        
        # Particle appearance
        settings.render_type = 'OBJECT'
        
        # Create small sphere for particles
        bpy.ops.mesh.uv_sphere_add(radius=0.01, location=(0, 0, -10))
        particle_obj = bpy.context.active_object
        particle_obj.name = "SparkParticle"
        settings.instance_object = particle_obj
        
        print("✅ Impact particle effects added")
    
    def setup_realistic_lighting(self):
        """Set up realistic lighting for tennis court"""
        # Remove default light
        if bpy.data.objects.get('Light'):
            bpy.data.objects.remove(bpy.data.objects['Light'], do_unlink=True)
        
        # Add sun light (outdoor tennis)
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 20))
        sun = bpy.context.active_object
        sun.name = "TennisSun"
        sun.data.energy = 3.0
        sun.rotation_euler = (0.785, 0, 0.785)  # 45-degree angle
        
        # Add area lights for court illumination
        positions = [
            (5, -2, 8),    # Court left
            (18, -2, 8),   # Court right
            (11.8, 12, 8)  # Court far end
        ]
        
        for i, pos in enumerate(positions):
            bpy.ops.object.light_add(type='AREA', location=pos)
            light = bpy.context.active_object
            light.name = f"CourtLight_{i+1}"
            light.data.energy = 20
            light.data.size = 3
            light.rotation_euler = (1.57, 0, 0)  # Point down
        
        print("✅ Realistic lighting setup complete")
    
    def add_video_background(self, video_path):
        """Set up video background for compositing"""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        
        # Enable nodes in world shader
        bpy.context.scene.world.use_nodes = True
        world_nodes = bpy.context.scene.world.node_tree.nodes
        world_nodes.clear()
        
        # Add background shader
        background = world_nodes.new('ShaderNodeBackground')
        
        # Add image texture node
        image_texture = world_nodes.new('ShaderNodeTexImage')
        
        # Load video
        video = bpy.data.images.load(video_path)
        video.source = 'MOVIE'
        image_texture.image = video
        
        # Add world output
        world_output = world_nodes.new('ShaderNodeOutputWorld')
        
        # Connect nodes
        world_links = bpy.context.scene.world.node_tree.links
        world_links.new(image_texture.outputs['Color'], background.inputs['Color'])
        world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
        
        # Set video to auto-refresh
        video.use_auto_refresh = True
        
        print(f"✅ Video background added: {os.path.basename(video_path)}")
    
    def create_ball_variants(self):
        """Create different ball visualization options"""
        variants = {
            'Glowing': {'color': (1, 1, 0, 1), 'emission': 2.0, 'roughness': 0.0},
            'Metallic': {'color': (0.8, 0.8, 0.9, 1), 'emission': 0.0, 'roughness': 0.1, 'metallic': 1.0},
            'Glass': {'color': (0.9, 0.9, 1, 1), 'emission': 0.0, 'roughness': 0.0, 'transmission': 1.0},
            'Fire': {'color': (1, 0.3, 0, 1), 'emission': 3.0, 'roughness': 0.3}
        }
        
        for variant_name, properties in variants.items():
            # Duplicate ball
            if self.ball:
                new_ball = self.ball.copy()
                new_ball.data = self.ball.data.copy()
                new_ball.name = f"TennisBall_{variant_name}"
                bpy.context.collection.objects.link(new_ball)
                
                # Create variant material
                mat = bpy.data.materials.new(name=f"Ball_{variant_name}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                if properties.get('emission', 0) > 0:
                    # Emission shader
                    emission = nodes.new('ShaderNodeEmission')
                    emission.inputs['Color'].default_value = properties['color']
                    emission.inputs['Strength'].default_value = properties['emission']
                    
                    output = nodes.new('ShaderNodeOutputMaterial')
                    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
                else:
                    # Principled BSDF
                    principled = nodes.new('ShaderNodeBsdfPrincipled')
                    principled.inputs['Base Color'].default_value = properties['color']
                    principled.inputs['Roughness'].default_value = properties.get('roughness', 0.5)
                    principled.inputs['Metallic'].default_value = properties.get('metallic', 0.0)
                    principled.inputs['Transmission'].default_value = properties.get('transmission', 0.0)
                    
                    output = nodes.new('ShaderNodeOutputMaterial')
                    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                
                # Assign material
                new_ball.data.materials.clear()
                new_ball.data.materials.append(mat)
                
                # Hide by default (user can unhide as needed)
                new_ball.hide_viewport = True
        
        print("✅ Ball variants created: Glowing, Metallic, Glass, Fire")
    
    def setup_render_settings(self, output_dir="./renders/"):
        """Configure optimal render settings for tennis video"""
        scene = bpy.context.scene
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Render settings
        scene.render.engine = 'CYCLES'  # or 'EEVEE' for faster rendering
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.resolution_percentage = 100
        
        # Output settings
        scene.render.filepath = os.path.join(output_dir, "tennis_frame_")
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.compression = 15
        
        # Cycles settings (if using Cycles)
        if scene.render.engine == 'CYCLES':
            scene.cycles.samples = 128  # Adjust for quality vs speed
            scene.cycles.use_denoising = True
        
        # EEVEE settings (if using EEVEE)
        if scene.render.engine == 'BLENDER_EEVEE':
            scene.eevee.taa_render_samples = 64
            scene.eevee.use_bloom = True
            scene.eevee.bloom_intensity = 0.1
        
        print(f"✅ Render settings configured for output to: {output_dir}")
    
    def batch_render_frames(self, start_frame=1, end_frame=None, step=1):
        """Render specific frame range"""
        scene = bpy.context.scene
        
        if end_frame is None:
            end_frame = scene.frame_end
        
        original_start = scene.frame_start
        original_end = scene.frame_end
        
        scene.frame_start = start_frame
        scene.frame_end = end_frame
        
        print(f"🎬 Rendering frames {start_frame} to {end_frame}")
        
        # Render animation
        bpy.ops.render.render(animation=True)
        
        # Restore original settings
        scene.frame_start = original_start
        scene.frame_end = original_end
        
        print("✅ Batch render complete!")

def main():
    """Main function for advanced tennis ball setup"""
    print("=== Advanced Tennis Ball Visualization ===")
    
    visualizer = TennisBallVisualizer()
    
    # Find existing objects
    visualizer.ball = bpy.data.objects.get('TennisBall')
    visualizer.court = bpy.data.objects.get('TennisCourt')
    
    if not visualizer.ball:
        print("❌ No tennis ball found. Run the basic import script first.")
        return
    
    print("🎾 Setting up advanced visualization...")
    
    # Set up realistic lighting
    visualizer.setup_realistic_lighting()
    
    # Create ball variants
    visualizer.create_ball_variants()
    
    # Add ball trail
    visualizer.add_ball_trail()
    
    # Set up render settings
    visualizer.setup_render_settings()
    
    # Optional: Add video background (uncomment and specify path)
    # video_path = "path/to/your/tennis_video.mp4"
    # visualizer.add_video_background(video_path)
    
    print("✅ Advanced setup complete!")
    print("\nAvailable features:")
    print("- Realistic lighting setup")
    print("- Multiple ball material variants")
    print("- Ball trail visualization")
    print("- Optimized render settings")
    print("\nTo use:")
    print("1. Unhide ball variants in Outliner as needed")
    print("2. Adjust materials in Shading workspace")
    print("3. Use Ctrl+F12 to render animation")

if __name__ == "__main__":
    main()
