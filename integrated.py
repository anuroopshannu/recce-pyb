import pybullet as p
import pybullet_data
import time
import os
from tello_simple import TelloGymProper

def load_tesla(position=[3, 0, 0]):
    """Load Tesla car into the environment with correct orientation"""
    obj_file_name = "modely.obj"
    
    if not os.path.exists(obj_file_name):
        print(f"Warning: {obj_file_name} not found. Creating simple car placeholder.")
        # Create a simple box as Tesla placeholder
        tesla_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[2, 1, 0.5],
            rgbaColor=[1, 0, 0, 1]  # Red color
        )
        tesla_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[2, 1, 0.5]
        )
        # Use normal orientation for placeholder
        orientation = p.getQuaternionFromEuler([0, 0, 0])
    else:
        print(f"Loading Tesla model from {obj_file_name}")
        # Load the actual Tesla model
        tesla_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_file_name,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        tesla_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_file_name,
            meshScale=[1.0, 1.0, 1.0],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        # Use correct orientation from hello.py
        orientation = p.getQuaternionFromEuler([1.5708, 0, 0])  # 90 degrees rotation on X-axis
    
    # Create the Tesla multibody as STATIC object (mass = 0)
    tesla_body_id = p.createMultiBody(
        baseMass=0,  # Static object - no physics simulation
        baseCollisionShapeIndex=tesla_collision_shape_id,
        baseVisualShapeIndex=tesla_visual_shape_id,
        basePosition=position,
        baseOrientation=orientation
    )
    
    # Apply textures if Tesla model is loaded
    if os.path.exists(obj_file_name):
        apply_tesla_textures(tesla_body_id)
    
    print(f"Loaded Tesla (static) with body ID: {tesla_body_id}")
    print(f"Tesla orientation: [1.5708, 0, 0] (90° X-axis rotation)")
    return tesla_body_id

def apply_tesla_textures(tesla_body_id):
    """Apply textures to Tesla model"""
    texture_files = {
        "body": "Material_color_red.png"
    }
    
    # Get visual shape data
    visual_data = p.getVisualShapeData(tesla_body_id)
    print(f"Tesla has {len(visual_data)} visual shapes")
    
    # Load textures
    textures = {}
    for name, filename in texture_files.items():
        if os.path.exists(filename):
            textures[name] = p.loadTexture(filename)
            print(f"Loaded texture: {filename}")
    
    # Apply materials to Tesla parts
    for i, shape in enumerate(visual_data):
        link_id = shape[1]
        
        # Apply main body texture
        if "body" in textures:
            try:
                p.changeVisualShape(tesla_body_id, link_id, textureUniqueId=textures["body"])
            except:
                pass

def demonstrate_tello_tesla_interaction(tello, tesla_id):
    """Demonstrate Tello flying around Tesla"""
    print("\n=== Starting Tello + Tesla Demonstration ===")
    
    # Get Tesla position for reference
    tesla_pos = p.getBasePositionAndOrientation(tesla_id)[0]
    print(f"Tesla position: [{tesla_pos[0]:.1f}, {tesla_pos[1]:.1f}, {tesla_pos[2]:.1f}]")
    
    # Tello flight demonstration
    print(f"Tello starting height: {tello.get_height()}cm")
    print(f"Tello battery: {tello.get_battery()}%")
    
    # Phase 1: Takeoff
    tello.takeoff(target_height=1.5)
    print(f"After takeoff: {tello.get_height()}cm")
    
    # Phase 2: Hover and observe Tesla
    print("Hovering to observe Tesla...")
    tello.hover(duration=2.0)
    
    # Phase 3: Fly towards Tesla
    print("Flying towards Tesla...")
    tello.move_forward(200)  # Move 2 meters forward towards Tesla
    print(f"Position: {tello.get_position()}")
    
    # Phase 4: Circle around Tesla
    print("Circling around Tesla...")
    tello.move_right(100)  # Move to side
    print(f"Position: {tello.get_position()}")
    
    tello.rotate_clockwise(90)  # Face Tesla
    tello.hover(duration=1.0)
    
    # Move in a square pattern around Tesla
    print("Moving in square pattern around Tesla...")
    tello.move_forward(100)
    tello.rotate_clockwise(90)
    tello.move_forward(100)
    tello.rotate_clockwise(90)
    tello.move_forward(100)
    tello.rotate_clockwise(90)
    tello.move_forward(100)
    tello.rotate_clockwise(90)  # Back to original orientation
    
    # Phase 5: Altitude inspection
    print("Performing altitude inspection...")
    tello.move_up(50)  # Go higher
    print(f"Height: {tello.get_height()}cm")
    tello.hover(duration=2.0)
    
    tello.move_down(30)  # Come down a bit
    print(f"Height: {tello.get_height()}cm")
    
    # Phase 6: Return to start and land
    print("Returning to starting position...")
    tello.move_back(200)  # Return to start area
    print(f"Final position: {tello.get_position()}")
    
    tello.hover(duration=1.0)
    tello.land()
    print(f"Landing complete. Final height: {tello.get_height()}cm")

def animate_tesla(tesla_id):
    """Add some subtle animation to Tesla"""
    try:
        # Slightly rotate Tesla (like showing off the car)
        current_pos, current_orn = p.getBasePositionAndOrientation(tesla_id)
        new_orn = p.getQuaternionFromEuler([0, 0, 0.1])  # Small rotation
        p.resetBasePositionAndOrientation(tesla_id, current_pos, new_orn)
        
        # Wait a moment
        time.sleep(2.0)
        
        # Rotate back
        new_orn = p.getQuaternionFromEuler([0, 0, -0.1])
        p.resetBasePositionAndOrientation(tesla_id, current_pos, new_orn)
        
        time.sleep(2.0)
        
        # Back to original
        new_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(tesla_id, current_pos, new_orn)
        
    except Exception as e:
        print(f"Tesla animation error: {e}")

def main():
    """Main integrated simulation"""
    print("🚁🚗 Starting Tello + Tesla Integrated Simulation...")
    
    # Create Tello instance (this will create its own PyBullet environment)
    tello = TelloGymProper()
    
    if not tello.connect():
        print("Failed to connect to Tello simulation")
        return
    
    # Get the PyBullet physics client from the Tello environment
    physics_client = tello.env.CLIENT
    
    try:
        # Load Tesla into the same environment (static object)
        print("\nLoading Tesla into Tello environment...")
        tesla_id = load_tesla(position=[4, 0, 0])  # Place Tesla 4 meters away
        
        # Set optimal camera position to see both objects
        p.resetDebugVisualizerCamera(
            cameraDistance=8,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[2, 0, 1],
            physicsClientId=physics_client
        )
        
        print("🎯 Both Tello and Tesla loaded successfully!")
        print("Tesla is now a static object with correct orientation")
        print("Camera positioned to view both objects")
        
        # Start the demonstration
        demonstrate_tello_tesla_interaction(tello, tesla_id)
        
        print("\n🎉 Demonstration complete!")
        print("Tesla remains static throughout the simulation")
        
        print("\nSimulation running... Press Ctrl+C to exit")
        
        # Keep simulation running for observation (Tesla stays static)
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
    
    finally:
        print("🔌 Disconnecting...")
        tello.disconnect()
        print("✅ Simulation ended")

def interactive_mode():
    """Interactive mode for manual control"""
    print("🎮 Interactive Mode - Manual Tello Control")
    
    tello = TelloGymProper()
    if not tello.connect():
        return
    
    # Load Tesla
    tesla_id = load_tesla(position=[4, 0, 0])
    
    # Set camera
    physics_client = tello.env.CLIENT
    p.resetDebugVisualizerCamera(
        cameraDistance=8, cameraYaw=45, cameraPitch=-25,
        cameraTargetPosition=[2, 0, 1], physicsClientId=physics_client
    )
    
    print("\n📋 Available Commands:")
    print("1. takeoff() - Take off")
    print("2. land() - Land")
    print("3. hover(duration) - Hover for specified seconds")
    print("4. move_forward(cm) - Move forward")
    print("5. move_back(cm) - Move backward")
    print("6. move_left(cm) - Move left")
    print("7. move_right(cm) - Move right")
    print("8. move_up(cm) - Move up")
    print("9. move_down(cm) - Move down")
    print("10. rotate_clockwise(degrees) - Rotate clockwise")
    print("11. get_position() - Get current position")
    print("12. get_height() - Get current height")
    print("Type 'quit' to exit")
    
    try:
        while True:
            command = input("\n🚁 Enter command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                # Execute the command on the tello object
                if command:
                    result = eval(f"tello.{command}")
                    if result is not None:
                        print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
                print("Example usage: takeoff(1.5) or move_forward(100)")
                
    except KeyboardInterrupt:
        print("\n🛑 Interactive mode ended")
    
    finally:
        tello.disconnect()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()