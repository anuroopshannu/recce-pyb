import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import cv2
import threading
import random
import math
from tello_simple import TelloGymProper

class TelloCameraSystem:
    """A simplified system to add an overlay to a camera frame."""
    
    def __init__(self, tello_instance):
        self.tello = tello_instance
        # FIXED: Get camera dimensions from the tello instance
        self.camera_width = getattr(tello_instance, 'camera_width', 320)
        self.camera_height = getattr(tello_instance, 'camera_height', 240)
        self.frame_count = 0
        print(f"✅ Overlay system initialized for {self.camera_width}x{self.camera_height}")

    def add_camera_overlay(self, image):
        """Add camera overlay with drone information"""
        if image is None:
            # Return a black frame with "No Signal" message
            black_frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            cv2.putText(black_frame, "NO CAMERA SIGNAL", (50, self.camera_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return black_frame

        self.frame_count += 1
        overlay_image = image.copy()
        
        # Get stats from the Tello wrapper
        pos = self.tello.get_position()
        drone_stats = {
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'height': self.tello.get_height(),
            'battery': self.tello.get_battery()
        }
        
        # Create semi-transparent overlay background - ADJUSTED for smaller resolution
        overlay_height = min(80, self.camera_height // 3)  # Scale overlay to image size
        overlay = overlay_image.copy()
        cv2.rectangle(overlay, (0, 0), (self.camera_width, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_image, 0.3, 0, overlay_image)
        
        # ADJUSTED: Smaller font for smaller resolution
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 if self.camera_width <= 320 else 0.6  # Smaller font for low res
        color = (0, 255, 0)  # Green text
        thickness = 1 if self.camera_width <= 320 else 2
        
        # FIXED: Position information - proper tuple format
        pos_text = f"Pos: ({drone_stats['x']:.1f},{drone_stats['y']:.1f},{drone_stats['z']:.1f})m"
        cv2.putText(overlay_image, pos_text, (5, 15), font, font_scale, color, thickness)
        
        # FIXED: Flight information - proper tuple format
        info_text = f"H:{drone_stats['height']:.0f}cm|Bat:{drone_stats['battery']}%|F:{self.frame_count}"
        cv2.putText(overlay_image, info_text, (5, 35), font, font_scale, color, thickness)
        
        # FIXED: Flight status - proper tuple format
        status = "FLYING" if self.tello.is_flying else "GROUNDED"
        status_color = (0, 255, 0) if self.tello.is_flying else (0, 165, 255)
        cv2.putText(overlay_image, f"Status: {status}", (5, 55), font, font_scale*0.8, status_color, thickness)
        
        # Add crosshair in center - ADJUSTED for resolution
        center_x, center_y = self.camera_width // 2, self.camera_height // 2
        cross_size = min(15, self.camera_width // 20)  # Scale crosshair
        cv2.line(overlay_image, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (255, 255, 255), 1)
        cv2.line(overlay_image, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (255, 255, 255), 1)
        cv2.circle(overlay_image, (center_x, center_y), 2, (255, 255, 255), -1)
        
        # FIXED: TELLO FPV watermark - proper tuple format and bounds checking
        watermark_x = max(5, self.camera_width - 120)
        watermark_y = max(self.camera_height - 15, overlay_height + 15)
        
        # Ensure watermark position is within image bounds
        if watermark_x < self.camera_width - 10 and watermark_y < self.camera_height - 5:
            cv2.putText(overlay_image, "TELLO FPV", (watermark_x, watermark_y), 
                       font, font_scale*0.7, (255, 255, 255), thickness)
        
        return overlay_image

def apply_tesla_textures(tesla_body_id, physics_client=None):
    """Apply textures to the Tesla model."""
    texture_path = "Material_color_red.png"
    if os.path.exists(texture_path):
        try:
            texture_id = p.loadTexture(texture_path, physicsClientId=physics_client)
            # FIXED: Use texture_id, not tesla_id
            p.changeVisualShape(tesla_body_id, -1, textureUniqueId=texture_id, physicsClientId=physics_client)
            print(f"✅ Applied texture '{texture_path}' to Tesla.")
        except Exception as e:
            print(f"⚠️ Could not apply texture: {e}")

def generate_random_positions():
    """Generate random positions for Tesla and drone - drone always behind Tesla"""
    # Random Tesla position in a 10x10 meter area
    tesla_x = random.uniform(-5, 5)
    tesla_y = random.uniform(-5, 5)
    tesla_z = 0  # On ground
    
    # Random Tesla orientation (0-360 degrees around Z-axis)
    tesla_yaw = random.uniform(0, 2 * math.pi)
    tesla_position = [tesla_x, tesla_y, tesla_z]
    
    print(f"🚗 Tesla positioned at ({tesla_x:.1f}, {tesla_y:.1f}) with rotation {math.degrees(tesla_yaw):.0f}°")
    
    # FIXED: Place drone behind Tesla at random distance and slight angle variation
    # Tesla's "forward" direction based on its yaw
    tesla_forward_x = math.cos(tesla_yaw)
    tesla_forward_y = math.sin(tesla_yaw)
    
    # Place drone behind Tesla (opposite to forward direction)
    distance = random.uniform(1, 2)  # 1-2 meters behind Tesla
    angle_variation = random.uniform(-30, 30)  # ±30° variation from directly behind
    angle_variation_rad = math.radians(angle_variation)
    
    # Calculate position behind Tesla with slight angle variation
    behind_angle = tesla_yaw + math.pi + angle_variation_rad  # 180° + variation
    
    drone_x = tesla_x + distance * math.cos(behind_angle)
    drone_y = tesla_y + distance * math.sin(behind_angle)
    drone_z = 0.1  # Just above ground
    
    # FIXED: Drone always faces Tesla (same calculation, but clearer)
    to_tesla_x = tesla_x - drone_x
    to_tesla_y = tesla_y - drone_y
    drone_yaw = math.atan2(to_tesla_y, to_tesla_x)
    
    drone_position = [drone_x, drone_y, drone_z]
    
    print(f"🚁 Drone positioned {distance:.1f}m behind Tesla at ({drone_x:.1f}, {drone_y:.1f})")
    print(f"🎯 Drone facing Tesla (yaw: {math.degrees(drone_yaw):.0f}°, variation: {angle_variation:.0f}°)")
    print(f"📏 Distance between drone and Tesla: {distance:.1f}m")
    
    return tesla_position, tesla_yaw, drone_position, drone_yaw

def load_tesla(position=[3, 0, 0], yaw_angle=0, physics_client=None):
    """Load Tesla car into the environment with random position and orientation"""
    obj_file_name = "modely.obj"
    
    kwargs = {'physicsClientId': physics_client} if physics_client is not None else {}
    
    if not os.path.exists(obj_file_name):
        print(f"Warning: {obj_file_name} not found. Creating simple car placeholder.")
        shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[2, 1, 0.5], rgbaColor=[1, 0, 0, 1], **kwargs)
        coll = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[2, 1, 0.5], **kwargs)
        # For simple box, just use the yaw rotation
        orientation = p.getQuaternionFromEuler([0, 0, yaw_angle])
    else:
        print(f"Loading Tesla model from {obj_file_name}")
        shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_file_name, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH, **kwargs)
        coll = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=obj_file_name, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH, **kwargs)
        # For Tesla model, combine the model's natural rotation with the random yaw
        orientation = p.getQuaternionFromEuler([1.5708, 0, yaw_angle])
        
    tesla_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=shape, basePosition=position, baseOrientation=orientation, **kwargs)
    print(f"Loaded Tesla (static) with body ID: {tesla_body_id} at position {position}")
    
    # Apply texture if Tesla model exists
    if os.path.exists(obj_file_name):
        apply_tesla_textures(tesla_body_id, physics_client)
        
    return tesla_body_id

def demonstrate_tello_tesla_with_camera(tello, tesla_id, camera_system, tesla_pos, drone_initial_yaw):
    """Demonstrate Tello taking off while facing Tesla with live camera feed"""
    print("\n=== Starting Tello + Tesla + Camera Demonstration ===")
    
    camera_running = threading.Event()
    camera_running.set()
    
    def camera_thread_func():
        """Function to run in a separate thread for continuous camera feed"""
        window_name = "Tello FPV Camera - LIVE FEED"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, 50, 50)
        
        print("📹 FPV camera thread started - Press 'q' to quit camera")
        
        while camera_running.is_set():
            try:
                # Get fresh frame from the Tello camera system
                camera_image = tello.capture_fresh_frame()
                
                if camera_image is not None and camera_image.size > 0:
                    # The image from PyBullet is RGB, convert to BGR for OpenCV
                    bgr_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
                    overlay_image = camera_system.add_camera_overlay(bgr_image)
                    cv2.imshow(window_name, overlay_image)
                else:
                    # Show "No Signal" frame if camera image is None
                    no_signal_frame = camera_system.add_camera_overlay(None)
                    cv2.imshow(window_name, no_signal_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("📹 Camera quit requested by user")
                    camera_running.clear()
                    break
                    
            except Exception as e:
                print(f"❌ Camera thread error: {e}")
            
            time.sleep(1/60) # Target ~60 FPS
        
        cv2.destroyAllWindows()
        print("📹 Camera thread stopped.")

    # Start the camera thread
    cam_thread = threading.Thread(target=camera_thread_func, name="CameraThread")
    cam_thread.daemon = True
    cam_thread.start()
    
    try:
        print("📹 Camera thread started. Beginning flight demonstration...")
        time.sleep(2) # Give camera window time to initialize

        print("🚁 Taking off while maintaining view of Tesla...")
        # FIXED: Maintain orientation toward Tesla during takeoff
        tello.takeoff_facing_target(target_height=1.5, target_position=tesla_pos)
        
        print("👀 Hovering behind Tesla with perfect view...")
        tello.hover(duration=5.0)  # Longer hover to appreciate the view
        
        # Optional: Add some gentle movements while maintaining Tesla focus
        print("📹 Demonstrating camera movement while keeping Tesla in view...")
        
        # Gentle movements that maintain Tesla visibility
        print("📈 Moving up for better perspective...")
        tello.move_up(100)  # Go higher
        tello.hover(duration=3.0)
        
        print("➡️ Gentle sideways movement...")
        tello.move_left(50)  # Move slightly left
        tello.hover(duration=2.0)
        tello.move_right(100)  # Move to the right
        tello.hover(duration=2.0)
        tello.move_left(50)  # Return to center
        
        print("🔄 Slight rotation while keeping Tesla in view...")
        tello.rotate_clockwise(15)  # Small rotation
        tello.hover(duration=2.0)
        tello.rotate_counter_clockwise(30)  # Turn to the other side
        tello.hover(duration=2.0)
        tello.rotate_clockwise(15)  # Return to center
        
        print("📉 Coming down...")
        tello.move_down(50)  # Come down a bit
        tello.hover(duration=2.0)
        
        print("🛬 Landing...")
        tello.land()
        
        print("✅ Demonstration complete! Camera will close shortly.")
        time.sleep(3)

    except KeyboardInterrupt:
        print("\n🛑 Mission interrupted by user")
    
    finally:
        # Signal the camera thread to stop and wait for it to finish
        camera_running.clear()
        if cam_thread.is_alive():
            cam_thread.join(timeout=3)
        print("Main thread finished.")

def main():
    """Main integrated simulation with random positioning"""
    print("🚁🚗📹 Starting Tello + Tesla + FPV Camera Integrated Simulation...")
    print("🎲 Randomizing positions each run...")
    
    # Generate random positions for Tesla and drone
    tesla_pos, tesla_yaw, drone_pos, drone_yaw = generate_random_positions()
    
    # Initialize Tello with camera enabled
    tello = TelloGymProper(camera_enabled=True)
    
    if not tello.connect():
        print("Failed to connect to Tello simulation")
        return
    
    # PERFORMANCE: Set camera to balanced quality/performance mode
    tello.set_camera_quality(width=320, height=240, frame_skip=2)
    
    physics_client = tello.env.CLIENT
    
    try:
        # Load Tesla with random position and orientation
        print(f"\nLoading Tesla at random position: ({tesla_pos[0]:.1f}, {tesla_pos[1]:.1f})...")
        tesla_id = load_tesla(position=tesla_pos, yaw_angle=tesla_yaw, physics_client=physics_client)
        
        # Set drone's initial position and orientation to face Tesla
        print(f"Positioning drone at ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}) facing Tesla...")
        
        # Update the drone's position in the simulation
        # First, we need to reset the drone to the new position
        p.resetBasePositionAndOrientation(
            0,  # Assuming drone is body ID 0
            drone_pos,
            p.getQuaternionFromEuler([0, 0, drone_yaw]),
            physicsClientId=physics_client
        )
        
        # Update the tello wrapper's internal position tracking
        tello.target_pos = np.array(drone_pos)
        tello.target_rpy = np.array([0, 0, drone_yaw])
        
        print("\n📹 Initializing OPTIMIZED FPV overlay system...")
        camera_system = TelloCameraSystem(tello)
        
        # Set up PyBullet debug camera to show both drone and Tesla
        # Position camera to show the scene from above and slightly to the side
        center_x = (tesla_pos[0] + drone_pos[0]) / 2
        center_y = (tesla_pos[1] + drone_pos[1]) / 2
        
        p.resetDebugVisualizerCamera(
            cameraDistance=10, cameraYaw=30, cameraPitch=-25,
            cameraTargetPosition=[center_x, center_y, 0.5], physicsClientId=physics_client
        )
        
        print("🎯 Tello, Tesla, and OPTIMIZED FPV Camera system loaded!")
        print("📹 Using FAST rendering mode for better framerate")
        print("🖥️  Watch both windows: PyBullet simulation + FPV camera feed")
        print(f"🎯 Drone is positioned behind Tesla and should see it clearly in FPV!")
        
        demonstrate_tello_tesla_with_camera(tello, tesla_id, camera_system, tesla_pos, drone_yaw)
        
        print("\n🎉 Demonstration complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔌 Disconnecting...")
        cv2.destroyAllWindows()
        tello.disconnect()
        print("✅ Simulation ended")

if __name__ == "__main__":
    main()