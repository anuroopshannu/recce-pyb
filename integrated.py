import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import cv2
import threading
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
            p.changeVisualShape(tesla_body_id, -1, textureUniqueId=texture_id, physicsClientId=physics_client)
            print(f"✅ Applied texture '{texture_path}' to Tesla.")
        except Exception as e:
            print(f"⚠️ Could not apply texture: {e}")

def load_tesla(position=[3, 0, 0], physics_client=None):
    """Load Tesla car into the environment"""
    obj_file_name = "modely.obj"
    
    kwargs = {'physicsClientId': physics_client} if physics_client is not None else {}
    
    if not os.path.exists(obj_file_name):
        print(f"Warning: {obj_file_name} not found. Creating simple car placeholder.")
        shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[2, 1, 0.5], rgbaColor=[1, 0, 0, 1], **kwargs)
        coll = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[2, 1, 0.5], **kwargs)
        orientation = p.getQuaternionFromEuler([0, 0, 0])
    else:
        print(f"Loading Tesla model from {obj_file_name}")
        shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_file_name, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH, **kwargs)
        coll = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=obj_file_name, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH, **kwargs)
        orientation = p.getQuaternionFromEuler([1.5708, 0, 0])
        
    tesla_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll, baseVisualShapeIndex=shape, basePosition=position, baseOrientation=orientation, **kwargs)
    print(f"Loaded Tesla (static) with body ID: {tesla_body_id}")
    
    # Apply texture if Tesla model exists
    if os.path.exists(obj_file_name):
        apply_tesla_textures(tesla_body_id, physics_client)
        
    return tesla_body_id

def demonstrate_tello_tesla_with_camera(tello, tesla_id, camera_system):
    """Demonstrate Tello flying with the built-in, non-blocking camera feed"""
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
                camera_image = tello.capture_fresh_frame()  # Use fresh frame method
                
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
        print("📹 Camera thread started. Beginning flight maneuvers...")
        time.sleep(2) # Give camera window time to initialize

        print("🚁 Taking off...")
        tello.takeoff(target_height=1.5)
        
        print("👀 Hovering to observe Tesla...")
        tello.hover(duration=3.0)
        
        print("➡️ Flying towards Tesla...")
        tello.move_forward(250) # Move 2.5 meters
        
        print("🔄 Circling around Tesla...")
        tello.rotate_clockwise(90)
        tello.hover(duration=2.0)
        tello.move_forward(150)
        tello.rotate_clockwise(90)
        tello.hover(duration=2.0)
        tello.move_forward(250)
        
        print("📈 Changing altitude...")
        tello.move_up(100)
        tello.hover(duration=2.0)
        tello.move_down(150)
        
        print("🏠 Returning to starting position...")
        tello.rotate_clockwise(180) # Turn around
        tello.move_forward(400) # Fly back roughly to start
        
        print("🛬 Landing...")
        tello.land()
        
        print("✅ Mission complete! Camera will close shortly.")
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
    """Main integrated simulation"""
    print("🚁🚗📹 Starting Tello + Tesla + FPV Camera Integrated Simulation...")
    
    # Initialize Tello with camera enabled
    tello = TelloGymProper(camera_enabled=True)
    
    if not tello.connect():
        print("Failed to connect to Tello simulation")
        return
    
    # PERFORMANCE: Set camera to balanced quality/performance mode
    tello.set_camera_quality(width=320, height=240, frame_skip=2)  # Higher framerate
    # For higher quality: tello.set_camera_quality(width=640, height=480, frame_skip=4)
    
    physics_client = tello.env.CLIENT
    
    try:
        print("\nLoading Tesla into Tello environment...")
        tesla_position = [4, 0, 0]
        tesla_id = load_tesla(position=tesla_position, physics_client=physics_client)
        
        print("\n📹 Initializing OPTIMIZED FPV overlay system...")
        camera_system = TelloCameraSystem(tello)
        
        # Set up PyBullet debug camera (main simulation view)
        p.resetDebugVisualizerCamera(
            cameraDistance=8, cameraYaw=45, cameraPitch=-25,
            cameraTargetPosition=[2, 0, 1], physicsClientId=physics_client
        )
        
        print("🎯 Tello, Tesla, and OPTIMIZED FPV Camera system loaded!")
        print("📹 Using FAST rendering mode for better framerate")
        print("🖥️  Watch both windows: PyBullet simulation + FPV camera feed")
        
        demonstrate_tello_tesla_with_camera(tello, tesla_id, camera_system)
        
        print("\n🎉 Demonstration complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔌 Disconnecting...")
        cv2.destroyAllWindows()  # Ensure camera windows are closed
        tello.disconnect()
        print("✅ Simulation ended")

if __name__ == "__main__":
    main()