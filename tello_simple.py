import numpy as np
import time
import pybullet as p
import math  # ADDED: Import math module for atan2 function
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Try to import the enums, but provide fallbacks for older versions
try:
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
    HAS_ENUMS = True
except ImportError:
    # Older versions don't have these enums
    HAS_ENUMS = False
    print("Using compatibility mode for older gym-pybullet-drones version")

class TelloGymProper:
    """
    DJI Tello simulation using gym-pybullet-drones
    """
    
    def __init__(self, camera_enabled=False):
        self.env = None
        self.ctrl = None
        self.drone_id = 0
        self.is_flying = False
        self.target_pos = np.array([0, 0, 1])
        self.target_rpy = np.array([0, 0, 0])
        self.real_time_mode = True
        self.camera_enabled = camera_enabled
        self.last_frame = None
        
        # Camera parameters - OPTIMIZED
        self.camera_width = 320  # Reduced from 640 for better performance
        self.camera_height = 240  # Reduced from 480 for better performance
        self.camera_fov = 60
        self.camera_near = 0.1
        self.camera_far = 50  # Reduced from 100 for better performance
        
        # PERFORMANCE: Camera update control
        self.camera_frame_skip = 3  # Only update camera every 3rd frame
        self.camera_step_counter = 0
        
    def connect(self):
        """Initialize the gym-pybullet-drones environment"""
        try:
            # Simple initialization that works with older versions
            self.env = CtrlAviary(
                drone_model=DroneModel.CF2X,
                num_drones=1,
                initial_xyzs=np.array([[0, 0, 0.1]]),
                initial_rpys=np.array([[0, 0, 0]]),
                physics=Physics.PYB,
                gui=True
                # Removed problematic parameters that don't exist in older versions
            )
            
            # Initialize PID controller
            self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
            
            # Reset environment
            obs, info = self.env.reset()
            
            print("Connected to gym-pybullet-drones environment")
            
            if self.camera_enabled:
                print("✅ Camera mode enabled - using PyBullet camera")
            
            # Check which frequency attribute exists
            if hasattr(self.env, 'SIM_FREQ'):
                print(f"Environment frequency: {self.env.SIM_FREQ} Hz")
            elif hasattr(self.env, 'FREQ'):
                print(f"Environment frequency: {self.env.FREQ} Hz")
            elif hasattr(self.env, 'freq'):
                print(f"Environment frequency: {self.env.freq} Hz")
            else:
                print("Using default frequency: 240 Hz")
            
            if self.real_time_mode:
                print("Real-time simulation enabled - Tello timing")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            print("Make sure gym-pybullet-drones is installed:")
            print("pip install gym-pybullet-drones")
            return False
    
    def _get_camera_image(self):
        """Capture camera image from drone's perspective using PyBullet - OPTIMIZED"""
        if not self.camera_enabled or not self.env:
            return None
        
        try:
            # Get drone state
            obs = self.env._getDroneStateVector(self.drone_id)
            drone_pos = obs[0:3]
            drone_quat = obs[3:7]
            
            # Convert quaternion to rotation matrix
            rotation_matrix = p.getMatrixFromQuaternion(drone_quat, physicsClientId=self.env.CLIENT)
            rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
            
            # Camera positioning
            camera_offset = np.array([0.1, 0, -0.05])
            camera_pos = drone_pos + rotation_matrix @ camera_offset
            
            forward_vector = rotation_matrix[:, 0]
            target_pos = camera_pos + forward_vector * 10
            up_vector = np.array([0, 0, 1])  # Keep level horizon
            
            # Compute view matrix
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=target_pos,
                cameraUpVector=up_vector,
                physicsClientId=self.env.CLIENT
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=self.camera_fov,
                aspect=self.camera_width / self.camera_height,
                nearVal=self.camera_near,
                farVal=self.camera_far,
                physicsClientId=self.env.CLIENT
            )
            
            # FIXED: Capture image and handle the tuple correctly
            width, height, rgb_img, _, _ = p.getCameraImage(
                width=self.camera_width,
                height=self.camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER,  # Fast rendering
                physicsClientId=self.env.CLIENT
            )
            
            # FIXED: Correct array conversion - rgb_img is already a list/array
            rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
            return rgb_array[:, :, :3]  # Remove alpha channel, return RGB
            
        except Exception as e:
            print(f"❌ Error capturing camera image: {e}")
            return None
    
    def _get_freq(self):
        """Get simulation frequency"""
        if hasattr(self.env, 'SIM_FREQ'):
            return self.env.SIM_FREQ
        elif hasattr(self.env, 'FREQ'):
            return self.env.FREQ
        elif hasattr(self.env, 'freq'):
            return self.env.freq
        else:
            return 240  # Default frequency
    
    def _get_ctrl_timestep(self):
        """Get control timestep"""
        if hasattr(self.env, 'CTRL_TIMESTEP'):
            return self.env.CTRL_TIMESTEP
        elif hasattr(self.env, 'TIMESTEP'):
            return self.env.TIMESTEP
        else:
            return 1.0 / self._get_freq()
    
    def _step_with_timing(self, action):
        """Step simulation with real-time synchronization - OPTIMIZED CAMERA"""
        if self.real_time_mode:
            step_start_time = time.time()
        
        # Step the simulation
        obs, reward, terminated, truncated, info = self.env.step(action.reshape(1, -1))
        
        # OPTIMIZED: Only update camera every few frames during movement
        if self.camera_enabled:
            self.camera_step_counter += 1
            if self.camera_step_counter >= self.camera_frame_skip:
                self.last_frame = self._get_camera_image()
                self.camera_step_counter = 0

        if self.real_time_mode:
            # OPTIMIZED: Shorter sleep time for better responsiveness
            target_step_time = self._get_ctrl_timestep() * 0.8  # 20% faster
            
            elapsed = time.time() - step_start_time
            if elapsed < target_step_time:
                time.sleep(target_step_time - elapsed)
        
        return obs, reward, terminated, truncated, info
    
    def takeoff(self, target_height=1.0):
        """Take off to specified height - Real Tello takes ~3-4 seconds"""
        if not self.env:
            return False
            
        print("Taking off...")
        self.is_flying = True
        
        # Tello takeoff sequence - realistic timing
        freq = self._get_freq()
        takeoff_steps = int(3.5 * freq)  # 3.5 seconds like real Tello
        
        for i in range(takeoff_steps):
            # Get current observation
            obs = self.env._getDroneStateVector(self.drone_id)
            
            # Smooth target height progression (faster initial climb)
            progress = min(1.0, i / (2.5 * freq))  # Reach target in 2.5 seconds
            # Ease-out curve for smooth landing at target height
            smooth_progress = 1 - (1 - progress) ** 2
            current_target_height = 0.1 + (target_height - 0.1) * smooth_progress
            
            target_pos = np.array([0, 0, current_target_height])
            target_rpy = np.array([0, 0, 0])
            
            # Compute control action
            action, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self._get_ctrl_timestep(),
                state=obs,
                target_pos=target_pos,
                target_rpy=target_rpy
            )
            
            # Step simulation with timing
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
            
        self.target_pos = np.array([0, 0, target_height])
        print("Takeoff complete")
        return True
    
    def takeoff_facing_target(self, target_height=1.0, target_position=None):
        """Take off while maintaining orientation toward a target position"""
        if not self.env:
            return False
            
        print("Taking off while maintaining target orientation...")
        self.is_flying = True
        
        # Get current position and calculate target yaw if target is provided
        obs = self.env._getDroneStateVector(self.drone_id)
        start_pos = obs[0:3]
        
        if target_position is not None:
            # Calculate yaw to face target throughout takeoff
            to_target_x = target_position[0] - start_pos[0]
            to_target_y = target_position[1] - start_pos[1]
            target_yaw = math.atan2(to_target_y, to_target_x)
            self.target_rpy = np.array([0, 0, target_yaw])
        
        # Tello takeoff sequence - realistic timing
        freq = self._get_freq()
        takeoff_steps = int(3.5 * freq)  # 3.5 seconds like real Tello
        
        for i in range(takeoff_steps):
            # Get current observation
            obs = self.env._getDroneStateVector(self.drone_id)
            current_pos = obs[0:3]
            
            # Update yaw to face target if position is provided
            if target_position is not None:
                to_target_x = target_position[0] - current_pos[0]
                to_target_y = target_position[1] - current_pos[1]
                target_yaw = math.atan2(to_target_y, to_target_x)
                target_rpy = np.array([0, 0, target_yaw])
            else:
                target_rpy = self.target_rpy
            
            # Smooth target height progression (faster initial climb)
            progress = min(1.0, i / (2.5 * freq))  # Reach target in 2.5 seconds
            # Ease-out curve for smooth landing at target height
            smooth_progress = 1 - (1 - progress) ** 2
            current_target_height = 0.1 + (target_height - 0.1) * smooth_progress
            
            target_pos = np.array([current_pos[0], current_pos[1], current_target_height])
            
            # Compute control action
            action, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self._get_ctrl_timestep(),
                state=obs,
                target_pos=target_pos,
                target_rpy=target_rpy
            )
            
            # Step simulation with timing
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
            
        self.target_pos = np.array([start_pos[0], start_pos[1], target_height])
        print("Takeoff complete - maintained target orientation")
        return True
    
    def land(self):
        """Land the drone - Real Tello takes ~2-3 seconds"""
        if not self.env:
            return False
            
        print("Landing...")
        
        # Get current position
        obs = self.env._getDroneStateVector(self.drone_id)
        start_height = obs[2]
        
        # Tello landing sequence - faster descent
        freq = self._get_freq()
        landing_time = max(2.0, start_height * 1.5)  # Faster descent
        landing_steps = int(landing_time * freq)
        
        for i in range(landing_steps):
            obs = self.env._getDroneStateVector(self.drone_id)
            
            # Smooth descent with ease-in curve
            progress = i / landing_steps
            smooth_progress = progress ** 0.7  # Faster initial descent
            target_height = start_height * (1 - smooth_progress) + 0.02 * smooth_progress
            target_pos = np.array([obs[0], obs[1], target_height])
            target_rpy = np.array([0, 0, 0])
            
            # Compute control action
            action, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self._get_ctrl_timestep(),
                state=obs,
                target_pos=target_pos,
                target_rpy=target_rpy
            )
            
            # Step simulation with timing
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
        
        # Quick final touchdown
        for i in range(30):  # 0.125 seconds
            action = np.array([0, 0, 0, 0])  # Zero thrust
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
        
        self.is_flying = False
        print("Landed")
        return True
    
    def hover(self, duration=1.0):
        """Hover in place for specified duration"""
        if not self.env:
            return
            
        print(f"Hovering for {duration} seconds...")
        
        # Get current position and maintain it
        obs = self.env._getDroneStateVector(self.drone_id)
        hover_pos = obs[0:3]
        
        freq = self._get_freq()
        steps = int(duration * freq)
        
        for i in range(steps):
            obs = self.env._getDroneStateVector(self.drone_id)
            
            # Compute control action to maintain position
            action, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self._get_ctrl_timestep(),
                state=obs,
                target_pos=hover_pos,
                target_rpy=self.target_rpy
            )
            
            # Step simulation with timing
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
    
    def _move_to_position(self, target_pos, duration=2.0):
        """Move to target position smoothly - Real Tello timing"""
        if not self.env:
            return
            
        print(f"Moving over {duration} seconds...")
        
        # Get current position
        obs = self.env._getDroneStateVector(self.drone_id)
        start_pos = obs[0:3]
        
        freq = self._get_freq()
        steps = int(duration * freq)
        
        for i in range(steps):
            obs = self.env._getDroneStateVector(self.drone_id)
            
            # Smooth interpolation with ease-in-out curve
            progress = i / steps
            # S-curve for smooth acceleration and deceleration
            smooth_progress = 3 * progress**2 - 2 * progress**3
            smooth_pos = start_pos + (target_pos - start_pos) * smooth_progress
            
            # Compute control action
            action, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self._get_ctrl_timestep(),
                state=obs,
                target_pos=smooth_pos,
                target_rpy=self.target_rpy
            )
            
            # Step simulation with timing
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
        
        # Update target position
        self.target_pos = target_pos.copy()
    
    def move_forward(self, distance_cm):
        """Move forward by distance in cm - Real Tello speed: ~1.5-2.5 seconds for 50cm"""
        if not self.env:
            return
            
        print(f"Moving forward {distance_cm}cm...")
        
        obs = self.env._getDroneStateVector(self.drone_id)
        current_pos = obs[0:3]
        # Use drone's yaw to calculate true forward direction
        current_yaw = obs[9]
        dx = np.cos(current_yaw) * (distance_cm / 100.0)
        dy = np.sin(current_yaw) * (distance_cm / 100.0)
        target_pos = current_pos + np.array([dx, dy, 0])
        
        # Calculate realistic duration based on distance
        duration = max(1.0, distance_cm / 30.0)  # 30 cm/second
        self._move_to_position(target_pos, duration=duration)
    
    def move_back(self, distance_cm):
        """Move back by distance in cm"""
        self.move_forward(-distance_cm)
    
    def move_left(self, distance_cm):
        """Move left by distance in cm"""
        if not self.env:
            return
            
        print(f"Moving left {distance_cm}cm...")
        
        obs = self.env._getDroneStateVector(self.drone_id)
        current_pos = obs[0:3]
        # Use drone's yaw to calculate true left direction
        current_yaw = obs[9]
        dx = -np.sin(current_yaw) * (distance_cm / 100.0)
        dy = np.cos(current_yaw) * (distance_cm / 100.0)
        target_pos = current_pos + np.array([dx, dy, 0])
        
        duration = max(1.0, distance_cm / 30.0)
        self._move_to_position(target_pos, duration=duration)
    
    def move_right(self, distance_cm):
        """Move right by distance in cm"""
        self.move_left(-distance_cm)
    
    def move_up(self, distance_cm):
        """Move up by distance in cm - Vertical movement is faster"""
        if not self.env:
            return
            
        print(f"Moving up {distance_cm}cm...")
        
        obs = self.env._getDroneStateVector(self.drone_id)
        current_pos = obs[0:3]
        target_pos = current_pos + np.array([0, 0, distance_cm/100.0])
        
        # Vertical movement is faster - 40 cm/second
        duration = max(0.8, distance_cm / 40.0)
        self._move_to_position(target_pos, duration=duration)
    
    def move_down(self, distance_cm):
        """Move down by distance in cm"""
        if not self.env:
            return
            
        print(f"Moving down {distance_cm}cm...")
        
        obs = self.env._getDroneStateVector(self.drone_id)
        current_pos = obs[0:3]
        new_height = max(0.1, current_pos[2] - distance_cm/100.0)
        target_pos = np.array([current_pos[0], current_pos[1], new_height])
        
        duration = max(0.8, distance_cm / 40.0)
        self._move_to_position(target_pos, duration=duration)
    
    def rotate_clockwise(self, degrees):
        """Rotate clockwise by degrees - Real Tello: ~1-2 seconds for 90°"""
        if not self.env:
            return
            
        print(f"Rotating clockwise {degrees}°...")
        
        obs = self.env._getDroneStateVector(self.drone_id)
        current_pos = obs[0:3]
        current_yaw = obs[9]  # Current yaw angle
        
        target_yaw = current_yaw - np.radians(degrees)
        target_rpy = np.array([0, 0, target_yaw])
        
        # Realistic rotation timing - Tello rotates at ~60-90°/second
        rotation_speed = 75  # degrees per second
        duration = max(0.5, abs(degrees) / rotation_speed)
        
        freq = self._get_freq()
        steps = int(duration * freq)
        
        for i in range(steps):
            obs = self.env._getDroneStateVector(self.drone_id)
            
            # Smooth rotation with ease-in-out
            progress = i / steps
            smooth_progress = 3 * progress**2 - 2 * progress**3
            smooth_yaw = current_yaw + (target_yaw - current_yaw) * smooth_progress
            smooth_rpy = np.array([0, 0, smooth_yaw])
            
            # Compute control action
            action, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self._get_ctrl_timestep(),
                state=obs,
                target_pos=current_pos,
                target_rpy=smooth_rpy
            )
            
            # Step simulation with timing
            obs, reward, terminated, truncated, info = self._step_with_timing(action)
        
        self.target_rpy = target_rpy
    
    def rotate_counter_clockwise(self, degrees):
        """Rotate counter-clockwise by degrees"""
        self.rotate_clockwise(-degrees)
    
    def set_real_time_mode(self, enabled=True):
        """Enable or disable real-time simulation"""
        self.real_time_mode = enabled
        print(f"Real-time mode: {'Enabled' if enabled else 'Disabled'}")
    
    def get_height(self):
        """Get current height in cm"""
        if not self.env:
            return 0
            
        obs = self.env._getDroneStateVector(self.drone_id)
        return int(obs[2] * 100)
    
    def get_battery(self):
        """Get simulated battery level"""
        return np.random.randint(70, 100)
    
    def get_position(self):
        """Get current position in meters"""
        if not self.env:
            return np.array([0, 0, 0])
            
        obs = self.env._getDroneStateVector(self.drone_id)
        return obs[0:3]
    
    def get_orientation(self):
        """Get current orientation as a quaternion [x, y, z, w]"""
        if not self.env:
            return np.array([0, 0, 0, 1])
            
        obs = self.env._getDroneStateVector(self.drone_id)
        return obs[3:7]

    def get_camera_frame(self):
        """Get the latest camera frame from the drone's front-facing camera."""
        if not self.camera_enabled:
            print("⚠️  Camera not enabled. Initialize with camera_enabled=True")
            return None
        
        # If we don't have a cached frame, capture one now
        if self.last_frame is None:
            self.last_frame = self._get_camera_image()
        
        return self.last_frame
    
    def capture_fresh_frame(self):
        """Capture a fresh camera frame immediately - OPTIMIZED"""
        if not self.camera_enabled:
            return None
        
        # Force immediate camera update regardless of frame skip
        self.last_frame = self._get_camera_image()
        self.camera_step_counter = 0  # Reset counter
        return self.last_frame
    
    def set_camera_quality(self, width=320, height=240, frame_skip=3):
        """Adjust camera quality vs performance trade-off"""
        self.camera_width = width
        self.camera_height = height
        self.camera_frame_skip = frame_skip
        print(f"📹 Camera set to {width}x{height}, updating every {frame_skip} frames")
        
    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        """Send RC control commands (-100 to 100)"""
        if not self.env:
            return
            
        # Convert RC commands to position adjustments
        scale = 0.003  # Slightly more responsive
        
        obs = self.env._getDroneStateVector(self.drone_id)
        current_pos = obs[0:3]
        
        # Apply incremental position changes
        pos_delta = np.array([
            forward_backward * scale,
            -left_right * scale,
            up_down * scale
        ])
        
        new_target = current_pos + pos_delta
        new_target[2] = max(0.1, min(3.0, new_target[2]))  # Altitude limits
        
        # Apply for one control step
        action, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self._get_ctrl_timestep(),
            state=obs,
            target_pos=new_target,
            target_rpy=self.target_rpy
        )
        
        # Step simulation with timing
        obs, reward, terminated, truncated, info = self._step_with_timing(action)
    
    def disconnect(self):
        """Close the environment"""
        if self.env:
            self.env.close()
        print("Disconnected from gym-pybullet-drones")