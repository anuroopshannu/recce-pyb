import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from norfair import Detection, Tracker
from norfair.distances import create_normalized_mean_euclidean_distance

class TF2Detector:
    """TensorFlow 2 Object Detector using TensorFlow Hub"""
    
    def __init__(self, model_url="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"):
        print("Loading TensorFlow 2 model...")
        self.detector = hub.load(model_url)
        print("Model loaded successfully!")
        
        # COCO class names
        self.class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, image, confidence_threshold=0.5):
        """Run detection on image and return detections"""
        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_image)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run detection
        detections = self.detector(input_tensor)
        
        # Extract results
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        
        # Filter by confidence
        valid_detections = scores > confidence_threshold
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        results = []
        
        for i in range(len(boxes)):
            if valid_detections[i]:
                box = boxes[i]
                y1, x1, y2, x2 = box
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': classes[i],
                    'score': scores[i],
                    'class_name': self.class_names[classes[i]] if classes[i] < len(self.class_names) else 'unknown'
                })
        
        return results

def non_max_suppression(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence score (descending)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep = []
    while detections:
        # Take the highest confidence detection
        current = detections.pop(0)
        keep.append(current)
        
        # Remove detections with high IoU overlap
        detections = [det for det in detections 
                     if compute_iou(current['bbox'], det['bbox']) < iou_threshold]
    
    return keep

def detections_to_norfair(detections_list, target_class=None):
    """Convert TF2 detections to Norfair Detection objects with enhanced features"""
    norfair_detections = []
    
    # Apply NMS to reduce duplicate detections
    filtered_detections = non_max_suppression(detections_list, iou_threshold=0.4)
    
    for detection in filtered_detections:
        # Filter by target class if specified
        if target_class is not None and detection['class'] != target_class:
            continue
            
        x1, y1, x2, y2 = detection['bbox']
        
        # Use multiple points for more robust tracking
        # This helps maintain track identity when bbox changes slightly
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        
        # Add corner points for better geometric matching
        width = x2 - x1
        height = y2 - y1
        
        # Create multiple tracking points
        points = np.array([
            [centroid_x, centroid_y],                           # Center
            [x1 + width * 0.25, y1 + height * 0.25],          # Top-left quarter
            [x2 - width * 0.25, y1 + height * 0.25],          # Top-right quarter
            [x1 + width * 0.25, y2 - height * 0.25],          # Bottom-left quarter
            [x2 - width * 0.25, y2 - height * 0.25],          # Bottom-right quarter
        ])
        
        # Create appearance descriptor for re-identification
        appearance_descriptor = np.array([
            width / height,  # Aspect ratio
            width,           # Width
            height,          # Height
            detection['score']  # Confidence
        ])
        
        norfair_detection = Detection(
            points=points,
            scores=np.array([detection['score']] * len(points)),
            data={
                'class': detection['class'],
                'class_name': detection['class_name'],
                'bbox': detection['bbox'],
                'score': detection['score'],
                'appearance': appearance_descriptor,
                'area': width * height
            }
        )
        norfair_detections.append(norfair_detection)
    
    return norfair_detections

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes"""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection coordinates
    xx1 = max(x1, x1_2)
    yy1 = max(y1, y1_2)
    xx2 = min(x2, x2_2)
    yy2 = min(y2, y2_2)
    
    # Intersection area
    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    
    # Union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def main():
    """Main tracking function using TensorFlow 2 and Norfair"""
    # Hardcoded path to the video file
    video_path = "tesla-drift.mp4"  # Change this to your desired path

    # Initialize TF2 detector
    detector = TF2Detector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Skip first 9 frames to get to the 10th frame
    for _ in range(9):
        ret = cap.grab()
        if not ret:
            print("Video ended before 10th frame")
            return

    # Read the 10th frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading the 10th frame")
        return

    # Let the user select the bounding box on the 10th frame
    print("Select the object to track on the 10th frame and press ENTER or SPACE")
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    if bbox == (0, 0, 0, 0):
        print("No bounding box selected. Exiting.")
        return

    print("Frame shape:", frame.shape)
    print("Selected bbox:", bbox)

    # Run TF2 detection on the 10th frame to find the class of the selected object
    detections = detector.detect(frame, confidence_threshold=0.3)

    # Find the detection that overlaps most with the selected ROI
    x, y, w, h = bbox
    roi_box = [x, y, x + w, y + h]
    
    best_iou = 0
    target_class = None
    target_class_name = None
    
    for detection in detections:
        det_box = detection['bbox']
        iou = compute_iou(roi_box, det_box)
        
        if iou > best_iou:
            best_iou = iou
            target_class = detection['class']
            target_class_name = detection['class_name']

    if best_iou < 0.1:
        print("No TF2 detection matches the selected ROI. Exiting.")
        return

    print(f"Tracking object of class {target_class} ({target_class_name}) with IoU: {best_iou:.3f}")

    # Initialize Norfair tracker with ONLY the key parameters for ID preservation
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,         # Same as before
        hit_counter_max=10,            # Increased from 5 to keep tracks alive longer
        initialization_delay=1,         # Same as before
        pointwise_hit_counter_max=4    # Same as before
    )

    # Rewind to 10th frame for tracking
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        frame_count += 1

        # Run TF2 detection
        detections = detector.detect(frame, confidence_threshold=0.4)
        
        # Convert to Norfair detections
        norfair_detections = detections_to_norfair(detections, target_class=target_class)
        
        # Update tracker
        tracked_objects = tracker.update(detections=norfair_detections)

        # Draw all detections of target class in blue (after NMS)
        filtered_detections = non_max_suppression(
            [d for d in detections if d['class'] == target_class], 
            iou_threshold=0.4
        )
        
        for detection in filtered_detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame, f"{detection['class_name']} {detection['score']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw tracked objects in green
        for tracked_obj in tracked_objects:
            if tracked_obj.last_detection is not None and tracked_obj.last_detection.data is not None:
                # Get bbox from detection data
                bbox = tracked_obj.last_detection.data['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw tracking box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw tracking points
                for point in tracked_obj.last_detection.points:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                
                # Draw track ID
                cv2.putText(frame, f"ID {tracked_obj.id}", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show confidence
                confidence = tracked_obj.last_detection.data['score']
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracking: {target_class_name} (Class {target_class})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(tracked_objects)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(filtered_detections)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("TF2 + Norfair Tracking", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Pause
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking completed!")

if __name__ == "__main__":
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    main()