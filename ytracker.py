import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
import types

def main():
    # Hardcoded path to the video file
    video_path = "tesla-drift.mp4"  # Change this to your desired path

    # Load YOLOv8 model (make sure you have a detection model, e.g., yolov8n.pt or your own)
    model = YOLO("yolov8n.pt")  # Change to your model if needed

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Skip first 9 frames to get to the 10th frame (frame index 9 if 0-based)
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

    if bbox == (0,0,0,0):
        print("No bounding box selected. Exiting.")
        return

    print("Frame shape:", frame.shape)
    print("Selected bbox:", bbox)

    # Run YOLO detection on the 10th frame to find the class of the selected object
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    scores = results[0].boxes.conf.cpu().numpy()      # (N,)
    classes = results[0].boxes.cls.cpu().numpy()      # (N,)

    # Find the detection that overlaps most with the selected ROI
    x, y, w, h = bbox
    roi_box = np.array([x, y, x + w, y + h])
    ious = []
    for det_box in detections:
        # Compute IoU
        xx1 = max(roi_box[0], det_box[0])
        yy1 = max(roi_box[1], det_box[1])
        xx2 = min(roi_box[2], det_box[2])
        yy2 = min(roi_box[3], det_box[3])
        inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        roi_area = (roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1])
        det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
        union_area = roi_area + det_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        ious.append(iou)
    if not ious or max(ious) < 0.1:
        print("No YOLO detection matches the selected ROI. Exiting.")
        return
    best_idx = int(np.argmax(ious))
    target_class = int(classes[best_idx])
    print(f"Tracking object of class {target_class} (YOLO class index)")

    # ---- FIX: Create args for BYTETracker manually ----
    args = types.SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        aspect_ratio_thresh=1.6,
        min_box_area=10,
        mot20=False
    )
    tracker = BYTETracker(args)
    # ---------------------------------------------------

    # Rewind to 10th frame for tracking
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9)

    target_id = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Run YOLOv8 detection+tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Filter tracks by class
        boxes = results[0].boxes
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else []
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        xyxys = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []

        # Draw only the tracks of the target class
        for box, tid, cls in zip(xyxys, track_ids, classes):
            if int(cls) == target_class:
                x1, y1, x2, y2 = [int(v) for v in box]
                color = (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, 1)
                cv2.putText(frame, f"ID {int(tid)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()