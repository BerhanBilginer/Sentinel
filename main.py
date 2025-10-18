from detection import detection
from video_source import video_source
from tracker import PersonTracker
import cv2
import json
    
from utils import draw_boxes, centroids, draw_kalman_predictions

def main():
    # Load configuration
    with open('detection_conf.json', 'r') as f:
        config = json.load(f)
    
    # Initialize detector, tracker and video source with proper parameters
    detector = detection(config['model_path'])
    tracker = PersonTracker(max_distance=100.0, max_age=30)
    video = video_source('test_cropped.mp4')  # Use video file

    # Check if video source is opened
    if not video.is_opened():
        print("Error: Could not open video source")
        return

    while True:
        frame = video.get_frame()
        
        # Validate frame
        if frame is None:
            print("No frame received, ending...")
            break
            
        results, obj = detector.detect(frame, config['conf'], config['iou'], config['class_id'])
        
        # Update tracker with detections
        tracker.update(obj)
        
        # Get Kalman predictions
        predictions = tracker.get_predictions()
        
        # Display frame with detections and predictions
        frame = draw_boxes(frame, obj)  # Green boxes for YOLO detections
        frame = centroids(frame, obj)   # Red dots for YOLO centroids
        frame = draw_kalman_predictions(frame, predictions)  # Blue for Kalman predictions
        cv2.imshow('Sentinel', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()