# Sentinel

A real-time multi-object tracking system that combines YOLO object detection with Kalman filtering for robust person tracking in video streams.

## Features

- **Real-time Person Detection**: Uses YOLO11 model for accurate person detection
- **Kalman Filter Tracking**: Implements Kalman filters for smooth trajectory prediction and tracking
- **Multi-Object Tracking**: Tracks multiple persons simultaneously with unique IDs
- **Configurable Parameters**: Adjustable confidence thresholds, IoU values, and tracking parameters
- **Visual Feedback**: Real-time visualization with bounding boxes, centroids, and predicted trajectories

## Architecture

The system consists of several key components:

- `detection.py`: YOLO-based object detection module
- `tracker.py`: Multi-object tracker using Kalman filters
- `kalman.py`: Kalman filter implementation for bounding box tracking
- `video_source.py`: Video input handling
- `utils.py`: Visualization and utility functions
- `main.py`: Main application entry point

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- SciPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Sentinel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLO model (yolo11m.pt should be in the project directory)

## Usage

Run the tracking system:
```bash
python main.py
```

The system will process the video file `test_cropped.mp4` and display:
- **Green boxes**: YOLO detections
- **Red dots**: Detection centroids
- **Blue elements**: Kalman filter predictions

Press 'q' to quit the application.

## Configuration

Modify `detection_conf.json` to adjust detection parameters:

```json
{
    "model_path": "yolo11m.pt",
    "class_id": [0],
    "conf": 0.5,
    "iou": 0.45,
    "class_name": "person"
}
```

- `conf`: Confidence threshold for detections (0.0-1.0)
- `iou`: Intersection over Union threshold for NMS
- `class_id`: COCO class IDs to detect (0 = person)

## Tracking Parameters

The tracker can be configured in `main.py`:

```python
tracker = PersonTracker(max_distance=100.0, max_age=30)
```

- `max_distance`: Maximum distance for associating detections with tracks
- `max_age`: Maximum frames a track can exist without detections

## Project Structure

```
Sentinel/
├── main.py              # Main application
├── detection.py         # YOLO detection module
├── tracker.py           # Multi-object tracker
├── kalman.py           # Kalman filter implementation
├── video_source.py     # Video input handling
├── utils.py            # Visualization utilities
├── detection_conf.json # Detection configuration
├── requirements.txt    # Python dependencies
├── yolo11m.pt         # YOLO model weights
└── test_cropped.mp4   # Test video file
```

## License

This project is open source and available under the MIT License.
