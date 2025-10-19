# Sentinel - Enhanced Multi-Object Tracking

A real-time multi-object tracking system that combines YOLO11 object detection with advanced Kalman filtering, multi-gate filtering, Hungarian association, and ReID appearance features for robust person tracking in video streams.

## ✨ Features

- **🎯 Real-time Person Detection**: YOLO11m model for accurate person detection
- **🔄 Advanced Kalman Filtering**: Variable-dt Kalman filter with Constant Velocity (CV) model for position and Random Walk (RW) for size
- **🚪 Multi-Gate Filtering**: Motion (Mahalanobis), IoU, scale, and aspect ratio gates to prevent impossible associations
- **🎲 Hungarian Association**: Optimal assignment using Hungarian algorithm with multi-stage matching
- **👤 ReID Appearance Features**: OSNet-AIN deep learning features for appearance-based matching (512-dim embeddings)
- **🆔 Stable Track IDs**: Maintains consistent IDs across frames for each tracked person
- **🎨 Rich Visualization**: Multiple display modes with real-time debug overlays
- **⚙️ Highly Configurable**: Tune gating, association, and Kalman parameters for different scenarios

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Sentinel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO model and ReID weights:
   - Place `yolo11m.pt` in project root
   - Place `osnet_ain_ms_d_c.pth.tar` in `weights/` directory

### Run the Tracker

```bash
python main.py
```

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| **q** | Quit |
| **d** | Toggle YOLO detection boxes |
| **t** | Toggle track predictions (blue boxes with IDs) |
| **g** | Toggle gating visualization (green lines) |
| **a** | Toggle association visualization (cyan lines) |
| **l** | Toggle legend |
| **i** | Toggle ID panel (top-left info box) |
| **o** | Toggle ID-only mode (large IDs, clean background) |
| **SPACE** | Pause/Resume |

## 📊 Visualization

### Default View
The system shows **track IDs with colored bounding boxes**:
- **Blue boxes**: Active tracks (recently updated)
- **Orange boxes**: Tracks not updated recently (1-5 frames)
- **Gray boxes**: Stale tracks (>5 frames without update)
- **Track ID**: Displayed above each track

### Debug Modes
Press **`g`** to see gating (green lines show which tracks can match which detections)  
Press **`a`** to see association (cyan=matched, yellow=new, magenta=lost)  
Press **`l`** to show color legend  
Press **`i`** to show ID panel with track statistics  
Press **`o`** for presentation mode (huge IDs only)

## ⚙️ Configuration

### Detection Settings (`detection_conf.json`)

```json
{
    "model_path": "yolo11m.pt",
    "class_id": [0],           // Person only (COCO class 0)
    "conf": 0.5,               // Detection confidence threshold
    "iou": 0.45,               // NMS IoU threshold
    "class_name": "person"
}
```

### Tracking Parameters (`main.py`)

#### Gating Parameters
Controls which track-detection pairs are considered:
```python
GatingParams(
    dof=4,                      # Degrees of freedom for chi-square
    chi2_p=0.99,                # Confidence for motion gate
    tau_motion_override=1000.0, # Motion gate threshold (disabled for stability)
    tau_iou=0.01,               # IoU threshold (1% overlap required)
    tau_log_s=2.0,              # Scale consistency threshold
    tau_ratio=1.5               # Aspect ratio consistency threshold
)
```

#### Association Parameters
Controls how costs are weighted and fused:
```python
AssocParams(
    w_m0=1.0,                   # Motion weight baseline
    w_i0=1.5,                   # IoU weight baseline
    w_a0=0.5,                   # Appearance (ReID) weight baseline
    alpha=0.3,                  # Motion weight decay with low confidence
    beta=0.5,                   # Appearance weight boost with low confidence
    stage1_conf_thr=0.6,        # High-confidence first-pass threshold
    l2_norm=150.0,              # L2 distance normalization
    app_default_cost=1.0,       # Cost when appearance feature missing
    gamma_dt=0.2,               # Motion weight reduction with large dt
    delta_dt=0.15               # Appearance weight boost with large dt
)
```

#### Kalman Filter Parameters
Controls motion model and noise:
```python
KalmanParams(
    dt=0.033,                   # Nominal frame interval (seconds)
    q_acc=1.0,                  # Process noise: position (px/s²)²
    q_w=0.05,                   # Process noise: width
    q_h=0.05,                   # Process noise: height
    r_epsilon=1.0               # Measurement noise (pixels²)
)
```

#### Tracker Settings
```python
PersonTracker(
    max_age=15,                 # Delete track after N frames without match
    reid_momentum=0.6,          # Feature EMA momentum (0=new only, 1=old only)
    reid_debug=False,           # Enable ReID debug logs
    reid_log_interval=10        # Log every N frames
)
```

## 🏗️ Architecture

### System Pipeline

```
Frame → YOLO Detection → Kalman Predict → Gating → Association → Kalman Update
                              ↓               ↓          ↓
                         Track State    Filter Pairs  Hungarian
                                                          ↓
                                                    Update Tracks
                                                    Create New
                                                    Age Unmatched
```

### Components

#### 1. **Detection** (`detection.py`)
- YOLO11m wrapper
- Person-only filtering
- Confidence thresholding

#### 2. **Kalman Filter** (`kalman.py`)
- State: `[cx, cy, vx, vy, w, h]`
- Constant Velocity (CV) model for position
- Random Walk (RW) model for size
- Variable dt support for frame drops

#### 3. **Gating** (`gating.py`)
- **Motion Gate**: Mahalanobis distance with chi-square threshold
- **IoU Gate**: Bounding box overlap
- **Scale Gate**: Size consistency (log scale difference)
- **Ratio Gate**: Aspect ratio consistency

#### 4. **Association** (`association.py`)
- **Stage 1**: High-confidence detections (IoU + motion heavy)
- **Stage 2**: Low-confidence detections (appearance heavy)
- **Hungarian Algorithm**: Optimal assignment (scipy)
- **Dynamic Weighting**: Confidence and dt-aware cost fusion

#### 5. **ReID Encoder** (`reid_encoder.py`)
- **Model**: OSNet-AIN x1.0
- **Features**: 512-dim L2-normalized embeddings
- **Batch Processing**: Efficient GPU inference
- **EMA Tracking**: Smooth feature updates with momentum

#### 6. **Tracker** (`tracker.py`)
- Track lifecycle management
- Feature EMA updates
- Debug info generation

## 📁 Project Structure

```
Sentinel/
├── main.py                       # Main tracking loop
├── detection.py                  # YOLO wrapper
├── tracker.py                    # PersonTracker with ReID
├── kalman.py                     # Kalman filter (CV + RW)
├── gating.py                     # Multi-gate filtering
├── association.py                # Hungarian matching
├── reid_encoder.py               # OSNet feature extraction
├── utils.py                      # Visualization utilities
├── video_source.py               # Video input handling
├── detection_conf.json           # Detection configuration
├── ID_STABILITY_FIX.md          # Track ID stability documentation
├── .gitignore                    # Git ignore rules
└── requirements.txt              # Python dependencies

# Not in repo (gitignored):
├── weights/
│   └── osnet_ain_ms_d_c.pth.tar # ReID model weights
├── yolo11m.pt                    # YOLO model weights
└── *.mp4                         # Video files
```

## 🔧 Advanced Tuning

### For Fast Motion
Tracks lost frequently:
```python
# Relax gating
tau_iou=0.001  # Very permissive
q_acc=2.0      # Higher process noise
```

### For Crowded Scenes
Many occlusions and crossings:
```python
# Rely more on appearance
w_a0=1.5       # Higher appearance weight
max_age=30     # Keep tracks longer
```

### For Stable Cameras
Minimal camera motion:
```python
# Stricter matching
tau_iou=0.10   # Higher IoU required
q_acc=0.5      # Lower process noise
```

### For ID Switches
Tracks swapping identities:
```python
# Increase appearance reliance
w_a0=1.0       # More appearance weight
reid_momentum=0.7  # Slower feature updates
```

## 📊 Performance

### Benchmarks (NVIDIA GPU)
- **YOLO11m**: ~30-35ms per frame
- **ReID Encoding**: ~3-5ms per detection
- **Tracking Pipeline**: ~2-3ms per frame
- **Total Latency**: ~35-40ms per frame (**~25-28 FPS**)

### Resource Usage
- **GPU Memory**: ~2-3 GB (YOLO + ReID)
- **CPU**: 1-2 cores at 50-70%
- **RAM**: ~500-800 MB

## 🐛 Troubleshooting

### Track IDs Keep Changing
**Solution**: See `ID_STABILITY_FIX.md` for detailed analysis. Key fixes:
- Increased `r_epsilon` from 1e-9 to 1.0
- Disabled motion gate with `tau_motion_override=1000.0`
- Relaxed all gating thresholds

### High Track ID Numbers (100+)
**Cause**: Tracks not matching → new tracks created every frame  
**Solution**: Relax gating parameters or increase `max_age`

### Tracks Jumping Between People
**Cause**: Association confusion in crowded scenes  
**Solution**: Increase `w_a0` (appearance weight) or `tau_iou`

### Missing ReID Weights Error
**Cause**: Weight file not found  
**Solution**: Place `osnet_ain_ms_d_c.pth.tar` in `weights/` directory

### Low FPS / Slow Performance
**Cause**: CPU inference or large video resolution  
**Solution**: Use GPU, reduce video resolution, or use lighter YOLO model (yolo11n)

## 📚 Technical Details

### Kalman Filter Math

**State Vector**:
```
x = [cx, cy, vx, vy, w, h]
```

**Transition Matrix** (CV model):
```
F = [1  0  dt 0  0  0]
    [0  1  0  dt 0  0]
    [0  0  1  0  0  0]
    [0  0  0  1  0  0]
    [0  0  0  0  1  0]
    [0  0  0  0  0  1]
```

**Process Noise** (CWNA for position, RW for size):
```
Q_cv = q_acc * [dt⁴/4    0     dt³/2   0   ]
               [0      dt⁴/4    0    dt³/2]
               [dt³/2    0      dt²    0   ]
               [0      dt³/2    0     dt²  ]

Q_rw = [q_w*dt²    0   ]
       [0      q_h*dt²]
```

### Association Cost

**Fused Cost**:
```
C[i,j] = w_m * C_motion[i,j] + w_i * C_iou[i,j] + w_a * C_app[i,j]
```

Where weights are dynamically adjusted based on detection confidence and frame interval (dt).

### ReID Features

**Model**: OSNet-AIN (Attention + Instance Normalization)  
**Input**: 256×128 RGB crops (normalized to ImageNet stats)  
**Output**: 512-dim L2-normalized embeddings  
**Similarity**: Cosine similarity (dot product of normalized vectors)

## 📄 Requirements

### Python Packages
```
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0
ultralytics>=8.0.0
torchreid>=0.2.5
gdown>=5.0.0
tensorboard>=2.14.0
```

### System Requirements
- **Python**: 3.12+ (tested on 3.12.3)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows
- **GPU**: CUDA-capable NVIDIA GPU (recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 500MB for models and code

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO11 implementation
- **TorchReID**: ReID model zoo and OSNet implementation
- **Scipy**: Hungarian algorithm (linear_sum_assignment)
- **OpenCV**: Computer vision utilities

## 📖 References

- [YOLO11 Paper](https://github.com/ultralytics/ultralytics)
- [OSNet Paper](https://arxiv.org/abs/1905.00953)
- [Deep OC-SORT](https://arxiv.org/abs/2302.11813) - Inspiration for appearance integration
- [SORT](https://arxiv.org/abs/1602.00763) - Original tracking framework

---

**Status**: ✅ Fully operational with stable track IDs and ReID appearance features

**Version**: 2.0 - Enhanced MOT with ReID (October 2025)
