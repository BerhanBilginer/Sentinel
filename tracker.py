import numpy as np
from kalman import KalmanBBox, KalmanParams
from scipy.spatial.distance import cdist

class PersonTracker:
    def __init__(self, max_distance=50.0, max_age=30):
        self.kalman_params = KalmanParams()
        self.tracks = {}  # track_id -> KalmanBBox
        self.track_ages = {}  # track_id -> age (frames since last update)
        self.next_id = 0
        self.max_distance = max_distance
        self.max_age = max_age
        
    def bbox_to_centroid_and_size(self, box):
        """Convert x1,y1,x2,y2 to cx,cy,w,h"""
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])
    
    def update(self, detections):
        """Update tracker with new detections"""
        # Convert detections to centroids
        if not detections:
            # No detections - just predict existing tracks
            self._predict_all()
            self._age_tracks()
            self._remove_old_tracks()
            return
            
        detection_centroids = [self.bbox_to_centroid_and_size(det) for det in detections]
        
        # Predict all existing tracks
        self._predict_all()
        
        # Get current track positions for association
        if self.tracks:
            track_positions = []
            track_ids = list(self.tracks.keys())
            
            for track_id in track_ids:
                cx, cy = self.tracks[track_id].center()
                track_positions.append([cx, cy])
            
            track_positions = np.array(track_positions)
            detection_positions = np.array([[d[0], d[1]] for d in detection_centroids])
            
            # Calculate distance matrix
            distances = cdist(track_positions, detection_positions)
            
            # Simple greedy assignment
            used_detections = set()
            used_tracks = set()
            
            for i in range(len(track_ids)):
                if len(used_detections) >= len(detection_centroids):
                    break
                    
                # Find closest detection for this track
                valid_detections = [j for j in range(len(detection_centroids)) 
                                 if j not in used_detections]
                
                if not valid_detections:
                    break
                    
                closest_det_idx = min(valid_detections, 
                                    key=lambda j: distances[i, j])
                
                if distances[i, closest_det_idx] < self.max_distance:
                    # Update this track
                    track_id = track_ids[i]
                    self.tracks[track_id].update(detection_centroids[closest_det_idx])
                    self.track_ages[track_id] = 0
                    used_detections.add(closest_det_idx)
                    used_tracks.add(track_id)
            
            # Create new tracks for unmatched detections
            for i, detection in enumerate(detection_centroids):
                if i not in used_detections:
                    self._create_new_track(detection)
        else:
            # No existing tracks - create new ones for all detections
            for detection in detection_centroids:
                self._create_new_track(detection)
        
        # Age unmatched tracks
        self._age_tracks()
        self._remove_old_tracks()
    
    def _predict_all(self):
        """Predict all existing tracks"""
        for track in self.tracks.values():
            track.predict()
    
    def _create_new_track(self, detection):
        """Create new Kalman track"""
        kalman = KalmanBBox(self.kalman_params)
        kalman.initialize(detection)
        self.tracks[self.next_id] = kalman
        self.track_ages[self.next_id] = 0
        self.next_id += 1
    
    def _age_tracks(self):
        """Increment age for all tracks"""
        for track_id in self.tracks:
            self.track_ages[track_id] += 1
    
    def _remove_old_tracks(self):
        """Remove tracks that are too old"""
        to_remove = [track_id for track_id, age in self.track_ages.items() 
                    if age > self.max_age]
        
        for track_id in to_remove:
            del self.tracks[track_id]
            del self.track_ages[track_id]
    
    def get_predictions(self):
        """Get current predictions from all tracks"""
        predictions = []
        for track_id, kalman in self.tracks.items():
            cx, cy = kalman.center()
            bbox = kalman.bbox()  # [cx, cy, w, h]
            predictions.append({
                'track_id': track_id,
                'center': (cx, cy),
                'bbox': bbox,
                'age': self.track_ages[track_id]
            })
        return predictions
