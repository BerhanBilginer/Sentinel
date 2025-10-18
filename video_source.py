import cv2

class video_source:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()
    
    def is_opened(self):
        return self.cap.isOpened()
    
    def get(self, prop_id):
        return self.cap.get(prop_id)
    
    def set(self, prop_id, value):
        self.cap.set(prop_id, value)