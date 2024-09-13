import threading
from time import sleep
from typing import Tuple
import warnings

try:
    from libcamera import controls
    from picamera2 import Picamera2
    is_libcamera = True
except ModuleNotFoundError:
    warnings.warn("libcamera module not found. Use openCV instead.")
    is_libcamera = False


class VideoCapture(threading.Thread):
    def __init__(self, cam, resolution):
        super().__init__()
        self.resolution = resolution
        self.cam = cam
        self.frame = None
        self.capture = False
    
    def run(self):
        try:
            while self.capture:
                if is_libcamera:
                    self.frame = self.cam.capture_array()
                else:
                    _, self.frame = self.cam.read()
        finally:
            pass

    def start(self):
        self.capture = True
        if is_libcamera:
            self.cam = Picamera2()
            # Picamera2.create_preview_configuration will generate a configuration suitable for displaying camera preview images on the display, or prior to capturing a still image
            # Picamera2.create_still_configuration will generate a configuration suitable for capturing a high-resolution still image
            # Picamera2.create_video_configuration will generate a configuration suitable for recording video files
            self.cam.configure(self.cam.create_preview_configuration(main={"format": 'RGB888', "size": self.resolution}))
            # There are three autoFocus modes: Manual, Auto, and Continuous
            self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            self.cam.start()
        else:
            pass
        sleep(1)  # delay for starting the cam
        super().start()
    
    def stop(self):
        self.capture = False
        self.join()
        if is_libcamera:
            self.cam.stop()
        else:
            self.cam.release()


class Camera():
    def __init__(self, resolution: Tuple = (320, 320)):
        self.resolution = resolution
        self.video_capture = None  # video capture thread

    def start(self):
        self.stop()
        
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])    
        
        self.video_capture = VideoCapture(self.cam, self.resolution)
        self.video_capture.start()
        

    def stop(self):
        if self.video_capture is not None:
            self.video_capture.stop()
            self.video_capture = None

    def get_frame(self):
        if self.video_capture is not None:
            return self.video_capture.frame
        else:
            return None