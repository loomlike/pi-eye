import logging
from time import time

import cv2
from IPython.display import display, Image


logger = logging.getLogger(__name__)


def play(video: cv2.VideoCapture, fps: int = 30):
    display_handle = display(None, display_id=True)

    logger.info(f"Playing video at {fps} fps")
    try:
        while True:
            st = time()
            _, frame = video.read()
            _, frame = cv2.imencode(".jpeg", frame)
            display_handle.update(Image(data=frame.tobytes()))
            time_elapsed = time() - st
            sleep_time = 1/fps - time_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
