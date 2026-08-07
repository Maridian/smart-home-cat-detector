import cv2
import time

class RTSPStream:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def read_frame(self):
        """Reads a frame from the stream; attempts reconnect if offline."""
        if not self.cap.isOpened():
            self._reconnect()

        ret, frame = self.cap.read()
        if not ret:
            print("Stream lost. Reconnecting...")
            self._reconnect()
            return False, None
            
        return True, frame

    def _reconnect(self):
        time.sleep(2)
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def release(self):
        if self.cap:
            self.cap.release()