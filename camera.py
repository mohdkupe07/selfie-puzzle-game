"""
camera.py
---------
Handles webcam initialization, frame capture, and release.
Provides a clean interface for accessing the user's webcam with error handling.
"""

import cv2
import numpy as np


class Camera:
    """
    Manages webcam access using OpenCV.

    Attributes:
        camera_index (int): Index of the webcam device (default: 0).
        width (int): Desired frame width.
        height (int): Desired frame height.
        fps (int): Desired frames per second.
        cap (cv2.VideoCapture): OpenCV video capture object.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize the Camera object.

        Args:
            camera_index: Index of the webcam (0 = default).
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frames per second.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None  # VideoCapture will be initialized in open()

    def open(self) -> bool:
        """
        Open the webcam and configure its properties.

        Returns:
            True if the camera opened successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open webcam at index {self.camera_index}.")
            return False

        # Set resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read actual values (camera may not support requested resolution)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Camera opened: {self.width}x{self.height} @ {self.fps} FPS")
        return True

    def read_frame(self):
        """
        Capture a single frame from the webcam.

        Returns:
            Tuple (success: bool, frame: np.ndarray or None).
            The frame is horizontally flipped (mirror effect) for natural interaction.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Flip horizontally so the user sees a mirror view
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        """Release the webcam resource."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[INFO] Camera released.")

    def is_opened(self) -> bool:
        """Check if the camera is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def get_resolution(self):
        """
        Return the current camera resolution.

        Returns:
            Tuple (width: int, height: int).
        """
        return self.width, self.height
