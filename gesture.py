"""
gesture.py
----------
Handles hand landmark detection using MediaPipe Hands.
Provides pinch gesture detection and finger position tracking.

Pinch Gesture Logic:
- A pinch is detected when the Euclidean distance between the thumb tip
  (landmark 4) and the index finger tip (landmark 8) falls below a threshold.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


class GestureDetector:
    """
    Detects hand landmarks and pinch gestures using MediaPipe Hands.

    Attributes:
        hands: MediaPipe Hands solution object.
        mp_draw: MediaPipe drawing utilities.
        pinch_threshold (float): Distance (normalized) below which a pinch is detected.
        pinch_cooldown (int): Frames to wait between consecutive pinch triggers.
        _cooldown_counter (int): Current cooldown frame count.
    """

    # MediaPipe landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        pinch_threshold: float = 0.06,
        pinch_cooldown: int = 20,
    ):
        """
        Initialize the GestureDetector.

        Args:
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
            pinch_threshold: Normalized distance threshold for pinch detection.
            pinch_cooldown: Frames between consecutive pinch triggers.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.pinch_threshold = pinch_threshold
        self.pinch_cooldown = pinch_cooldown
        self._cooldown_counter = 0

        # Track previous pinch state for edge detection
        self._was_pinching = False

    def process(self, frame: np.ndarray):
        """
        Process a frame and detect hand landmarks.

        Args:
            frame: BGR image frame from OpenCV.

        Returns:
            MediaPipe hand detection results object (may be None if no hands).
        """
        # MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance optimization
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def get_landmark_positions(
        self, results, frame_width: int, frame_height: int
    ) -> Optional[dict]:
        """
        Extract pixel positions of all landmarks for the first detected hand.

        Args:
            results: MediaPipe hand detection results.
            frame_width: Width of the frame in pixels.
            frame_height: Height of the frame in pixels.

        Returns:
            Dictionary mapping landmark index to (x, y) pixel coordinates,
            or None if no hand is detected.
        """
        if not results.multi_hand_landmarks:
            return None

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        positions = {}

        for idx, lm in enumerate(hand_landmarks.landmark):
            # Convert normalized [0,1] coordinates to pixel coordinates
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            positions[idx] = (x, y)

        return positions

    def get_pinch_center(
        self, results, frame_width: int, frame_height: int
    ) -> Optional[Tuple[int, int]]:
        """
        Calculate the midpoint between thumb tip and index finger tip.

        Args:
            results: MediaPipe hand detection results.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            (cx, cy) pixel coordinates of the pinch center, or None if no hand.
        """
        positions = self.get_landmark_positions(results, frame_width, frame_height)
        if positions is None:
            return None

        thumb = positions[self.THUMB_TIP]
        index = positions[self.INDEX_TIP]

        cx = (thumb[0] + index[0]) // 2
        cy = (thumb[1] + index[1]) // 2
        return cx, cy

    def is_pinching(self, results, frame_width: int, frame_height: int) -> bool:
        """
        Determine if a pinch gesture is currently active.

        The pinch is detected when the normalized distance between thumb tip
        and index finger tip is below the configured threshold.

        Args:
            results: MediaPipe hand detection results.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            True if a pinch is currently being made, False otherwise.
        """
        if not results.multi_hand_landmarks:
            return False

        hand_landmarks = results.multi_hand_landmarks[0]
        thumb = hand_landmarks.landmark[self.THUMB_TIP]
        index = hand_landmarks.landmark[self.INDEX_TIP]

        # Compute Euclidean distance in normalized coordinates
        dist = np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)
        return dist < self.pinch_threshold

    def detected_pinch_start(self, results, frame_width: int, frame_height: int) -> bool:
        """
        Detect the leading edge of a pinch (transition from open to pinched).

        This fires ONCE when the pinch first starts, not continuously.
        Includes a cooldown to prevent rapid repeated triggers.

        Args:
            results: MediaPipe hand detection results.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            True on the first frame a pinch is detected (with cooldown), else False.
        """
        # Decrement cooldown counter
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        currently_pinching = self.is_pinching(results, frame_width, frame_height)

        # Detect rising edge: was open, now pinched
        pinch_started = currently_pinching and not self._was_pinching
        self._was_pinching = currently_pinching

        if pinch_started and self._cooldown_counter == 0:
            self._cooldown_counter = self.pinch_cooldown
            return True

        return False

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to draw on.
            results: MediaPipe hand detection results.

        Returns:
            Frame with landmarks drawn.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def draw_pinch_indicator(
        self, frame: np.ndarray, results, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """
        Draw a visual indicator at the pinch center when pinching.

        Args:
            frame: BGR image to draw on.
            results: MediaPipe hand detection results.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Frame with pinch indicator drawn.
        """
        if self.is_pinching(results, frame_width, frame_height):
            center = self.get_pinch_center(results, frame_width, frame_height)
            if center:
                cv2.circle(frame, center, 15, (0, 255, 0), -1)
                cv2.circle(frame, center, 18, (255, 255, 255), 2)
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
