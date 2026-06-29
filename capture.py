"""
capture.py
----------
Manages frame selection using hand gestures and captures the selected image region.

Interaction flow:
  1. No selection: first pinch sets the ANCHOR point (top-left corner).
  2. Drag mode: while pinching, the current pinch position defines the second corner,
     dynamically drawing the selection rectangle.
  3. Release pinch: rectangle is locked in place.
  4. Second pinch INSIDE the locked rectangle: confirms the capture.

The module draws the selection UI overlay on the live camera frame.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# Selection state machine states
STATE_IDLE = "idle"          # Waiting for first pinch
STATE_DRAWING = "drawing"    # User is drawing the rectangle (pinch held)
STATE_LOCKED = "locked"      # Rectangle drawn and waiting for confirm pinch
STATE_CAPTURED = "captured"  # Image has been captured


class FrameCapture:
    """
    Handles the interactive rectangular region selection and image capture.

    The user draws a selection box using pinch gestures:
      - First pinch-and-hold: sets the start corner and drags to resize.
      - Release: locks the rectangle.
      - Second pinch inside the box: confirms and captures.

    Attributes:
        state (str): Current state of the selection state machine.
        anchor (tuple): The fixed corner of the selection rectangle (x, y).
        current_pos (tuple): The moving corner of the selection rectangle (x, y).
        captured_image (np.ndarray): The cropped image after capture.
        rect (tuple): Normalized (x1, y1, x2, y2) of the locked rectangle.
    """

    def __init__(self):
        """Initialize with idle state and no selection."""
        self.state = STATE_IDLE
        self.anchor: Optional[Tuple[int, int]] = None
        self.current_pos: Optional[Tuple[int, int]] = None
        self.captured_image: Optional[np.ndarray] = None
        self.rect: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

        # Minimum rectangle size to prevent accidental tiny captures
        self.min_rect_size = 80

        # Flash animation for capture confirmation
        self._flash_frames = 0
        self._flash_duration = 8

    def update(
        self,
        frame: np.ndarray,
        is_pinching: bool,
        pinch_start: bool,
        pinch_center: Optional[Tuple[int, int]],
    ) -> bool:
        """
        Update the selection state based on the current gesture input.

        Args:
            frame: The current BGR camera frame.
            is_pinching: Whether a pinch gesture is currently active.
            pinch_start: Whether a new pinch just started this frame.
            pinch_center: The (x, y) midpoint of the pinch, or None.

        Returns:
            True if the image has just been captured, False otherwise.
        """
        h, w = frame.shape[:2]
        just_captured = False

        if self.state == STATE_IDLE:
            # --- IDLE: Wait for a pinch to start drawing ---
            if is_pinching and pinch_center:
                self.anchor = pinch_center
                self.current_pos = pinch_center
                self.state = STATE_DRAWING

        elif self.state == STATE_DRAWING:
            # --- DRAWING: Update the second corner while pinch is held ---
            if is_pinching and pinch_center:
                self.current_pos = pinch_center
            else:
                # Pinch released — lock the rectangle if it's large enough
                if self.anchor and self.current_pos:
                    x1 = min(self.anchor[0], self.current_pos[0])
                    y1 = min(self.anchor[1], self.current_pos[1])
                    x2 = max(self.anchor[0], self.current_pos[0])
                    y2 = max(self.anchor[1], self.current_pos[1])

                    if (x2 - x1) >= self.min_rect_size and (y2 - y1) >= self.min_rect_size:
                        self.rect = (x1, y1, x2, y2)
                        self.state = STATE_LOCKED
                    else:
                        # Too small — go back to idle
                        self._reset_selection()

        elif self.state == STATE_LOCKED:
            # --- LOCKED: Wait for a second pinch inside the rectangle ---
            if pinch_start and pinch_center and self.rect:
                x1, y1, x2, y2 = self.rect
                px, py = pinch_center
                if x1 <= px <= x2 and y1 <= py <= y2:
                    # Capture the region
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        self.captured_image = cropped.copy()
                        self.state = STATE_CAPTURED
                        self._flash_frames = self._flash_duration
                        just_captured = True
                else:
                    # Pinch detected OUTSIDE the locked box — reset to idle
                    self._reset_selection()

        return just_captured

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the selection UI overlay on the frame.

        Draws the rectangle, instructions, and capture flash effect.

        Args:
            frame: BGR frame to draw on.

        Returns:
            Frame with overlay drawn.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Draw the instructions at the top
        self._draw_instructions(frame, h, w)

        if self.state == STATE_DRAWING and self.anchor and self.current_pos:
            # Draw the dynamic selection rectangle
            x1 = min(self.anchor[0], self.current_pos[0])
            y1 = min(self.anchor[1], self.current_pos[1])
            x2 = max(self.anchor[0], self.current_pos[0])
            y2 = max(self.anchor[1], self.current_pos[1])

            # Semi-transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 150, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Solid border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 150, 255), 3)
            # Corner handles
            self._draw_corner_handles(frame, x1, y1, x2, y2, (50, 150, 255))

        elif self.state == STATE_LOCKED and self.rect:
            x1, y1, x2, y2 = self.rect

            # Semi-transparent fill (darker to indicate locked state)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 100), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Dashed / solid border in green for "confirmed"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 3)
            self._draw_corner_handles(frame, x1, y1, x2, y2, (0, 220, 100))

            # Instruction inside the box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self._put_centered_text(frame, "Pinch inside to capture!", cx, cy, (255, 255, 255), 0.7)

        elif self.state == STATE_CAPTURED:
            # Flash white to signal capture
            if self._flash_frames > 0:
                alpha = self._flash_frames / self._flash_duration
                white = np.ones_like(frame) * 255
                cv2.addWeighted(white, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)
                self._flash_frames -= 1

        return frame

    def _draw_instructions(self, frame: np.ndarray, h: int, w: int):
        """Draw instructional text based on the current state."""
        instructions = {
            STATE_IDLE:     "Pinch & drag to draw selection box",
            STATE_DRAWING:  "Release pinch to lock the box",
            STATE_LOCKED:   "Pinch inside box to capture | Pinch outside to reset",
            STATE_CAPTURED: "Image captured!",
        }

        text = instructions.get(self.state, "")
        colors = {
            STATE_IDLE:     (200, 200, 200),
            STATE_DRAWING:  (50, 150, 255),
            STATE_LOCKED:   (0, 220, 100),
            STATE_CAPTURED: (50, 255, 150),
        }
        color = colors.get(self.state, (200, 200, 200))

        # Draw background pill
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = (w - tw) // 2
        ty = 30
        cv2.rectangle(frame, (tx - 10, ty - th - 6), (tx + tw + 10, ty + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_corner_handles(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple
    ):
        """Draw L-shaped corner handles on the rectangle."""
        length = 20
        thickness = 4
        corners = [
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ]
        for (cx, cy), (dx, dy) in corners:
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness)

    def _put_centered_text(
        self, frame: np.ndarray, text: str, cx: int, cy: int, color: tuple, scale: float = 0.7
    ):
        """Put text centered at (cx, cy)."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        tx = cx - tw // 2
        ty = cy + th // 2
        # Shadow
        cv2.putText(frame, text, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    def _reset_selection(self):
        """Reset the selection state back to idle."""
        self.state = STATE_IDLE
        self.anchor = None
        self.current_pos = None
        self.rect = None

    def reset(self):
        """Public method to fully reset the capture state."""
        self._reset_selection()
        self.captured_image = None

    def is_done(self) -> bool:
        """Return True if an image has been successfully captured."""
        return self.state == STATE_CAPTURED and self.captured_image is not None

    def get_captured_image(self) -> Optional[np.ndarray]:
        """Return the captured image (BGR) or None."""
        return self.captured_image
