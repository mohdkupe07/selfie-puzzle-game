"""
main.py
-------
Application entry point for the Selfie Puzzle Game.

Workflow:
  1. Open webcam via the Camera module.
  2. Show a live feed with gesture overlay via OpenCV.
  3. Use GestureDetector (MediaPipe) to detect pinch gestures.
  4. Use FrameCapture to let the user select and capture a region.
  5. Once captured, close OpenCV, build the Puzzle, and launch the Pygame GameUI.

Press 'Q' or 'ESC' in either window to quit.
"""

import sys
import cv2

from camera import Camera
from gesture import GestureDetector
from capture import FrameCapture, STATE_CAPTURED
from puzzle import Puzzle
from game import GameUI


WEBCAM_WINDOW_NAME = "Selfie Puzzle Game — Select Region"


def run_capture_phase() -> "np.ndarray | None":
    """
    Run the webcam phase: show live feed and let the user select a region.

    Returns:
        The captured BGR image as a NumPy array, or None if the user quit early.
    """
    # ── Initialize camera ──────────────────────────────────────────────────
    camera = Camera(camera_index=0, width=1280, height=720, fps=30)
    if not camera.open():
        print("[ERROR] Failed to open webcam. Check camera connection and permissions.")
        return None

    width, height = camera.get_resolution()

    # ── Initialize gesture detector ────────────────────────────────────────
    gesture = GestureDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        pinch_threshold=0.06,
        pinch_cooldown=15,
    )

    # ── Initialize frame capture state machine ─────────────────────────────
    frame_capture = FrameCapture()

    # ── Create the OpenCV display window ──────────────────────────────────
    cv2.namedWindow(WEBCAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WEBCAM_WINDOW_NAME, width, height)

    print("[INFO] Webcam phase started.")
    print("[INFO] Instructions:")
    print("         1. Pinch and drag to draw a selection rectangle.")
    print("         2. Release the pinch to lock the rectangle.")
    print("         3. Pinch inside the rectangle to capture the image.")
    print("         4. Press 'Q' or 'ESC' to quit.")

    captured_image = None

    # ── Main webcam loop ──────────────────────────────────────────────────
    while True:
        # Read frame from webcam
        success, frame = camera.read_frame()
        if not success or frame is None:
            print("[WARNING] Failed to read frame. Retrying...")
            continue

        # Process hand landmarks with MediaPipe
        results = gesture.process(frame)

        # Extract gesture info
        is_pinching = gesture.is_pinching(results, width, height)
        pinch_start = gesture.detected_pinch_start(results, width, height)
        pinch_center = gesture.get_pinch_center(results, width, height)

        # Update selection state machine
        just_captured = frame_capture.update(
            frame, is_pinching, pinch_start, pinch_center
        )

        # Draw hand landmarks
        gesture.draw_landmarks(frame, results)
        gesture.draw_pinch_indicator(frame, results, width, height)

        # Draw selection overlay
        frame_capture.draw_overlay(frame)

        # Show the frame
        cv2.imshow(WEBCAM_WINDOW_NAME, frame)

        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or ESC
            print("[INFO] Quit requested by user.")
            break

        # Check if capture was completed
        if frame_capture.is_done():
            captured_image = frame_capture.get_captured_image()
            print("[INFO] Image captured! Launching puzzle game...")
            break

    # ── Cleanup ────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    gesture.close()
    camera.release()

    return captured_image


def run_puzzle_phase(captured_image):
    """
    Create the puzzle from the captured image and run the Pygame game.

    Args:
        captured_image: BGR NumPy array of the selected image.
    """
    # ── Build the puzzle ──────────────────────────────────────────────────
    puzzle = Puzzle()
    puzzle.create_from_image(captured_image, piece_size=160)
    puzzle.shuffle()

    print(f"[INFO] Puzzle created: {puzzle.piece_size[0]}x{puzzle.piece_size[1]} px per piece.")
    print("[INFO] Launching Pygame game window...")

    # ── Run the game ──────────────────────────────────────────────────────
    game = GameUI(puzzle)
    game.run()

    print("[INFO] Game closed.")


def main():
    """Application entry point."""
    print("=" * 55)
    print("   🧩  Selfie Puzzle Game  🧩")
    print("=" * 55)

    # Phase 1: Capture a selfie region via webcam gesture
    captured_image = run_capture_phase()

    if captured_image is None:
        print("[INFO] No image captured. Exiting.")
        sys.exit(0)

    # Phase 2: Play the puzzle game with the captured image
    run_puzzle_phase(captured_image)

    print("[INFO] Thanks for playing!")


if __name__ == "__main__":
    main()
