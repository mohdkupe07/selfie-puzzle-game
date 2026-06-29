"""
puzzle.py
---------
Handles all puzzle logic:
  - Splitting the captured image into a 3×3 grid of pieces.
  - Storing the correct (solved) arrangement.
  - Shuffling the pieces to create a solvable puzzle.
  - Checking for puzzle completion.

Each piece is stored as a NumPy array (BGR image).
The puzzle state is represented as a flat list of 9 indices,
where index[i] is the piece at position i.
"""

import numpy as np
import random
from typing import List, Optional, Tuple


# Grid configuration
GRID_SIZE = 3           # 3×3 puzzle
NUM_PIECES = GRID_SIZE * GRID_SIZE  # 9 pieces


class PuzzlePiece:
    """
    Represents a single puzzle piece.

    Attributes:
        image (np.ndarray): The BGR image data for this piece.
        correct_index (int): The solved position index (0–8, row-major).
    """

    def __init__(self, image: np.ndarray, correct_index: int):
        """
        Initialize a puzzle piece.

        Args:
            image: The cropped BGR image for this piece.
            correct_index: The index this piece belongs to when solved.
        """
        self.image = image
        self.correct_index = correct_index


class Puzzle:
    """
    Manages the puzzle state: pieces, their arrangement, and game logic.

    The puzzle uses a flat list `arrangement` of length 9.
    `arrangement[i]` holds the `correct_index` of the piece currently
    placed at board position i.

    Attributes:
        pieces (List[PuzzlePiece]): All 9 puzzle pieces.
        arrangement (List[int]): Current board — arrangement[position] = piece_index.
        piece_size (Tuple[int, int]): (width, height) of each piece in pixels.
        source_image (np.ndarray): The original captured image (resized to fit grid).
    """

    def __init__(self):
        """Initialize an empty puzzle."""
        self.pieces: List[PuzzlePiece] = []
        self.arrangement: List[int] = list(range(NUM_PIECES))  # Solved by default
        self.piece_size: Tuple[int, int] = (0, 0)
        self.source_image: Optional[np.ndarray] = None

    def create_from_image(self, image: np.ndarray, piece_size: int = 160):
        """
        Split the given image into a 3×3 grid of puzzle pieces.

        The image is resized to exactly GRID_SIZE×piece_size pixels to ensure
        all pieces are equal and no remainder pixels are lost.

        Args:
            image: The captured BGR image to split.
            piece_size: The size (width and height) of each piece in pixels.
        """
        import cv2

        # Target dimensions for the puzzle (must be divisible by GRID_SIZE)
        target_w = piece_size * GRID_SIZE
        target_h = piece_size * GRID_SIZE

        # Resize source image to the exact grid size
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        self.source_image = resized
        self.piece_size = (piece_size, piece_size)

        self.pieces = []

        # Slice the image into GRID_SIZE × GRID_SIZE pieces
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                y1 = row * piece_size
                y2 = (row + 1) * piece_size
                x1 = col * piece_size
                x2 = (col + 1) * piece_size

                piece_img = resized[y1:y2, x1:x2].copy()
                correct_index = row * GRID_SIZE + col
                self.pieces.append(PuzzlePiece(piece_img, correct_index))

        # Start with the solved arrangement
        self.arrangement = list(range(NUM_PIECES))

    def shuffle(self, max_attempts: int = 1000):
        """
        Shuffle the puzzle arrangement.

        Performs a random shuffle and repeats if the result is already solved.
        This guarantees the shuffled state is never identical to the solved state.

        Args:
            max_attempts: Maximum shuffle attempts to avoid infinite loop.
        """
        solved = list(range(NUM_PIECES))

        for _ in range(max_attempts):
            random.shuffle(self.arrangement)
            if self.arrangement != solved:
                return

        # Fallback: swap first two pieces if all shuffles happened to be solved
        if len(self.arrangement) >= 2:
            self.arrangement[0], self.arrangement[1] = self.arrangement[1], self.arrangement[0]

    def swap_pieces(self, pos_a: int, pos_b: int):
        """
        Swap the puzzle pieces at two board positions.

        Args:
            pos_a: Board position index of the first piece.
            pos_b: Board position index of the second piece.
        """
        if 0 <= pos_a < NUM_PIECES and 0 <= pos_b < NUM_PIECES:
            self.arrangement[pos_a], self.arrangement[pos_b] = (
                self.arrangement[pos_b],
                self.arrangement[pos_a],
            )

    def is_solved(self) -> bool:
        """
        Check if the puzzle is in the solved arrangement.

        Returns:
            True if all pieces are in their correct positions.
        """
        return self.arrangement == list(range(NUM_PIECES))

    def get_piece_at(self, position: int) -> Optional[PuzzlePiece]:
        """
        Get the puzzle piece currently placed at a board position.

        Args:
            position: Board position index (0–8).

        Returns:
            The PuzzlePiece at that position, or None if index is invalid.
        """
        if 0 <= position < NUM_PIECES:
            piece_index = self.arrangement[position]
            return self.pieces[piece_index]
        return None

    def get_piece_image_at(self, position: int) -> Optional[np.ndarray]:
        """
        Retrieve the image of the piece at a given board position.

        Args:
            position: Board position index (0–8).

        Returns:
            BGR image of the piece, or None.
        """
        piece = self.get_piece_at(position)
        return piece.image if piece else None

    def get_full_preview_image(self) -> Optional[np.ndarray]:
        """
        Return the full assembled (solved) puzzle image.

        Returns:
            The original resized source image, or None if not yet created.
        """
        return self.source_image

    def position_to_grid(self, position: int) -> Tuple[int, int]:
        """
        Convert a flat board position to (row, col) grid coordinates.

        Args:
            position: Flat index (0–8).

        Returns:
            (row, col) tuple.
        """
        return divmod(position, GRID_SIZE)

    def grid_to_position(self, row: int, col: int) -> int:
        """
        Convert (row, col) grid coordinates to a flat board position.

        Args:
            row: Grid row (0–2).
            col: Grid column (0–2).

        Returns:
            Flat board position index.
        """
        return row * GRID_SIZE + col
