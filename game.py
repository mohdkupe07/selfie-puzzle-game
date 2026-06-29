"""
game.py
-------
The main Pygame-based game interface for the Selfie Puzzle Game.

Features:
  - Renders the 3×3 puzzle board with piece images.
  - Drag-and-drop interaction: click to pick up, release to drop and swap.
  - Timer and move counter HUD.
  - Preview button: temporarily shows the solved image.
  - Restart button: reshuffles the puzzle.
  - Completion detection with animated victory screen.
  - Clean, modern dark UI with rounded elements.
"""

import pygame
import numpy as np
import cv2
import time
import math
from typing import Optional, Tuple

from puzzle import Puzzle, GRID_SIZE, NUM_PIECES

# ─── Layout & Styling Constants ────────────────────────────────────────────────

PIECE_SIZE = 160          # Pixels per puzzle piece (each piece is square)
GRID_PIXEL = PIECE_SIZE * GRID_SIZE  # Total puzzle grid size in pixels

PANEL_WIDTH = 280         # Right-side control panel width
BOARD_PADDING = 30        # Padding around the puzzle board

WINDOW_W = GRID_PIXEL + PANEL_WIDTH + BOARD_PADDING * 3
WINDOW_H = GRID_PIXEL + BOARD_PADDING * 2

BOARD_X = BOARD_PADDING  # Top-left X of puzzle board
BOARD_Y = BOARD_PADDING  # Top-left Y of puzzle board

# Color palette (R, G, B)
C_BG            = (15,  17,  26)   # Deep dark background
C_PANEL         = (22,  26,  40)   # Panel background
C_BOARD_BG      = (28,  32,  48)   # Board background
C_BORDER        = (50,  58,  85)   # Default tile border
C_BORDER_DRAG   = (100, 180, 255)  # Highlighted border when dragging
C_BORDER_SOLVED = (50,  220, 120)  # Green border when piece is in correct position
C_TEXT_PRIMARY  = (240, 242, 255)  # Primary text
C_TEXT_SECONDARY= (140, 148, 175)  # Secondary / label text
C_ACCENT        = (100, 160, 255)  # Accent color (blue)
C_GREEN         = (60,  200, 110)  # Success green
C_BTN_NORMAL    = (35,  42,  65)   # Button background normal
C_BTN_HOVER     = (50,  60,  95)   # Button background on hover
C_BTN_BORDER    = (70,  85, 130)   # Button border
C_DRAG_SHADOW   = (0,   0,   0)    # Shadow color
C_OVERLAY_BG    = (10,  12,  20)   # Victory overlay background

FPS = 60
BORDER_RADIUS = 8
TILE_GAP = 4


def cv2_to_pygame(img: np.ndarray) -> pygame.Surface:
    """
    Convert an OpenCV BGR image (numpy array) to a Pygame Surface.

    Args:
        img: BGR image as numpy array.

    Returns:
        Pygame Surface in RGB format.
    """
    # OpenCV uses BGR; Pygame uses RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return surface


class Button:
    """
    A simple, hover-aware button widget for Pygame.

    Attributes:
        rect (pygame.Rect): Bounding rectangle.
        label (str): Button text.
        font (pygame.Font): Font for rendering.
    """

    def __init__(self, x: int, y: int, w: int, h: int, label: str, font: pygame.font.Font):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.font = font
        self._is_hovered = False

    def draw(self, surface: pygame.Surface):
        """Render the button with hover effect."""
        bg_color = C_BTN_HOVER if self._is_hovered else C_BTN_NORMAL
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=BORDER_RADIUS)
        pygame.draw.rect(surface, C_BTN_BORDER, self.rect, 2, border_radius=BORDER_RADIUS)

        text_surf = self.font.render(self.label, True, C_TEXT_PRIMARY)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos: Tuple[int, int]):
        """Update hover state based on mouse position."""
        self._is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event: pygame.event.Event) -> bool:
        """Return True if this button was clicked by a MOUSEBUTTONDOWN event."""
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class GameUI:
    """
    The main Pygame game interface for the Selfie Puzzle Game.

    Manages rendering, input handling, drag-and-drop, and game state.

    Attributes:
        puzzle (Puzzle): The puzzle model.
        screen (pygame.Surface): The main Pygame window surface.
        clock (pygame.Clock): Pygame clock for FPS control.
        start_time (float): Timestamp when the game started.
        move_count (int): Number of moves the player has made.
        dragging_pos (int | None): Board position of the piece being dragged.
        drag_offset (Tuple): Pixel offset from piece origin to mouse cursor.
        drag_surface (pygame.Surface | None): The surface being dragged.
        is_solved (bool): Whether the puzzle has been completed.
        show_preview (bool): Whether the full preview is currently visible.
        piece_surfaces (List): Pygame surfaces for each piece (by correct_index).
    """

    def __init__(self, puzzle: Puzzle):
        """
        Initialize the GameUI with a configured Puzzle object.

        Args:
            puzzle: A Puzzle with pieces created and shuffled.
        """
        pygame.init()
        pygame.display.set_caption("🧩 Selfie Puzzle Game")

        self.puzzle = puzzle
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        # Game state
        self.start_time = time.time()
        self.move_count = 0
        self.is_solved = False
        self.show_preview = False
        self.completion_time: Optional[float] = None

        # Drag-and-drop state
        self.dragging_pos: Optional[int] = None  # Board position of dragged piece
        self.drag_surface: Optional[pygame.Surface] = None
        self.drag_offset: Tuple[int, int] = (0, 0)
        self.mouse_pos: Tuple[int, int] = (0, 0)

        # Pre-convert all piece images to Pygame surfaces
        self.piece_surfaces = self._precompute_piece_surfaces()

        # Pre-convert preview image
        preview_img = puzzle.get_full_preview_image()
        if preview_img is not None:
            self.preview_surface = pygame.transform.scale(
                cv2_to_pygame(preview_img), (GRID_PIXEL, GRID_PIXEL)
            )
        else:
            self.preview_surface = None

        # Load fonts
        self._init_fonts()

        # Buttons (positioned in right panel)
        panel_x = BOARD_X + GRID_PIXEL + BOARD_PADDING
        panel_center = panel_x + PANEL_WIDTH // 2
        btn_w, btn_h = 200, 46
        btn_x = panel_center - btn_w // 2

        self.btn_preview = Button(btn_x, 200, btn_w, btn_h, "👁  Preview", self.font_btn)
        self.btn_restart = Button(btn_x, 260, btn_w, btn_h, "🔀  Restart", self.font_btn)

        # Victory animation state
        self._victory_alpha = 0
        self._victory_anim_done = False
        self._confetti: list = []
        self._init_confetti()

        # Pulse animation for solved tiles
        self._pulse_phase = 0.0

    def _init_fonts(self):
        """Load fonts — uses system fonts with fallback to pygame default."""
        try:
            self.font_large  = pygame.font.SysFont("segoeui", 36, bold=True)
            self.font_medium = pygame.font.SysFont("segoeui", 22)
            self.font_small  = pygame.font.SysFont("segoeui", 16)
            self.font_btn    = pygame.font.SysFont("segoeui", 18, bold=True)
            self.font_mono   = pygame.font.SysFont("consolas", 20, bold=True)
            self.font_huge   = pygame.font.SysFont("segoeui", 52, bold=True)
        except Exception:
            self.font_large  = pygame.font.Font(None, 40)
            self.font_medium = pygame.font.Font(None, 26)
            self.font_small  = pygame.font.Font(None, 18)
            self.font_btn    = pygame.font.Font(None, 22)
            self.font_mono   = pygame.font.Font(None, 24)
            self.font_huge   = pygame.font.Font(None, 60)

    def _precompute_piece_surfaces(self) -> list:
        """
        Pre-convert all puzzle piece images to Pygame surfaces.

        Returns:
            List of Pygame surfaces indexed by piece.correct_index.
        """
        surfaces = [None] * NUM_PIECES
        for piece in self.puzzle.pieces:
            img = pygame.transform.scale(
                cv2_to_pygame(piece.image), (PIECE_SIZE, PIECE_SIZE)
            )
            surfaces[piece.correct_index] = img
        return surfaces

    def _init_confetti(self, n: int = 80):
        """Initialize confetti particles for the victory animation."""
        self._confetti = []
        for _ in range(n):
            self._confetti.append({
                "x": np.random.randint(0, WINDOW_W),
                "y": np.random.randint(-WINDOW_H, 0),
                "vx": np.random.uniform(-1.5, 1.5),
                "vy": np.random.uniform(2, 6),
                "size": np.random.randint(6, 14),
                "color": (
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                ),
                "rotation": np.random.uniform(0, 360),
                "rot_speed": np.random.uniform(-5, 5),
            })

    def _get_tile_rect(self, position: int) -> pygame.Rect:
        """
        Get the screen pixel rectangle for a given board position.

        Args:
            position: Flat board index (0–8).

        Returns:
            pygame.Rect covering this tile on screen (includes gap).
        """
        row, col = self.puzzle.position_to_grid(position)
        x = BOARD_X + col * (PIECE_SIZE + TILE_GAP)
        y = BOARD_Y + row * (PIECE_SIZE + TILE_GAP)
        return pygame.Rect(x, y, PIECE_SIZE, PIECE_SIZE)

    def _get_position_from_mouse(self, mx: int, my: int) -> Optional[int]:
        """
        Map a mouse coordinate to a board position.

        Args:
            mx, my: Mouse pixel coordinates.

        Returns:
            Board position index (0–8), or None if outside the board.
        """
        # Check bounds
        board_right = BOARD_X + GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP
        board_bottom = BOARD_Y + GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP
        if mx < BOARD_X or mx >= board_right or my < BOARD_Y or my >= board_bottom:
            return None

        col = (mx - BOARD_X) // (PIECE_SIZE + TILE_GAP)
        row = (my - BOARD_Y) // (PIECE_SIZE + TILE_GAP)

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return self.puzzle.grid_to_position(row, col)
        return None

    # ─── Drawing Methods ──────────────────────────────────────────────────────

    def _draw_background(self):
        """Fill the window with the background color and panel."""
        self.screen.fill(C_BG)
        # Right panel
        panel_rect = pygame.Rect(
            BOARD_X + GRID_PIXEL + BOARD_PADDING,
            BOARD_PADDING,
            PANEL_WIDTH,
            WINDOW_H - BOARD_PADDING * 2,
        )
        pygame.draw.rect(self.screen, C_PANEL, panel_rect, border_radius=12)

    def _draw_board(self):
        """Draw the puzzle board background."""
        board_rect = pygame.Rect(
            BOARD_X - 4,
            BOARD_Y - 4,
            GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP + 8,
            GRID_PIXEL + (GRID_SIZE - 1) * TILE_GAP + 8,
        )
        pygame.draw.rect(self.screen, C_BOARD_BG, board_rect, border_radius=12)

    def _draw_tiles(self):
        """Render all puzzle tiles. Skip the tile being dragged."""
        for pos in range(NUM_PIECES):
            if pos == self.dragging_pos:
                continue  # Drawn separately as floating piece

            rect = self._get_tile_rect(pos)
            piece_index = self.puzzle.arrangement[pos]
            piece_surf = self.piece_surfaces[piece_index]

            # Determine border color
            if self.is_solved:
                # Pulse effect on solved tiles
                t = (math.sin(self._pulse_phase + pos * 0.5) + 1) / 2
                r = int(C_BORDER_SOLVED[0] * t + C_BORDER[0] * (1 - t))
                g = int(C_BORDER_SOLVED[1] * t + C_BORDER[1] * (1 - t))
                b = int(C_BORDER_SOLVED[2] * t + C_BORDER[2] * (1 - t))
                border_color = (r, g, b)
            elif piece_index == pos:
                # Piece is in correct position
                border_color = C_BORDER_SOLVED
            else:
                border_color = C_BORDER

            # Draw piece image
            if piece_surf:
                self.screen.blit(piece_surf, rect)

            # Draw border
            pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=4)

    def _draw_dragging_piece(self):
        """Render the piece being dragged, following the mouse cursor."""
        if self.dragging_pos is None or self.drag_surface is None:
            return

        mx, my = self.mouse_pos
        draw_x = mx - self.drag_offset[0]
        draw_y = my - self.drag_offset[1]

        # Drop shadow
        shadow_surf = pygame.Surface((PIECE_SIZE, PIECE_SIZE), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 80))
        self.screen.blit(shadow_surf, (draw_x + 6, draw_y + 6))

        # Piece with slight scale-up (110%)
        scaled = pygame.transform.scale(
            self.drag_surface, (int(PIECE_SIZE * 1.08), int(PIECE_SIZE * 1.08))
        )
        offset_x = (scaled.get_width() - PIECE_SIZE) // 2
        offset_y = (scaled.get_height() - PIECE_SIZE) // 2
        self.screen.blit(scaled, (draw_x - offset_x, draw_y - offset_y))

        # Bright border on dragged piece
        drag_rect = pygame.Rect(draw_x - offset_x, draw_y - offset_y, scaled.get_width(), scaled.get_height())
        pygame.draw.rect(self.screen, C_BORDER_DRAG, drag_rect, 3, border_radius=6)

    def _draw_preview(self):
        """If preview mode is active, overlay the full solved image on the board."""
        if self.show_preview and self.preview_surface:
            # Semi-transparent dark overlay
            overlay = pygame.Surface((GRID_PIXEL, GRID_PIXEL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 40))
            self.screen.blit(self.preview_surface, (BOARD_X, BOARD_Y))
            self.screen.blit(overlay, (BOARD_X, BOARD_Y))

            # "PREVIEW" label
            label = self.font_medium.render("PREVIEW", True, (255, 255, 255))
            lx = BOARD_X + (GRID_PIXEL - label.get_width()) // 2
            self.screen.blit(label, (lx, BOARD_Y + 10))

    def _draw_panel(self):
        """Render the right-side HUD panel with timer, moves, and buttons."""
        panel_x = BOARD_X + GRID_PIXEL + BOARD_PADDING
        cx = panel_x + PANEL_WIDTH // 2

        # Title
        title = self.font_large.render("🧩 Puzzle", True, C_TEXT_PRIMARY)
        self.screen.blit(title, (cx - title.get_width() // 2, BOARD_PADDING + 20))

        # Divider
        pygame.draw.line(
            self.screen,
            C_BORDER,
            (panel_x + 20, BOARD_PADDING + 70),
            (panel_x + PANEL_WIDTH - 20, BOARD_PADDING + 70),
            1,
        )

        # Timer
        elapsed = self.completion_time - self.start_time if self.completion_time else time.time() - self.start_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        time_str = f"{mins:02d}:{secs:02d}"

        self._draw_stat_card(panel_x + 20, BOARD_PADDING + 90, PANEL_WIDTH - 40, "⏱  Time", time_str)
        self._draw_stat_card(panel_x + 20, BOARD_PADDING + 155, PANEL_WIDTH - 40, "🔢  Moves", str(self.move_count))

        # Buttons
        self.btn_preview.draw(self.screen)
        self.btn_restart.draw(self.screen)

        # Status
        if self.is_solved:
            status_color = C_GREEN
            status_text = "✅  Solved!"
        else:
            status_color = C_TEXT_SECONDARY
            pieces_correct = sum(1 for i, p in enumerate(self.puzzle.arrangement) if p == i)
            status_text = f"📍  {pieces_correct}/9 correct"

        status_surf = self.font_medium.render(status_text, True, status_color)
        sy = BOARD_PADDING + 330
        self.screen.blit(status_surf, (cx - status_surf.get_width() // 2, sy))

        # Instructions
        instructions = [
            "Click & drag to move pieces.",
            "Drop on another piece to swap.",
            "Hold Preview to see original.",
        ]
        iy = BOARD_Y + GRID_PIXEL - 130
        for line in instructions:
            surf = self.font_small.render(line, True, C_TEXT_SECONDARY)
            self.screen.blit(surf, (cx - surf.get_width() // 2, iy))
            iy += 22

    def _draw_stat_card(self, x: int, y: int, w: int, label: str, value: str):
        """Draw a small stat card with a label and a large value."""
        card_h = 55
        card_rect = pygame.Rect(x, y, w, card_h)
        pygame.draw.rect(self.screen, C_BOARD_BG, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, C_BORDER, card_rect, 1, border_radius=8)

        label_surf = self.font_small.render(label, True, C_TEXT_SECONDARY)
        value_surf = self.font_mono.render(value, True, C_TEXT_PRIMARY)

        self.screen.blit(label_surf, (x + 12, y + 8))
        self.screen.blit(value_surf, (x + 12, y + 28))

    def _draw_victory_overlay(self):
        """Render the animated victory screen when the puzzle is solved."""
        # Fade in the dark overlay
        if self._victory_alpha < 180:
            self._victory_alpha = min(180, self._victory_alpha + 6)

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((*C_OVERLAY_BG, self._victory_alpha))
        self.screen.blit(overlay, (0, 0))

        # Update and draw confetti
        for p in self._confetti:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["rotation"] += p["rot_speed"]
            if p["y"] > WINDOW_H + 20:
                p["y"] = np.random.randint(-50, -10)
                p["x"] = np.random.randint(0, WINDOW_W)

            # Draw rectangle rotated (approximated with diamond)
            cx, cy = int(p["x"]), int(p["y"])
            s = p["size"]
            points = [
                (cx, cy - s),
                (cx + s, cy),
                (cx, cy + s),
                (cx - s, cy),
            ]
            pygame.draw.polygon(self.screen, p["color"], points)

        # Victory text
        cx = WINDOW_W // 2
        text_y = WINDOW_H // 2 - 80

        title = self.font_huge.render("🎉 Puzzle Solved!", True, C_GREEN)
        self.screen.blit(title, (cx - title.get_width() // 2, text_y))

        elapsed = self.completion_time - self.start_time if self.completion_time else 0
        mins, secs = divmod(int(elapsed), 60)
        stats = [
            f"Time:  {mins:02d}:{secs:02d}",
            f"Moves: {self.move_count}",
        ]
        ty = text_y + 80
        for line in stats:
            s = self.font_large.render(line, True, C_TEXT_PRIMARY)
            self.screen.blit(s, (cx - s.get_width() // 2, ty))
            ty += 48

        hint = self.font_medium.render("Press R to play again  |  Q to quit", True, C_TEXT_SECONDARY)
        self.screen.blit(hint, (cx - hint.get_width() // 2, ty + 30))

    # ─── Event Handling ───────────────────────────────────────────────────────

    def handle_events(self) -> bool:
        """
        Process all Pygame events for one frame.

        Returns:
            False if the game should exit, True otherwise.
        """
        mouse_pos = pygame.mouse.get_pos()
        self.mouse_pos = mouse_pos

        self.btn_preview.update(mouse_pos)
        self.btn_restart.update(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self._restart()
                if event.key == pygame.K_p:
                    self.show_preview = not self.show_preview

            # Preview button (hold = show, release = hide)
            if self.btn_preview.is_clicked(event):
                self.show_preview = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.show_preview = False

            # Drag start
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not self.is_solved:
                    self._handle_mouse_down(event)

            # Drop
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._handle_mouse_up(event)

            # Restart button
            if self.btn_restart.is_clicked(event):
                self._restart()

        return True

    def _handle_mouse_down(self, event: pygame.event.Event):
        """Start dragging a puzzle piece."""
        pos = self._get_position_from_mouse(*event.pos)
        if pos is not None:
            tile_rect = self._get_tile_rect(pos)
            piece_index = self.puzzle.arrangement[pos]
            self.drag_surface = self.piece_surfaces[piece_index]
            self.dragging_pos = pos
            # Offset: where in the tile did the user click?
            self.drag_offset = (
                event.pos[0] - tile_rect.x,
                event.pos[1] - tile_rect.y,
            )

    def _handle_mouse_up(self, event: pygame.event.Event):
        """Drop the dragged piece, swapping with the target."""
        if self.dragging_pos is None:
            return

        target_pos = self._get_position_from_mouse(*event.pos)
        if target_pos is not None and target_pos != self.dragging_pos:
            self.puzzle.swap_pieces(self.dragging_pos, target_pos)
            self.move_count += 1

            # Check for completion
            if self.puzzle.is_solved() and not self.is_solved:
                self.is_solved = True
                self.completion_time = time.time()
                self._init_confetti()
                self._victory_alpha = 0

        # Reset drag state
        self.dragging_pos = None
        self.drag_surface = None

    def _restart(self):
        """Reset the puzzle and game state."""
        self.puzzle.shuffle()
        self.move_count = 0
        self.start_time = time.time()
        self.is_solved = False
        self.completion_time = None
        self._victory_alpha = 0
        self.dragging_pos = None
        self.drag_surface = None
        self.show_preview = False
        self._init_confetti()

    # ─── Main Game Loop ───────────────────────────────────────────────────────

    def run(self):
        """
        Start and run the game loop.

        Blocks until the user closes the game.
        """
        running = True
        while running:
            self.clock.tick(FPS)

            # Update animations
            self._pulse_phase += 0.05

            running = self.handle_events()

            # ── Render ────────────────────────────────────────────────────
            self._draw_background()
            self._draw_board()
            self._draw_tiles()
            self._draw_panel()
            self._draw_dragging_piece()

            if self.show_preview:
                self._draw_preview()

            if self.is_solved:
                self._draw_victory_overlay()

            pygame.display.flip()

        pygame.quit()
