"""Chess pieces detection source package."""

from src.board_segmentor import (
    BoardSegmentor,
    draw_extracted_squares,
    extract_chessboard_perspective,
)

__all__ = [
    "BoardSegmentor",
    "extract_chessboard_perspective",
    "draw_extracted_squares",
]
