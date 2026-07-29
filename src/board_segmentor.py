"""Board Segmentor module using YOLO / Roboflow Segmentation and Edge-Separated RANSAC line fitting.

This module provides multi-backend segmentation (Local YOLO or Roboflow Cloud Inference),
robust edge-separated RANSAC line fitting to detect the 4 corners of a chessboard,
and perspective transformation with 64-square grid extraction.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helper functions (RANSAC line fitting internals)
# ---------------------------------------------------------------------------

def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Calculate perpendicular distance from point p to line segment ab."""
    v = b - a
    w = p - a
    c1 = np.dot(w, v)
    if c1 <= 0:
        return float(np.linalg.norm(p - a))
    c2 = np.dot(v, v)
    if c2 <= c1:
        return float(np.linalg.norm(p - b))
    proj = a + (c1 / c2) * v
    return float(np.linalg.norm(p - proj))


def _best_fit_line_ransac(
    points: np.ndarray, threshold: float = 5.0, max_iters: int = 1000
) -> Tuple[Optional[Tuple[float, float, float, float]], np.ndarray]:
    """Fit a 2D vector line (vx, vy, x0, y0) using RANSAC on a point cloud.

    Returns:
        Fitted line tuple (vx, vy, x0, y0) and array of inlier points.
    """
    if len(points) < 2:
        return None, np.array([])

    best_inliers = np.array([])
    best_line = None
    num_points = len(points)

    for _ in range(max_iters):
        idx = np.random.choice(num_points, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        if np.all(p1 == p2):
            continue

        fit_res = cv2.fitLine(
            np.array([p1, p2], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )
        vx, vy, x0, y0 = (
            float(fit_res[0][0]),
            float(fit_res[1][0]),
            float(fit_res[2][0]),
            float(fit_res[3][0]),
        )

        # Perpendicular distance to candidate line
        dists = np.abs((points - np.array([x0, y0])) @ np.array([-vy, vx]))
        inliers = points[dists < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (vx, vy, x0, y0)

    # Refine using all best inliers
    if best_line is not None and len(best_inliers) >= 2:
        fit_res = cv2.fitLine(
            best_inliers.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )
        best_line = (
            float(fit_res[0][0]),
            float(fit_res[1][0]),
            float(fit_res[2][0]),
            float(fit_res[3][0]),
        )

    return best_line, best_inliers


def _extract_initial_4_corners(polygon_points: np.ndarray) -> np.ndarray:
    """Find initial 4 corner coordinates bounding the polygon in clockwise order [TL, TR, BR, BL]."""
    pts = polygon_points.astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)

    # Attempt polygon approximation to find convex quadrilateral
    for eps_factor in np.linspace(0.08, 0.005, 20):
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2)
            return _sort_corners_by_polar_angle(corners)

    # Fallback: pick 4 extreme hull vertices using centroid + polar angle ordering.
    # This avoids the sum/diff method which fails on perspective-angled boards.
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    centroid = hull_pts.mean(axis=0)
    angles = np.arctan2(hull_pts[:, 1] - centroid[1], hull_pts[:, 0] - centroid[0])
    # Sample 4 points at roughly 0, 90, 180, 270 degree intervals
    target_angles = [-np.pi, -np.pi / 2, 0.0, np.pi / 2]
    extreme_pts = []
    for target in target_angles:
        diffs_angle = np.abs(angles - target)
        diffs_angle = np.minimum(diffs_angle, 2 * np.pi - diffs_angle)
        extreme_pts.append(hull_pts[np.argmin(diffs_angle)])
    return _sort_corners_by_polar_angle(np.array(extreme_pts, dtype=np.float32))


def _sort_corners_by_polar_angle(corners: np.ndarray) -> np.ndarray:
    """Sort 4 corners clockwise [TL, TR, BR, BL] using centroid and polar angle.

    Robust for any camera angle / perspective, unlike sum/diff which breaks on rotated boards.
    """
    pts = corners.astype(np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    # Rotate so that the "top-left" quadrant starts at -135 degrees
    angles = (angles - (-3 * np.pi / 4)) % (2 * np.pi)
    order = np.argsort(angles)
    sorted_pts = pts[order]  # clockwise from top-left
    return sorted_pts


def _extract_4_edge_lines_ransac(
    polygon_points: np.ndarray, threshold: float = 10.0, max_iters: int = 1000
) -> Tuple[List[Tuple[float, float, float, float]], np.ndarray]:
    """Partition polygon contour into 4 edge clusters and fit 4 RANSAC lines.

    Uses contour perimeter splitting instead of geometric distance assignment
    to correctly partition points on perspective-angled boards.
    """
    init_corners = _extract_initial_4_corners(polygon_points)

    # 4 perimeter segments for fallback: TL->TR, TR->BR, BR->BL, BL->TL
    segments = [
        (init_corners[0], init_corners[1]),
        (init_corners[1], init_corners[2]),
        (init_corners[2], init_corners[3]),
        (init_corners[3], init_corners[0]),
    ]

    # --- Perimeter split clustering ---
    # Find the index in the contour array closest to each of the 4 corners,
    # then slice the contour between adjacent corner indices to get 4 clean edge clusters.
    pts = polygon_points.astype(np.float32)
    n = len(pts)
    corner_indices = []
    for corner in init_corners:
        dists = np.linalg.norm(pts - corner, axis=1)
        corner_indices.append(int(np.argmin(dists)))

    # Sort corner indices so we can split contour in order
    corner_indices_sorted = sorted(corner_indices)
    edge_clusters: List[List[np.ndarray]] = []
    num_corners = len(corner_indices_sorted)
    for i in range(num_corners):
        start = corner_indices_sorted[i]
        end = corner_indices_sorted[(i + 1) % num_corners]
        if end > start:
            cluster = pts[start:end + 1]
        else:
            # Wrap-around case
            cluster = np.concatenate([pts[start:], pts[:end + 1]], axis=0)
        edge_clusters.append(cluster.tolist())

    # Re-order edge clusters to match [Top, Right, Bottom, Left] based on corner order
    # The corner at index 0 (TL) -> index 1 (TR) is Top edge, etc.
    reordered_clusters: List[List[np.ndarray]] = [[], [], [], []]
    for i in range(4):
        ci_start = corner_indices_sorted.index(corner_indices[i])
        reordered_clusters[i] = edge_clusters[ci_start]

    lines = []
    for k in range(4):
        cluster_pts = np.array(reordered_clusters[k], dtype=np.float32)
        line = None
        if len(cluster_pts) >= 2:
            line, _ = _best_fit_line_ransac(
                cluster_pts, threshold=threshold, max_iters=max_iters
            )

        if line is None:
            # Fallback: line through segment endpoints
            p1, p2 = segments[k][0], segments[k][1]
            fit_res = cv2.fitLine(
                np.array([p1, p2], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01
            )
            line = (
                float(fit_res[0][0]),
                float(fit_res[1][0]),
                float(fit_res[2][0]),
                float(fit_res[3][0]),
            )

        lines.append(line)

    return lines, init_corners


def _line_intersection(
    line1: Tuple[float, float, float, float],
    line2: Tuple[float, float, float, float],
    img_shape: Optional[Tuple[int, ...]] = None,
    margin: float = 100.0,
) -> Optional[np.ndarray]:
    """Find intersection point of two 2D vector lines (vx, vy, x0, y0).

    Returns:
        Intersection coordinate [x, y] or None if parallel / out of bounds.
    """
    vx1, vy1, x01, y01 = line1
    vx2, vy2, x02, y02 = line2

    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x02 - x01, y02 - y01], dtype=np.float64)

    if np.linalg.matrix_rank(A) < 2:
        return None  # Parallel lines

    t = np.linalg.solve(A, b)
    intersection = np.array([x01, y01], dtype=np.float32) + t[0] * np.array(
        [vx1, vy1], dtype=np.float32
    )

    if img_shape is not None:
        h, w = img_shape[:2]
        x, y = intersection
        if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
            return None

    return intersection


# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------

def order_corners(pts: np.ndarray, ordering: str = "standard") -> np.ndarray:
    """Order 4 corner points into standard sequence.

    Args:
        pts: 4 corner points array of shape (4, 2).
        ordering:
            - "standard": [top_left, top_right, bottom_left, bottom_right]
            - "clockwise": [top_left, top_right, bottom_right, bottom_left]

    Returns:
        Ordered float32 corners of shape (4, 2).
    """
    pts_arr = np.array(pts, dtype=np.float32).reshape(4, 2)
    sums = pts_arr.sum(axis=1)
    diffs = np.diff(pts_arr, axis=1)

    top_left = pts_arr[np.argmin(sums)]
    bottom_right = pts_arr[np.argmax(sums)]
    top_right = pts_arr[np.argmin(diffs)]
    bottom_left = pts_arr[np.argmax(diffs)]

    if ordering == "clockwise":
        return np.float32([top_left, top_right, bottom_right, bottom_left])
    return np.float32([top_left, top_right, bottom_left, bottom_right])


def extract_chessboard_perspective(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    target_size: int = 1200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Perform perspective transformation given 4 board corners and extract 64 square polygons.

    Args:
        image_bgr: Input BGR image.
        corners: 4 ordered corner points [top_left, top_right, bottom_left, bottom_right].
        target_size: Resolution of top-down warped board.

    Returns:
        Tuple of (warped_board, M, M_inv, squares_data_original, cell_dict).
    """
    corners_ordered = order_corners(corners, ordering="standard")

    dst_pts = np.float32([
        [0, 0],
        [target_size, 0],
        [0, target_size],
        [target_size, target_size],
    ])

    M = cv2.getPerspectiveTransform(corners_ordered, dst_pts)
    _, M_inv = cv2.invert(M)
    warped_board = cv2.warpPerspective(image_bgr, M, (target_size, target_size))

    # Build 8x8 grid in warped space
    sq_w = target_size // 8
    sq_h = target_size // 8

    squares_data_warped = []
    for i in range(7, -1, -1):  # Bottom row (rank 1) up to top row (rank 8)
        for j in range(8):      # Left (file a) to right (file h)
            tl = (j * sq_w, i * sq_h)
            tr = ((j + 1) * sq_w, i * sq_h)
            bl = (j * sq_w, (i + 1) * sq_h)
            br = ((j + 1) * sq_w, (i + 1) * sq_h)
            center = ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
            squares_data_warped.append([center, br, tr, tl, bl])

    # Transform grid points back to original camera perspective
    warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)
    orig_np = cv2.perspectiveTransform(warped_np, M_inv)
    squares_data_original = orig_np.reshape(-1, 5, 2)

    # Build cell dictionary: cell_id (1..64) -> polygon + center
    cell_dict = {}
    for idx, sq in enumerate(squares_data_original, 1):
        center, br, tr, tl, bl = sq[0], sq[1], sq[2], sq[3], sq[4]
        cell_dict[idx] = {
            "center": (int(center[0]), int(center[1])),
            "polygon": np.array([tl, tr, br, bl], dtype=np.int32),
            "points": [br, tr, tl, bl],
        }

    return warped_board, M, M_inv, squares_data_original, cell_dict


def draw_extracted_squares(image_bgr: np.ndarray, cell_dict: dict) -> np.ndarray:
    """Draw extracted 64 chessboard square polygons on original image.

    Args:
        image_bgr: Original BGR image.
        cell_dict: Cell dictionary from extract_chessboard_perspective.

    Returns:
        Image with 64-square grid lines overlaid (BGR).
    """
    annotated = image_bgr.copy()
    for info in cell_dict.values():
        cv2.polylines(
            annotated,
            [info["polygon"]],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------------
# BoardSegmentor class
# ---------------------------------------------------------------------------

class BoardSegmentor:
    """Chessboard segmentation and 4-corner detection using YOLO or Roboflow + RANSAC."""

    def __init__(
        self,
        source_type: str = "local",
        model_path: Optional[Union[str, Path]] = None,
        model_id: Optional[str] = "chessboard-segmentation/1",
        api_url: str = "https://serverless.roboflow.com",
        api_key: Optional[str] = None,
        device: Union[str, int] = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        ransac_threshold: float = 10.0,
        ransac_max_iters: int = 1000,
    ):
        """Initialize BoardSegmentor with local YOLO or Roboflow Cloud backend.

        Args:
            source_type: Backend type - 'local' (YOLO) or 'roboflow' / 'server' / 'remote'.
            model_path: Path to local YOLO segmentation .pt model.
            model_id: Roboflow model ID (e.g. 'chessboard-segmentation/1').
            api_url: Roboflow server URL.
            api_key: Roboflow API key.
            device: Inference device for local model ("cpu", "cuda", 0, etc.).
            conf_threshold: Confidence threshold.
            iou_threshold: NMS IoU threshold.
            imgsz: Target inference image size.
            ransac_threshold: RANSAC distance threshold in pixels.
            ransac_max_iters: Maximum RANSAC iterations.
        """
        self.source_type = source_type.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.model_id = model_id
        self.ransac_threshold = ransac_threshold
        self.ransac_max_iters = ransac_max_iters

        self._local_model = None
        self._roboflow_client = None

        # Load backend on init
        if self.source_type == "local":
            if model_path:
                if YOLO is None:
                    raise ImportError(
                        "Ultralytics is not installed. Install via 'pip install ultralytics'."
                    )
                logger.info("Loading local YOLO segmentation model from %s...", model_path)
                self._local_model = YOLO(str(model_path))
        elif self.source_type in ("roboflow", "server", "remote"):
            if InferenceHTTPClient is None:
                raise ImportError(
                    "inference_sdk is not installed. Install via 'pip install inference-sdk'."
                )
            logger.info("Initializing Roboflow InferenceHTTPClient (URL: %s)...", api_url)
            self._roboflow_client = InferenceHTTPClient(
                api_url=api_url, api_key=api_key
            )
        else:
            raise ValueError(
                f"Unsupported source_type: '{source_type}'. Choose 'local' or 'roboflow'."
            )

    # ---- Private helpers ----

    @staticmethod
    def _parse_roboflow_polygons(res_json: dict) -> List[np.ndarray]:
        """Extract segmentation polygons from Roboflow API response JSON."""
        polygons = []
        preds = res_json.get("predictions", [])
        if isinstance(preds, dict):
            preds = [preds]

        for p in preds:
            points = p.get("points")
            if not points:
                continue
            poly_pts = []
            for pt in points:
                if isinstance(pt, dict):
                    poly_pts.append([pt.get("x", 0.0), pt.get("y", 0.0)])
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    poly_pts.append([pt[0], pt[1]])

            if len(poly_pts) >= 3:
                polygons.append(np.array(poly_pts, dtype=np.float32))

        return polygons

    def _predict_corners_from_polygon(
        self,
        polygon_points: np.ndarray,
        img_shape: Optional[Tuple[int, ...]] = None,
        ordering: str = "standard",
    ) -> Optional[np.ndarray]:
        """Extract 4 corners from a segmentation polygon using Edge-Separated RANSAC."""
        lines, init_corners = _extract_4_edge_lines_ransac(
            polygon_points,
            threshold=self.ransac_threshold,
            max_iters=self.ransac_max_iters,
        )

        # Intersections: TL = Left(3) x Top(0), TR = Top(0) x Right(1),
        #                BR = Right(1) x Bottom(2), BL = Bottom(2) x Left(3)
        tl = _line_intersection(lines[3], lines[0], img_shape=img_shape) 
        tr = _line_intersection(lines[0], lines[1], img_shape=img_shape)
        br = _line_intersection(lines[1], lines[2], img_shape=img_shape)
        bl = _line_intersection(lines[2], lines[3], img_shape=img_shape)

        # Fallback to initial corners if intersection fails
        if tl is None: tl = init_corners[0]
        if tr is None: tr = init_corners[1]
        if br is None: br = init_corners[2]
        if bl is None: bl = init_corners[3]

        return order_corners(np.float32([tl, tr, br, bl]), ordering=ordering)

    # ---- Public API ----

    def segment_board(
        self,
        image: Union[str, Path, np.ndarray],
        ordering: str = "standard",
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Run board segmentation and extract 4 corners.

        Args:
            image: Input image path or BGR numpy array.
            ordering: Corner ordering ("standard" or "clockwise").

        Returns:
            Tuple of (ordered corners (4,2) float32, debug_info dict) or (None, None).
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not read image from {image}")
            image_path_str = str(image)
        else:
            img_bgr = image.copy()
            image_path_str = None

        h, w = img_bgr.shape[:2]
        polygons = []
        raw_results = None

        # Run inference
        if self.source_type == "local":
            if self._local_model is None:
                raise RuntimeError(
                    "Local YOLO model is not loaded. Provide model_path in constructor."
                )
            raw_results = self._local_model.predict(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            if raw_results and raw_results[0].masks is not None and len(raw_results[0].masks) > 0:
                polygons = [np.array(poly, dtype=np.float32) for poly in raw_results[0].masks.xy]

        elif self.source_type in ("roboflow", "server", "remote"):
            if self._roboflow_client is None:
                raise RuntimeError(
                    "Roboflow client is not initialized. Provide api_key in constructor."
                )
            if image_path_str:
                raw_results = self._roboflow_client.infer(image_path_str, model_id=self.model_id)
            else:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    raw_results = self._roboflow_client.infer(tmp.name, model_id=self.model_id)
            if isinstance(raw_results, dict):
                polygons = self._parse_roboflow_polygons(raw_results)

        if not polygons:
            logger.warning("No segmentation masks returned by the model.")
            return None, None

        # Select polygon with largest area
        best_poly = max(polygons, key=lambda p: cv2.contourArea(p))

        if len(best_poly) < 4:
            logger.warning("No valid segmentation polygon found.")
            return None, None

        # Fit 4 RANSAC lines and compute corner intersections
        lines, _ = _extract_4_edge_lines_ransac(
            best_poly,
            threshold=self.ransac_threshold,
            max_iters=self.ransac_max_iters,
        )

        corners = self._predict_corners_from_polygon(
            best_poly, img_shape=(h, w), ordering=ordering
        )

        debug_info = {
            "polygon": best_poly,
            "lines": lines,
            "raw_results": raw_results,
        }

        return corners, debug_info

    def extract_chessboard_perspective(
        self,
        image: Union[str, Path, np.ndarray],
        corners: Optional[np.ndarray] = None,
        target_size: int = 1200,
        ordering: str = "standard",
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[dict],
    ]:
        """Perform perspective transformation and extract 64 chessboard squares.

        If corners is None, runs segment_board() to auto-detect corners.

        Args:
            image: Input image path or BGR numpy array.
            corners: 4 ordered corner points (4, 2) or None for auto detection.
            target_size: Resolution of top-down warped board.
            ordering: Corner ordering ("standard" or "clockwise").

        Returns:
            Tuple of (warped_board, M, M_inv, square_polygons_original, cell_dict)
            or (None, None, None, None, None) if detection fails.
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not read image from {image}")
        else:
            img_bgr = image

        if corners is None:
            corners, _ = self.segment_board(image, ordering=ordering)
            if corners is None:
                return None, None, None, None, None

        return extract_chessboard_perspective(img_bgr, corners, target_size=target_size)
