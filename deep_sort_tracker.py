"""
Tracking module integrating Deep SORT for multi-person tracking.
Assigns persistent IDs across frames to prevent duplicate counts and enable wait timing.
"""
from typing import List, Tuple, Optional
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

import config

class DeepSortTracker:
    def __init__(
        self,
        max_age: Optional[int] = None,
        nn_budget: Optional[int] = None,
        max_cosine_distance: Optional[float] = None,
        n_init: Optional[int] = None,
    ):
        """
        Initializes the Deep SORT tracker with high-accuracy parameters.
        :param max_age: Maximum frames a track is kept without being matched (tolerates occlusion).
        :param nn_budget: Maximum appearance vectors kept per track.
        :param max_cosine_distance: Appearance cosine distance threshold (0.38 prevents false ID switches).
        :param n_init: Consecutive frames to confirm a track (2 eliminates single-frame noise).
        """
        cuda_avail = getattr(config, "USE_CUDA", False)
        self.max_age = max_age or getattr(config, "MAX_AGE", 60)
        self.nn_budget = nn_budget or getattr(config, "NN_BUDGET", 100)
        self.max_cosine_distance = max_cosine_distance or getattr(config, "MAX_COSINE_DISTANCE", 0.38)
        self.n_init = n_init or getattr(config, "MIN_HITS", 2)

        self.tracker = DeepSort(
            max_age=self.max_age,
            nn_budget=self.nn_budget,
            n_init=self.n_init,
            max_cosine_distance=self.max_cosine_distance,
            embedder="mobilenet",
            half=cuda_avail,
            bgr=True,
            embedder_gpu=cuda_avail,
        )

    def warmup(self):
        """
        Pre-warms the Deep SORT MobileNet embedder and pre-allocates CUDA memory
        so that the first video frame does NOT lag or freeze.
        """
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_det = [([100.0, 100.0, 80.0, 150.0], 0.99, "Human")]
        try:
            self.tracker.update_tracks(dummy_det, frame=dummy_frame)
            self.tracker.delete_all_tracks()
        except Exception:
            pass

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float, str]],
        frame: np.ndarray,
    ) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """
        Updates tracks with new detections from the current frame.
        :param detections: List of tuples (x1, y1, x2, y2, confidence, class_name)
        :param frame: Current BGR image frame (numpy array)
        :return: List of active confirmed tracks: (track_id, (left, top, right, bottom), confidence)
        """
        raw_detections = []
        for x1, y1, x2, y2, conf, cls_name in detections:
            # DeepSort expects [left, top, width, height]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            raw_detections.append(([float(x1), float(y1), float(w), float(h)], float(conf), cls_name))

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        active_tracks = []
        for track in tracks:
            # Only keep confirmed tracks or actively updated tracks
            if not track.is_confirmed() and track.time_since_update > 0:
                continue
            try:
                track_id = int(track.track_id)
            except (ValueError, TypeError):
                track_id = abs(hash(str(track.track_id))) % 100000
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            active_tracks.append((track_id, (x1, y1, x2, y2), track.get_det_conf() or 1.0))

        return active_tracks

    def reset(self):
        """Resets tracker state."""
        self.tracker.delete_all_tracks()
