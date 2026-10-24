"""
Queue Manager Module.
Handles:
- User-defined queue zones (polygons or rectangles)
- ROI filtering and entry/exit tracking
- Real-time queue counting, smoothing, and status classification (Normal/Busy/ALERT)
- Per-customer wait timing and historical served records
"""
from datetime import datetime
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

import config

def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable Mm Ss or MM:SS."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s}s"

def format_mm_ss(seconds: float) -> str:
    """Formats seconds into MM:SS format."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"

class QueueZone:
    def __init__(self, zone_id: int, points: List[Tuple[int, int]]):
        self.zone_id = zone_id
        self.points = points
        self.polygon = np.array(points, dtype=np.int32)

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Checks if a point (x, y) is inside the polygon zone."""
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0


class QueueManager:
    def __init__(self):
        self.zones: List[QueueZone] = []
        
        # Active tracking: track_id -> dict
        # { 'zone_id': int, 'entry_time': datetime, 'last_seen': datetime, 'frames_seen': int, 'confirmed': bool }
        self.active_in_zone: Dict[int, Dict[str, Any]] = {}
        
        # History of served customers
        self.served_customers: List[Dict[str, Any]] = []
        
        # Count smoothing buffer
        self.count_history = deque(maxlen=config.SMOOTHING_WINDOW)
        
        # Cache of current frame statistics
        self.current_in_queue = 0
        self.current_entering = 0
        self.current_status = "Normal"
        self.load_predictor = None
        self._predictor_initialized = False

    def get_effective_service_time(self, current_time: datetime, zone_id: int = 1, current_load: int = 1) -> float:
        """Returns dynamic LightGBM estimated wait time or fallback config average."""
        if getattr(config, "USE_LIGHTGBM_PREDICTOR", False):
            if not self._predictor_initialized:
                try:
                    from load_predictor import QueueLoadPredictor
                    predictor = QueueLoadPredictor()
                    if predictor.is_ready():
                        self.load_predictor = predictor
                except Exception:
                    self.load_predictor = None
                self._predictor_initialized = True

            if self.load_predictor is not None:
                try:
                    day_name = current_time.strftime("%A")
                    hour = current_time.hour
                    res = self.load_predictor.predict(day=day_name, hour=hour, zone=zone_id, known_load=current_load)
                    return float(res["predicted_wait_seconds"])
                except Exception:
                    pass

        return float(getattr(config, "AVERAGE_WAIT_TIME_SECONDS", 14.12))

    def add_zone(self, points: List[Tuple[int, int]]) -> int:
        """Adds a new zone and returns its zone_id."""
        zone_id = len(self.zones) + 1  # 1-indexed as in GUI screenshots
        zone = QueueZone(zone_id=zone_id, points=points)
        self.zones.append(zone)
        return zone_id

    def undo_zone(self) -> bool:
        """Removes the last added zone."""
        if self.zones:
            self.zones.pop()
            return True
        return False

    def clear_zones(self):
        """Clears all defined zones."""
        self.zones.clear()
        self.active_in_zone.clear()

    def clear_records(self):
        """Resets active tracking and served records."""
        self.active_in_zone.clear()
        self.served_customers.clear()
        self.count_history.clear()
        self.current_in_queue = 0
        self.current_entering = 0
        self.current_status = "Normal"

    def is_box_in_zone(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[bool, Optional[int]]:
        """
        Checks if a bounding box (x1, y1, x2, y2) is inside any queue zone.
        Uses bottom-center (ground contact/feet) and lower-torso point for robustness.
        Returns (is_inside, matched_zone_id).
        """
        if not self.zones:
            return False, None
        
        cx = (x1 + x2) // 2
        feet_pt = (cx, y2)
        lower_pt = (cx, int(y1 + 0.82 * (y2 - y1)))

        for zone in self.zones:
            if zone.contains_point(feet_pt) or zone.contains_point(lower_pt):
                return True, zone.zone_id

        return False, None

    def update(
        self,
        tracks: List[Tuple[int, Tuple[int, int, int, int], float]],
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Updates queue state for a single frame of tracked persons.
        :param tracks: List of (track_id, (x1, y1, x2, y2), conf)
        :param current_time: Timestamp of frame processing
        :return: Summary dictionary for GUI display
        """
        if current_time is None:
            current_time = datetime.now()

        # If no zones are defined, no customers can be in queue
        if not self.zones:
            self.active_in_zone.clear()
            self.current_in_queue = 0
            self.current_entering = 0
            self.current_status = "Normal"
            return {
                "in_queue": 0,
                "entering": 0,
                "served": len(self.served_customers),
                "status": "Normal",
                "zone_counts": {},
                "queue_table": [],
                "served_table": [
                    {
                        "id": str(s["customer_id"]),
                        "zone": str(s["zone"]),
                        "total_wait": s["wait_time_str"],
                    }
                    for s in reversed(self.served_customers[-10:])
                ],
            }

        seen_track_ids = set()
        raw_in_queue = 0
        raw_entering = 0

        # Process each tracked person in the current frame
        for track_id, (x1, y1, x2, y2), _ in tracks:
            is_in, zone_id = self.is_box_in_zone(x1, y1, x2, y2)
            if is_in and zone_id is not None:
                seen_track_ids.add(track_id)
                if track_id not in self.active_in_zone:
                    # Newly arrived in zone
                    self.active_in_zone[track_id] = {
                        "zone_id": zone_id,
                        "entry_time": current_time,
                        "last_seen": current_time,
                        "frames_seen": 1,
                        "confirmed": False,
                    }
                else:
                    # Update existing record in zone
                    rec = self.active_in_zone[track_id]
                    rec["last_seen"] = current_time
                    rec["frames_seen"] += 1
                    rec["zone_id"] = zone_id

                rec = self.active_in_zone[track_id]
                if rec["frames_seen"] >= config.MIN_FRAMES_IN_ZONE:
                    rec["confirmed"] = True
                    raw_in_queue += 1
                else:
                    raw_entering += 1

        # Check for departures (customers leaving zone or frame)
        to_remove = []
        for track_id, rec in list(self.active_in_zone.items()):
            if track_id not in seen_track_ids:
                # Time elapsed since last seen in zone
                gap_seconds = (current_time - rec["last_seen"]).total_seconds()
                if gap_seconds > 1.5:  # Grace period threshold
                    to_remove.append(track_id)
                    # If they were confirmed in queue for at least 3 seconds, mark as served
                    total_wait = (rec["last_seen"] - rec["entry_time"]).total_seconds()
                    if rec["confirmed"] and total_wait >= 3.0:
                        self.served_customers.append({
                            "customer_id": track_id,
                            "zone": rec["zone_id"],
                            "entry_time": rec["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                            "exit_time": rec["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                            "wait_time_str": format_duration(total_wait),
                            "wait_seconds": round(total_wait, 1),
                            "status": "Served",
                        })

        for tid in to_remove:
            del self.active_in_zone[tid]

        # Confirmed people currently in queue (sorted by arrival time)
        confirmed_people = [
            (tid, data) for tid, data in self.active_in_zone.items() if data.get("confirmed", False)
        ]
        confirmed_people.sort(key=lambda item: item[1]["entry_time"])

        confirmed_count = len(confirmed_people)
        self.count_history.append(confirmed_count)
        smoothed_in_queue = confirmed_count

        # Determine overload status
        if smoothed_in_queue >= config.ALERT_QUEUE_THRESHOLD:
            status = "ALERT"
        elif smoothed_in_queue >= config.BUSY_QUEUE_THRESHOLD:
            status = "Busy"
        else:
            status = "Normal"

        self.current_in_queue = smoothed_in_queue
        self.current_entering = raw_entering
        self.current_status = status

        queue_table_rows = []
        for pos, (tid, data) in enumerate(confirmed_people):
            waited_sec = (current_time - data["entry_time"]).total_seconds()
            waited_str = format_duration(waited_sec)
            
            # Estimated remaining wait: position in line * dynamic service time
            service_time = self.get_effective_service_time(
                current_time=current_time,
                zone_id=data["zone_id"],
                current_load=len(confirmed_people),
            )
            if pos == 0:
                est_wait_sec = max(1.0, round(service_time / 2.0))
            else:
                est_wait_sec = max(1.0, round(pos * service_time))
            est_wait_str = format_duration(est_wait_sec)

            queue_table_rows.append({
                "id": str(tid),
                "zone": str(data["zone_id"]),
                "waited": waited_str,
                "est_wait": est_wait_str,
            })

        # Build table data for "Recently Served" (last 10)
        recently_served_rows = []
        for s in reversed(self.served_customers[-10:]):
            recently_served_rows.append({
                "id": str(s["customer_id"]),
                "zone": str(s["zone"]),
                "total_wait": s["wait_time_str"],
            })

        # Calculate per-zone counts for status bar
        zone_counts = {z.zone_id: 0 for z in self.zones}
        for tid, data in confirmed_people:
            zid = data.get("zone_id", 1)
            if zid in zone_counts:
                zone_counts[zid] += 1

        return {
            "in_queue": smoothed_in_queue,
            "entering": raw_entering,
            "served": len(self.served_customers),
            "status": status,
            "zone_counts": zone_counts,
            "queue_table": queue_table_rows,
            "served_table": recently_served_rows,
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates operational summary statistics for reporting."""
        total_served = len(self.served_customers)
        if total_served == 0:
            return {
                "total_served": 0,
                "avg_wait": "00:00",
                "min_wait": "00:00",
                "max_wait": "00:00",
                "zone_stats": [],
            }

        waits = [c["wait_seconds"] for c in self.served_customers]
        avg_wait = sum(waits) / len(waits)
        min_wait = min(waits)
        max_wait = max(waits)

        # Per-zone statistics
        zone_groups: Dict[int, List[float]] = {}
        for c in self.served_customers:
            z = c["zone"]
            zone_groups.setdefault(z, []).append(c["wait_seconds"])

        zone_stats = []
        for z, z_waits in sorted(zone_groups.items()):
            z_avg = sum(z_waits) / len(z_waits)
            zone_stats.append({
                "zone": str(z),
                "customers": len(z_waits),
                "avg_wait": format_mm_ss(z_avg),
                "min_wait": format_mm_ss(min(z_waits)),
                "max_wait": format_mm_ss(max(z_waits)),
            })

        return {
            "total_served": total_served,
            "avg_wait": format_mm_ss(avg_wait),
            "min_wait": format_mm_ss(min_wait),
            "max_wait": format_mm_ss(max_wait),
            "zone_stats": zone_stats,
        }
