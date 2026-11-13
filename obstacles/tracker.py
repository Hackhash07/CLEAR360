"""
obstacles/tracker.py — Dynamic Obstacle Tracking & Temporal Consistency for CLEAR360
====================================================================================

Manages dynamic obstacles (e.g. animated pedestrian/mover) across synchronized frame sets:
  - Tracks centroid trajectories and estimated velocities
  - Prevents temporal ghosting by isolating synchronized four-camera frame sets
  - Demonstrates that subsequent ray carving dynamically clears previous obstacle positions
  - Quantifies the impact of timestamp latency and inter-camera desynchronization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from obstacles.clustering import Obstacle3D


@dataclass
class DynamicTrack:
    track_id: int
    semantic_class: int
    centroids: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    bboxes: List[dict] = field(default_factory=list)
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    age_frames: int = 1
    lost_frames: int = 0

    @property
    def latest_centroid(self) -> np.ndarray:
        return self.centroids[-1] if self.centroids else np.zeros(3)

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_mps))


class DynamicObstacleTracker:
    """
    Tracks dynamic obstacles across time steps and evaluates temporal consistency.

    Parameters
    ----------
    max_association_distance_m : float
        Maximum gating distance for centroid association (default 2.0 m).
    max_lost_frames : int
        Frames before dropping lost track (default 3).
    """

    def __init__(
        self,
        max_association_distance_m: float = 2.0,
        max_lost_frames: int = 3,
    ):
        self.max_dist = max_association_distance_m
        self.max_lost = max_lost_frames
        self.tracks: Dict[int, DynamicTrack] = {}
        self.next_track_id: int = 1

    def update(
        self,
        obstacles: List[Obstacle3D],
        timestamp_s: float,
    ) -> List[DynamicTrack]:
        """
        Update dynamic tracks with new frame observations.
        """
        # Filter candidate dynamic obstacles
        dynamic_obs = [o for o in obstacles if o.is_dynamic]

        if not self.tracks:
            # Initialize tracks
            for obs in dynamic_obs:
                track = DynamicTrack(
                    track_id=self.next_track_id,
                    semantic_class=obs.semantic_class,
                    centroids=[obs.centroid],
                    timestamps=[timestamp_s],
                    bboxes=[obs.bbox.to_dict()],
                    velocity_mps=np.zeros(3),
                    age_frames=1,
                    lost_frames=0,
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
            return list(self.tracks.values())

        # Associate existing tracks with detections
        track_ids = list(self.tracks.keys())
        matched_obs = set()
        matched_tracks = set()

        for tid in track_ids:
            tr = self.tracks[tid]
            last_c = tr.latest_centroid
            best_idx = None
            best_d = float("inf")

            for i, obs in enumerate(dynamic_obs):
                if i in matched_obs:
                    continue
                d = np.linalg.norm(obs.centroid - last_c)
                if d < self.max_dist and d < best_d:
                    best_d = d
                    best_idx = i

            if best_idx is not None:
                matched_obs.add(best_idx)
                matched_tracks.add(tid)
                obs = dynamic_obs[best_idx]

                # Update trajectory & velocity
                dt = max(1e-3, timestamp_s - tr.timestamps[-1])
                vel = (obs.centroid - tr.latest_centroid) / dt

                tr.centroids.append(obs.centroid)
                tr.timestamps.append(timestamp_s)
                tr.bboxes.append(obs.bbox.to_dict())
                tr.velocity_mps = 0.7 * tr.velocity_mps + 0.3 * vel
                tr.age_frames += 1
                tr.lost_frames = 0
            else:
                tr.lost_frames += 1

        # Delete expired tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].lost_frames > self.max_lost:
                del self.tracks[tid]

        # Spawn new tracks
        for i, obs in enumerate(dynamic_obs):
            if i not in matched_obs:
                track = DynamicTrack(
                    track_id=self.next_track_id,
                    semantic_class=obs.semantic_class,
                    centroids=[obs.centroid],
                    timestamps=[timestamp_s],
                    bboxes=[obs.bbox.to_dict()],
                    velocity_mps=np.zeros(3),
                    age_frames=1,
                    lost_frames=0,
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1

        return list(self.tracks.values())

    @staticmethod
    def simulate_temporal_desynchronization_ghosting(
        pcd_t0: o3d.geometry.PointCloud,
        pcd_t1: o3d.geometry.PointCloud,
    ) -> Tuple[o3d.geometry.PointCloud, float]:
        """
        Simulate the ghosting distortion that occurs when unsynchronized frames are blindly merged.
        Returns the merged corrupted cloud and the resulting bounding box dilation.
        """
        pts0 = np.asarray(pcd_t0.points)
        pts1 = np.asarray(pcd_t1.points)

        merged_pts = np.vstack([pts0, pts1])
        ghost_pcd = o3d.geometry.PointCloud()
        ghost_pcd.points = o3d.utility.Vector3dVector(merged_pts)

        # Measure spatial dilation
        span0 = pts0.max(axis=0) - pts0.min(axis=0)
        span_m = merged_pts.max(axis=0) - merged_pts.min(axis=0)
        dilation = float(np.linalg.norm(span_m - span0))

        return ghost_pcd, dilation

