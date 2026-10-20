"""
Enterprise AI-Powered Person Tracking & Crowd Analytics
Module: Core Computer Vision Tracking, ReID, Line-Crossing & Heatmap Pipeline

Author: Aryan Jha / Antigravity Team
Description: Full-featured computer vision pipeline integrating YOLOv8, BoT-SORT
             appearance-based persistent tracking, Supervision virtual line zone
             counting, motion trajectory tails, spatial dwell-time heatmaps, and
             automatic SQLite transaction logging.
"""

import os
import sys
import time
import argparse
import logging
from typing import Tuple, Dict, Any, Optional, Generator
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from database import CrowdAnalyticsDB

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrowdTracker")


class CrowdTracker:
    """
    Enterprise-grade video analytics pipeline for person detection,
    occlusion-resistant tracking with BoT-SORT, line crossing counting,
    trajectory tail tracing, and dwell-time heatmap accumulation.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_config: str = "custom_botsort.yaml",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        line_start: Optional[Tuple[int, int]] = None,
        line_end: Optional[Tuple[int, int]] = None,
        db_path: str = "crowd_analytics.db",
        enable_heatmap: bool = True,
        heatmap_alpha: float = 0.35,
        trace_length: int = 40,
    ):
        """
        Args:
            model_path: Path to YOLOv8 weights (.pt or .onnx).
            tracker_config: BoT-SORT configuration file.
            conf_threshold: Minimum detection confidence for human class.
            iou_threshold: NMS IoU threshold.
            line_start: (x, y) start coordinate for virtual counting line.
            line_end: (x, y) end coordinate for virtual counting line.
            db_path: Path to SQLite analytics database.
            enable_heatmap: Whether to overlay dwell-time heatmap on frames.
            heatmap_alpha: Transparency blending weight for heatmap (0.0 to 1.0).
            trace_length: Number of past trajectory frames to render per track.
        """
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.db = CrowdAnalyticsDB(db_path=db_path)
        self.enable_heatmap = enable_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.trace_length = trace_length

        logger.info(f"Loading YOLO tracking model from: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Fallback to default botsort if custom not found
        if not os.path.exists(self.tracker_config):
            logger.warning(f"Tracker config '{self.tracker_config}' not found. Using 'botsort.yaml'.")
            self.tracker_config = "botsort.yaml"

        # Coordinates for line counting
        self.line_start_coord = line_start
        self.line_end_coord = line_end
        self.line_zone: Optional[sv.LineZone] = None
        self.line_zone_annotator: Optional[sv.LineZoneAnnotator] = None

        # Supervision Annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=6,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            trace_length=self.trace_length,
            thickness=2,
            color_lookup=sv.ColorLookup.TRACK,
        )

        # Dwell time tracking state: {track_id: {"first_seen": timestamp, "last_seen": timestamp, "x": float, "y": float}}
        self.dwell_tracker: Dict[int, Dict[str, Any]] = {}
        self.heatmap_accumulator: Optional[np.ndarray] = None
        self.frame_dimensions: Optional[Tuple[int, int]] = None  # (width, height)

        # Telemetry
        self.fps_monitor = sv.FPSMonitor()
        self.frame_count = 0
        self.in_count = 0
        self.out_count = 0

    def _setup_line_zone(self, width: int, height: int):
        """Initializes LineZone coordinates if not explicitly provided."""
        if self.line_start_coord is None:
            # Default horizontal line across middle-lower third
            self.line_start_coord = (int(width * 0.05), int(height * 0.60))
        if self.line_end_coord is None:
            self.line_end_coord = (int(width * 0.95), int(height * 0.60))

        start_pt = sv.Point(x=self.line_start_coord[0], y=self.line_start_coord[1])
        end_pt = sv.Point(x=self.line_end_coord[0], y=self.line_end_coord[1])

        self.line_zone = sv.LineZone(
            start=start_pt,
            end=end_pt,
            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        )
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_scale=0.55,
            text_thickness=2,
            text_padding=8,
            custom_in_text="IN",
            custom_out_text="OUT",
            display_in_count=True,
            display_out_count=True,
        )
        logger.info(f"Initialized LineZone from {self.line_start_coord} to {self.line_end_coord}")

    def update_dwell_times(
        self,
        detections: sv.Detections,
        current_time: float,
    ) -> Dict[int, float]:
        """
        Updates persistent dwell times and centroids for active tracks.
        """
        active_dwells: Dict[int, float] = {}
        if detections.tracker_id is None:
            return active_dwells

        for idx, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            bbox = detections.xyxy[idx]
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float(bbox[3])  # bottom center anchor

            if track_id not in self.dwell_tracker:
                self.dwell_tracker[track_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "x": cx,
                    "y": cy,
                }
                dwell_sec = 0.0
            else:
                self.dwell_tracker[track_id]["last_seen"] = current_time
                self.dwell_tracker[track_id]["x"] = cx
                self.dwell_tracker[track_id]["y"] = cy
                dwell_sec = current_time - self.dwell_tracker[track_id]["first_seen"]

            active_dwells[track_id] = dwell_sec

            # Periodically sync dwell to SQLite (every ~10 frames)
            if self.frame_count % 10 == 0:
                self.db.update_dwell(
                    track_id=track_id,
                    centroid_x=cx,
                    centroid_y=cy,
                    dwell_seconds=dwell_sec,
                )

        return active_dwells

    def accumulate_heatmap(self, detections: sv.Detections, height: int, width: int):
        """
        Accumulates spatial intensity for dwell heatmaps based on person centroids.
        """
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

        if detections.tracker_id is None or len(detections) == 0:
            return

        for bbox in detections.xyxy:
            cx = int((bbox[0] + bbox[2]) / 2.0)
            cy = int(bbox[3])  # feet position
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)

            # Draw a Gaussian splat at centroid
            cv2.circle(self.heatmap_accumulator, (cx, cy), radius=28, color=1.0, thickness=-1)

    def generate_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Generates and blends the colored 2D dwell-time heatmap overlay onto frame.
        """
        if self.heatmap_accumulator is None or np.max(self.heatmap_accumulator) == 0:
            return frame

        # Apply spatial Gaussian blur to smooth the accumulator
        blurred = cv2.GaussianBlur(self.heatmap_accumulator, (45, 45), 0)
        norm_map = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply Turbo or Jet colormap
        heatmap_colored = cv2.applyColorMap(norm_map, cv2.COLORMAP_TURBO)

        # Create mask for low intensity regions (so cold background remains natural)
        _, mask = cv2.threshold(norm_map, 20, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.merge([mask, mask, mask])

        # Blend only where heatmap has non-zero intensity
        blended = cv2.addWeighted(frame, 1.0, heatmap_colored, self.heatmap_alpha, 0)
        output = np.where(mask_3ch > 0, blended, frame)
        return output

    def draw_telemetry_hud(
        self,
        frame: np.ndarray,
        active_count: int,
        fps: float,
    ) -> np.ndarray:
        """
        Draws an enterprise dark-glass telemetry dashboard card on the video frame.
        """
        h, w = frame.shape[:2]
        # Top-left HUD card
        card_w, card_h = 320, 110
        card_x, card_y = 20, 20

        sub_img = frame[card_y : card_y + card_h, card_x : card_x + card_w]
        black_rect = np.full(sub_img.shape, 20, dtype=np.uint8)
        # 60% opacity dark background
        res = cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 1.0)
        frame[card_y : card_y + card_h, card_x : card_x + card_w] = res

        # Outline border
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (60, 220, 130),
            1,
            cv2.LINE_AA,
        )

        # Header title
        cv2.putText(
            frame,
            "CROWD ANALYTICS AI",
            (card_x + 12, card_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 130),
            2,
            cv2.LINE_AA,
        )

        # Metrics values
        net_traffic = self.in_count - self.out_count
        stat_line1 = f"Occupancy: {active_count:02d} | Flow Net: {net_traffic:+d}"
        stat_line2 = f"Entered: {self.in_count:02d}   | Exited: {self.out_count:02d}"
        stat_line3 = f"FPS: {fps:.1f}   | Tracks: {len(self.dwell_tracker)}"

        cv2.putText(
            frame,
            stat_line1,
            (card_x + 12, card_y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            stat_line2,
            (card_x + 12, card_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            stat_line3,
            (card_x + 12, card_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return frame

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes end-to-end detection, BoT-SORT tracking, line triggers,
        trajectory annotation, heatmap overlay, and database sync for a single frame.

        Args:
            frame: BGR numpy image frame.
            timestamp: Frame presentation timestamp in seconds (optional).

        Returns:
            Tuple of (annotated_bgr_frame, frame_telemetry_dict).
        """
        self.fps_monitor.tick()
        self.frame_count += 1
        current_time = timestamp if timestamp is not None else time.time()
        height, width = frame.shape[:2]

        if self.line_zone is None:
            self._setup_line_zone(width, height)

        # 1. Run YOLO inference with BoT-SORT tracker
        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            persist=True,
            classes=[0],  # Class 0 = person
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        # 2. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results[0])

        # Filter detections without valid tracker_id
        if detections.tracker_id is not None and len(detections) > 0:
            valid_mask = detections.tracker_id >= 0
            detections = detections[valid_mask]
        else:
            detections = sv.Detections.empty()

        active_count = len(detections)

        # 3. Update Dwell Times & Persistent DB states
        active_dwells = self.update_dwell_times(detections, current_time)

        # 4. Trigger Line Crossing Detection
        if self.line_zone is not None and len(detections) > 0:
            in_mask, out_mask = self.line_zone.trigger(detections=detections)
            self.in_count = self.line_zone.in_count
            self.out_count = self.line_zone.out_count

            # Log events to SQLite
            for idx in range(len(detections)):
                t_id = int(detections.tracker_id[idx])
                conf = float(detections.confidence[idx])
                bbox = detections.xyxy[idx]
                cx = float((bbox[0] + bbox[2]) / 2.0)
                cy = float(bbox[3])

                if in_mask[idx]:
                    self.db.log_crossing_event(
                        track_id=t_id,
                        event_type="IN",
                        confidence=conf,
                        centroid_x=cx,
                        centroid_y=cy,
                    )
                    logger.info(f"Line Crossing [IN]: Person #{t_id} (Conf: {conf:.2f})")

                if out_mask[idx]:
                    self.db.log_crossing_event(
                        track_id=t_id,
                        event_type="OUT",
                        confidence=conf,
                        centroid_x=cx,
                        centroid_y=cy,
                    )
                    logger.info(f"Line Crossing [OUT]: Person #{t_id} (Conf: {conf:.2f})")

        # 5. Accumulate Dwell Heatmap
        if self.enable_heatmap:
            self.accumulate_heatmap(detections, height, width)

        # 6. Periodic Crowd Snapshot into SQLite (every 30 frames)
        if self.frame_count % 30 == 0:
            avg_dwell = float(np.mean(list(active_dwells.values()))) if active_dwells else 0.0
            self.db.log_snapshot(
                current_occupancy=active_count,
                cumulative_in=self.in_count,
                cumulative_out=self.out_count,
                avg_dwell_time_seconds=avg_dwell,
            )

        # 7. Render Annotations
        annotated_frame = frame.copy()

        # Render Heatmap underlay if enabled
        if self.enable_heatmap:
            annotated_frame = self.generate_heatmap_overlay(annotated_frame)

        # Trajectory Tails
        if len(detections) > 0:
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )

        # Bounding Boxes
        if len(detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )

            # Labels with Track ID and Dwell Time badge
            labels = [
                f"#{t_id} | {active_dwells.get(t_id, 0.0):.1f}s"
                for t_id in detections.tracker_id
            ]
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )

        # Virtual Counting Line
        if self.line_zone_annotator is not None and self.line_zone is not None:
            annotated_frame = self.line_zone_annotator.annotate(
                annotated_frame,
                line_counter=self.line_zone,
            )

        # Telemetry HUD Card
        fps = self.fps_monitor.fps if self.fps_monitor.fps > 0 else 0.0
        annotated_frame = self.draw_telemetry_hud(annotated_frame, active_count, fps)

        telemetry = {
            "frame": self.frame_count,
            "fps": round(fps, 1),
            "active_occupancy": active_count,
            "cumulative_in": self.in_count,
            "cumulative_out": self.out_count,
            "net_flow": self.in_count - self.out_count,
            "total_tracked": len(self.dwell_tracker),
        }

        return annotated_frame, telemetry

    def process_video_stream(
        self,
        video_source: str,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """
        Generator for streaming annotated frames to Streamlit or web feeds.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source at: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        current_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, telemetry = self.process_frame(
                    frame=frame,
                    timestamp=current_time,
                )
                yield annotated_frame, telemetry

                current_time += frame_delay
                if max_frames and self.frame_count >= max_frames:
                    break
        finally:
            cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="AI Person Tracking & Crowd Analytics")
    parser.add_argument("--source", type=str, default="1000296905.mp4", help="Video file or webcam index (0).")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 weights path.")
    parser.add_argument("--tracker", type=str, default="custom_botsort.yaml", help="BoT-SORT config file.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--heatmap", action="store_true", default=True, help="Enable dwell heatmap overlay.")
    parser.add_argument("--save-out", type=str, default=None, help="Save processed video to output file.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process.")
    parser.add_argument("--show", action="store_true", help="Display OpenCV live GUI window.")
    return parser.parse_args()


def main():
    args = parse_args()

    tracker = CrowdTracker(
        model_path=args.weights,
        tracker_config=args.tracker,
        conf_threshold=args.conf,
        enable_heatmap=args.heatmap,
    )

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("=" * 60)
    logger.info("Starting Crowd Analytics Pipeline Execution")
    logger.info("=" * 60)
    logger.info(f"Video Source : {args.source} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
    logger.info(f"YOLO Model   : {args.weights}")
    logger.info(f"Tracker      : {args.tracker} (BoT-SORT with ReID & Motion Compensation)")
    logger.info(f"Heatmap      : {args.heatmap}")
    logger.info("-" * 60)

    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_out, fourcc, fps, (width, height))
        logger.info(f"Saving output video to: {args.save_out}")

    start_time = time.time()
    try:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            simulated_ts = start_time + (frame_idx / fps)
            annotated_frame, telemetry = tracker.process_frame(frame, timestamp=simulated_ts)

            if writer:
                writer.write(annotated_frame)

            if args.show:
                cv2.imshow("Enterprise Crowd Analytics - AI Pipeline", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Interrupted by user.")
                    break

            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                logger.info(
                    f"Frame [{frame_idx:04d}/{total_frames:04d}] ({pct:.1f}%) | "
                    f"Occupancy: {telemetry['active_occupancy']:02d} | "
                    f"IN: {telemetry['cumulative_in']:02d} | "
                    f"OUT: {telemetry['cumulative_out']:02d} | "
                    f"FPS: {telemetry['fps']}"
                )

            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Execution Completed in {total_time:.2f}s ({frame_idx / total_time:.1f} avg FPS)")
    summary = tracker.db.get_summary_metrics()
    logger.info(f"Total Cumulative IN  : {summary['total_in']}")
    logger.info(f"Total Cumulative OUT : {summary['total_out']}")
    logger.info(f"Unique Persons Seen  : {summary['total_unique_persons']}")
    logger.info(f"Average Dwell Time   : {summary['average_dwell_seconds']}s")
    logger.info(f"Maximum Dwell Time   : {summary['maximum_dwell_seconds']}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
