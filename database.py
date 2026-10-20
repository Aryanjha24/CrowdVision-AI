"""
Enterprise AI-Powered Person Tracking & Crowd Analytics
Module: Enterprise SQLite / SQLAlchemy 2.0 Database Layer

Author: Aryan Jha / Antigravity Team
Description: Production-ready ORM schema and thread-safe Data Access Object (DAO)
             for logging real-time line-crossing foot traffic (IN/OUT), individual
             person dwell times, trajectory centroids, and periodic crowd metrics.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Index,
    func,
    desc,
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, Session

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrowdAnalyticsDB")

Base = declarative_base()


class TrackEvent(Base):
    """
    Records discreet entry and exit line-crossing events.
    """
    __tablename__ = "track_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    event_type = Column(String(10), nullable=False, index=True)  # 'IN' or 'OUT'
    line_name = Column(String(50), default="Main Zone", nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    centroid_x = Column(Float, nullable=True)
    centroid_y = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_event_type_timestamp", "event_type", "timestamp"),
        Index("idx_track_event", "track_id", "event_type"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "line_name": self.line_name,
            "confidence": round(self.confidence, 3),
            "centroid_x": round(self.centroid_x, 1) if self.centroid_x else None,
            "centroid_y": round(self.centroid_y, 1) if self.centroid_y else None,
        }


class DwellLog(Base):
    """
    Maintains persistent tracking state, dwell times, and spatial centroids
    for individual persons (supporting dwell-time heatmaps).
    """
    __tablename__ = "dwell_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    dwell_time_seconds = Column(Float, default=0.0, nullable=False)
    centroid_x = Column(Float, nullable=False, default=0.0)
    centroid_y = Column(Float, nullable=False, default=0.0)
    total_detections = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "dwell_time_seconds": round(self.dwell_time_seconds, 2),
            "centroid_x": round(self.centroid_x, 1),
            "centroid_y": round(self.centroid_y, 1),
            "total_detections": self.total_detections,
            "is_active": self.is_active,
        }


class CrowdAnalyticsSnapshot(Base):
    """
    Periodic statistical time-series aggregations of crowd density and flow.
    """
    __tablename__ = "crowd_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    current_occupancy = Column(Integer, default=0, nullable=False)
    cumulative_in = Column(Integer, default=0, nullable=False)
    cumulative_out = Column(Integer, default=0, nullable=False)
    net_occupancy = Column(Integer, default=0, nullable=False)
    avg_dwell_time_seconds = Column(Float, default=0.0, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "current_occupancy": self.current_occupancy,
            "cumulative_in": self.cumulative_in,
            "cumulative_out": self.cumulative_out,
            "net_occupancy": self.net_occupancy,
            "avg_dwell_time_seconds": round(self.avg_dwell_time_seconds, 2),
        }


class CrowdAnalyticsDB:
    """
    Enterprise Data Access Object providing thread-safe SQLite operations
    for real-time computer vision inference loops and Streamlit dashboards.
    """

    def __init__(self, db_path: str = "crowd_analytics.db"):
        self.db_path = os.path.abspath(db_path)
        # SQLite connection url with multi-threading support
        self.database_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            echo=False,
        )
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.ScopedSession = scoped_session(self.session_factory)
        self.init_db()

    def init_db(self):
        """Creates tables and indices if they do not exist."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Initialized database schema at: {self.db_path}")

    @contextmanager
    def get_session(self) -> Session:
        """Context manager for thread-safe transactional sessions."""
        session: Session = self.ScopedSession()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction error: {e}", exc_info=True)
            raise e
        finally:
            session.close()

    def log_crossing_event(
        self,
        track_id: int,
        event_type: str,
        confidence: float = 1.0,
        centroid_x: Optional[float] = None,
        centroid_y: Optional[float] = None,
        line_name: str = "Main Zone",
        timestamp: Optional[datetime] = None,
    ) -> TrackEvent:
        """
        Logs a line crossing event ('IN' or 'OUT').
        """
        ts = timestamp or datetime.now(timezone.utc)
        normalized_type = event_type.strip().upper()
        if normalized_type not in ("IN", "OUT"):
            raise ValueError(f"event_type must be 'IN' or 'OUT', got '{event_type}'")

        with self.get_session() as session:
            event = TrackEvent(
                timestamp=ts,
                track_id=track_id,
                event_type=normalized_type,
                line_name=line_name,
                confidence=confidence,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            )
            session.add(event)
            session.flush()
            logger.debug(f"Logged crossing: Track #{track_id} -> {normalized_type}")
            return event

    def update_dwell(
        self,
        track_id: int,
        centroid_x: float,
        centroid_y: float,
        timestamp: Optional[datetime] = None,
        dwell_seconds: Optional[float] = None,
    ) -> DwellLog:
        """
        Updates the dwell duration and spatial coordinates for a tracked person.
        Creates a record if the track_id is seen for the first time.
        """
        ts = timestamp or datetime.now(timezone.utc)
        with self.get_session() as session:
            record: Optional[DwellLog] = (
                session.query(DwellLog).filter(DwellLog.track_id == track_id).first()
            )

            if record is None:
                record = DwellLog(
                    track_id=track_id,
                    first_seen=ts,
                    last_seen=ts,
                    dwell_time_seconds=dwell_seconds if dwell_seconds is not None else 0.0,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    total_detections=1,
                    is_active=True,
                )
                session.add(record)
            else:
                record.last_seen = ts
                if dwell_seconds is not None:
                    record.dwell_time_seconds = dwell_seconds
                else:
                    first_seen_dt = record.first_seen
                    if first_seen_dt.tzinfo is None and ts.tzinfo is not None:
                        first_seen_dt = first_seen_dt.replace(tzinfo=timezone.utc)
                    elif first_seen_dt.tzinfo is not None and ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    delta = (ts - first_seen_dt).total_seconds()
                    record.dwell_time_seconds = max(0.0, delta)

                record.centroid_x = centroid_x
                record.centroid_y = centroid_y
                record.total_detections += 1
                record.is_active = True

            session.flush()
            return record

    def mark_person_departed(self, track_id: int):
        """Flags person as no longer actively tracked in frame."""
        with self.get_session() as session:
            record = session.query(DwellLog).filter(DwellLog.track_id == track_id).first()
            if record:
                record.is_active = False

    def log_snapshot(
        self,
        current_occupancy: int,
        cumulative_in: int,
        cumulative_out: int,
        avg_dwell_time_seconds: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> CrowdAnalyticsSnapshot:
        """Logs periodic summary metric snapshot."""
        ts = timestamp or datetime.now(timezone.utc)
        net_occ = max(0, cumulative_in - cumulative_out)
        with self.get_session() as session:
            snap = CrowdAnalyticsSnapshot(
                timestamp=ts,
                current_occupancy=current_occupancy,
                cumulative_in=cumulative_in,
                cumulative_out=cumulative_out,
                net_occupancy=net_occ,
                avg_dwell_time_seconds=avg_dwell_time_seconds,
            )
            session.add(snap)
            session.flush()
            return snap

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates current high-level KPI metrics."""
        with self.get_session() as session:
            total_in = (
                session.query(func.count(TrackEvent.id))
                .filter(TrackEvent.event_type == "IN")
                .scalar()
                or 0
            )
            total_out = (
                session.query(func.count(TrackEvent.id))
                .filter(TrackEvent.event_type == "OUT")
                .scalar()
                or 0
            )
            active_occupancy = (
                session.query(func.count(DwellLog.id))
                .filter(DwellLog.is_active == True)
                .scalar()
                or 0
            )
            avg_dwell = (
                session.query(func.avg(DwellLog.dwell_time_seconds)).scalar()
                or 0.0
            )
            max_dwell = (
                session.query(func.max(DwellLog.dwell_time_seconds)).scalar()
                or 0.0
            )
            total_tracked = (
                session.query(func.count(DwellLog.id)).scalar()
                or 0
            )

            return {
                "total_in": total_in,
                "total_out": total_out,
                "net_traffic": total_in - total_out,
                "active_occupancy": active_occupancy,
                "total_unique_persons": total_tracked,
                "average_dwell_seconds": round(float(avg_dwell), 1),
                "maximum_dwell_seconds": round(float(max_dwell), 1),
            }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent crossing events."""
        with self.get_session() as session:
            events = (
                session.query(TrackEvent)
                .order_by(desc(TrackEvent.timestamp))
                .limit(limit)
                .all()
            )
            return [e.to_dict() for e in events]

    def get_recent_snapshots(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns historical time-series snapshots."""
        with self.get_session() as session:
            snapshots = (
                session.query(CrowdAnalyticsSnapshot)
                .order_by(desc(CrowdAnalyticsSnapshot.timestamp))
                .limit(limit)
                .all()
            )
            return [s.to_dict() for s in reversed(snapshots)]

    def get_heatmap_points(self) -> List[Dict[str, float]]:
        """
        Retrieves spatial centroid coordinates with dwell times as intensity weights.
        Used to render 2D spatial crowd dwell heatmaps.
        """
        with self.get_session() as session:
            records = session.query(
                DwellLog.centroid_x,
                DwellLog.centroid_y,
                DwellLog.dwell_time_seconds,
            ).filter(DwellLog.dwell_time_seconds > 0.0).all()

            return [
                {
                    "x": float(r[0]),
                    "y": float(r[1]),
                    "weight": max(1.0, float(r[2])),
                }
                for r in records
            ]

    def reset_database(self):
        """Clears all records for a clean benchmark or live run."""
        with self.get_session() as session:
            session.query(TrackEvent).delete()
            session.query(DwellLog).delete()
            session.query(CrowdAnalyticsSnapshot).delete()
        logger.info("Cleared all records from database tables.")


def parse_args():
    parser = argparse.ArgumentParser(description="Crowd Analytics Database Manager")
    parser.add_argument("--init", action="store_true", help="Initialize schema tables.")
    parser.add_argument("--seed-test", action="store_true", help="Seed mock analytics data for testing.")
    parser.add_argument("--stats", action="store_true", help="Print current database metrics.")
    parser.add_argument("--reset", action="store_true", help="Reset all database tables.")
    parser.add_argument("--db-path", type=str, default="crowd_analytics.db", help="Path to SQLite DB.")
    return parser.parse_args()


def main():
    args = parse_args()
    db = CrowdAnalyticsDB(db_path=args.db_path)

    if args.reset:
        confirm = input("Are you sure you want to delete all database records? (y/N): ")
        if confirm.lower() == "y":
            db.reset_database()
            print("Database wiped successfully.")

    if args.seed_test:
        logger.info("Seeding realistic test dataset into SQLite...")
        import random
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        # Seed 20 persons with varied dwell times and crossings
        for person_id in range(1, 21):
            x = random.uniform(100.0, 1180.0)
            y = random.uniform(150.0, 680.0)
            dwell = random.uniform(2.5, 45.0)
            seen_time = now - timedelta(seconds=random.randint(5, 300))

            # Dwell entry
            dwell_entry = db.update_dwell(
                track_id=person_id,
                centroid_x=x,
                centroid_y=y,
                timestamp=seen_time,
                dwell_seconds=dwell,
            )
            # Crossing event
            event_type = "IN" if person_id % 3 != 0 else "OUT"
            db.log_crossing_event(
                track_id=person_id,
                event_type=event_type,
                confidence=round(random.uniform(0.82, 0.98), 2),
                centroid_x=x,
                centroid_y=y,
                timestamp=seen_time,
            )

        # Seed snapshots
        for i in range(10):
            snap_time = now - timedelta(seconds=(10 - i) * 10)
            db.log_snapshot(
                current_occupancy=random.randint(5, 15),
                cumulative_in=random.randint(12, 18),
                cumulative_out=random.randint(3, 7),
                avg_dwell_time_seconds=random.uniform(8.0, 22.0),
                timestamp=snap_time,
            )
        logger.info("Test dataset successfully populated!")

    if args.stats or args.seed_test or not any([args.init, args.seed_test, args.reset]):
        metrics = db.get_summary_metrics()
        print("\n" + "=" * 50)
        print("CROWD ANALYTICS DATABASE KPI REPORT")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k.replace('_', ' ').title():<28}: {v}")
        print("=" * 50)
        recent = db.get_recent_events(limit=5)
        print(f"\nLast {len(recent)} Line Crossing Events:")
        for r in recent:
            print(f"  [ID #{r['track_id']:02d}] {r['event_type']} | Conf: {r['confidence']} | Time: {r['timestamp']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
