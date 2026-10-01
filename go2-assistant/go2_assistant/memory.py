"""Persistent semantic memory, namespaced by a stable localized map identity."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path


def clean_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Names must be text.")
    value = " ".join(value.strip().lower().split())
    if not value or len(value) > 80 or any(ord(c) < 32 for c in value):
        raise ValueError("Use a name between 1 and 80 characters.")
    return value


class Memory:
    def __init__(self, path: str | Path, map_id: str):
        if not isinstance(map_id, str) or not map_id.strip():
            raise ValueError("A stable map identity is required.")
        self.map_id = map_id
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS rooms (
            map_id TEXT NOT NULL, name TEXT NOT NULL, x REAL NOT NULL,
            y REAL NOT NULL, yaw REAL NOT NULL, source TEXT NOT NULL,
            updated_at REAL NOT NULL, PRIMARY KEY(map_id,name));
          CREATE TABLE IF NOT EXISTS objects (
            map_id TEXT NOT NULL, label TEXT NOT NULL, track_id TEXT NOT NULL,
            x REAL NOT NULL, y REAL NOT NULL, confidence REAL NOT NULL,
            room TEXT, seen_at REAL NOT NULL, PRIMARY KEY(map_id,label,track_id));
          CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY, map_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, map_id TEXT NOT NULL,
            content TEXT NOT NULL, created_at REAL NOT NULL);
        """)
        self.db.commit()

    def remember_room(self, name, x, y, yaw=0., *, source="taught"):
        name = clean_name(name)
        values = [float(x), float(y), float(yaw)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("A room needs a finite localized position.")
        with self._lock, self.db:
            self.db.execute("INSERT INTO rooms VALUES (?,?,?,?,?,?,?) ON CONFLICT(map_id,name) "
                            "DO UPDATE SET x=excluded.x,y=excluded.y,yaw=excluded.yaw,"
                            "source=excluded.source,updated_at=excluded.updated_at",
                            (self.map_id, name, *values, source, time.time()))
        return self.room(name)

    def room(self, name):
        with self._lock:
            row = self.db.execute("SELECT * FROM rooms WHERE map_id=? AND name=?",
                                  (self.map_id, clean_name(name))).fetchone()
        return dict(row) if row else None

    def rooms(self):
        with self._lock:
            return [dict(row) for row in self.db.execute(
                "SELECT * FROM rooms WHERE map_id=? ORDER BY name", (self.map_id,))]

    def nearest_room(self, x, y):
        rooms = self.rooms()
        if not rooms:
            return None
        return min(rooms, key=lambda r: math.hypot(x-r['x'], y-r['y']))['name']

    def observe_object(self, detection):
        label = clean_name(detection['label'])
        track = str(detection['track_id'])[:128]
        x, y, confidence = [float(detection[k]) for k in ('x', 'y', 'confidence')]
        if not all(math.isfinite(v) for v in (x, y, confidence)) or not 0 <= confidence <= 1:
            raise ValueError("Invalid object observation.")
        room = self.nearest_room(x, y)
        with self._lock, self.db:
            self.db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(map_id,label,track_id) "
                            "DO UPDATE SET x=excluded.x,y=excluded.y,confidence=excluded.confidence,"
                            "room=excluded.room,seen_at=excluded.seen_at",
                            (self.map_id, label, track, x, y, confidence, room, time.time()))

    def objects(self):
        with self._lock:
            return [dict(row) for row in self.db.execute(
                "SELECT * FROM objects WHERE map_id=? ORDER BY seen_at DESC", (self.map_id,))]

    def add_message(self, role, content):
        if role not in {'user', 'assistant'} or not isinstance(content, str):
            raise ValueError("Invalid conversation message.")
        with self._lock, self.db:
            self.db.execute("INSERT INTO messages(map_id,role,content,created_at) VALUES (?,?,?,?)",
                            (self.map_id, role, content[:12000], time.time()))

    def history(self, limit=30):
        with self._lock:
            rows = list(self.db.execute("SELECT role,content FROM messages WHERE map_id=? "
                                       "ORDER BY id DESC LIMIT ?", (self.map_id, limit)))
        return [dict(row) for row in reversed(rows)]

    def event(self, data):
        with self._lock, self.db:
            self.db.execute("INSERT INTO events(map_id,content,created_at) VALUES (?,?,?)",
                            (self.map_id, json.dumps(data, allow_nan=False), time.time()))

    def close(self):
        self.db.close()
