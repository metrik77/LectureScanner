"""SQLite persistence for the local photo order and per-image OCR results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Photo:
    path: str
    sha256: str
    width: int
    height: int
    order_index: int
    included: bool
    status: str
    original_text: str | None = None
    corrected_text: str | None = None
    result_path: str | None = None
    error: str | None = None
    processing_method: str | None = None
    quality_status: str | None = None
    warnings_json: str | None = None
    attempt_manifest_path: str | None = None


class Store:
    def __init__(self, database: str | Path):
        self.database = Path(database).expanduser().resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS photos (
                    path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    order_index INTEGER NOT NULL,
                    included INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    original_text TEXT,
                    corrected_text TEXT,
                    result_path TEXT,
                    error TEXT,
                    processing_method TEXT,
                    quality_status TEXT,
                    warnings_json TEXT,
                    attempt_manifest_path TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    status TEXT NOT NULL DEFAULT 'idle',
                    started_at REAL,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    current_path TEXT,
                    total INTEGER NOT NULL DEFAULT 0,
                    processed INTEGER NOT NULL DEFAULT 0,
                    succeeded INTEGER NOT NULL DEFAULT 0,
                    errors INTEGER NOT NULL DEFAULT 0,
                    review_count INTEGER NOT NULL DEFAULT 0,
                    samples_json TEXT NOT NULL DEFAULT '[]'
                );
                INSERT OR IGNORE INTO job(id) VALUES(1);
                """
            )
            self._ensure_photo_columns(db)

    @staticmethod
    def _ensure_photo_columns(db: sqlite3.Connection) -> None:
        """Additive migration; it deliberately does not rewrite past OCR rows."""
        existing = {row["name"] for row in db.execute("PRAGMA table_info(photos)")}
        for name in (
            "processing_method",
            "quality_status",
            "warnings_json",
            "attempt_manifest_path",
        ):
            if name not in existing:
                db.execute(f"ALTER TABLE photos ADD COLUMN {name} TEXT")

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def dimensions(path: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            width, height = image.size
        with Image.open(path) as image:
            image.verify()
        return width, height

    def scan_folder(self, folder: str | Path) -> dict[str, int]:
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Папка не найдена: {root}")
        # Directory discovery order only provides provisional row positions.
        # Processing is disabled until the user explicitly confirms the order.
        candidates = [p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        new_count = changed_count = unreadable_count = duplicate_count = 0
        with self._connect() as db:
            existing = {row["path"]: row for row in db.execute("SELECT * FROM photos")}
            known_hashes = {row["sha256"] for row in existing.values()}
            next_index = db.execute("SELECT COALESCE(MAX(order_index), -1) + 1 FROM photos").fetchone()[0]
            for path in candidates:
                path_text = str(path)
                try:
                    digest = self.sha256(path)
                    width, height = self.dimensions(path)
                except Exception:
                    unreadable_count += 1
                    continue
                row = existing.get(path_text)
                if row is None and digest in known_hashes:
                    duplicate_count += 1
                    continue
                if row is None:
                    db.execute(
                        "INSERT INTO photos(path,sha256,width,height,order_index) VALUES(?,?,?,?,?)",
                        (path_text, digest, width, height, next_index),
                    )
                    next_index += 1
                    known_hashes.add(digest)
                    new_count += 1
                elif row["sha256"] != digest:
                    db.execute(
                        "UPDATE photos SET sha256=?,width=?,height=?,status='pending',original_text=NULL,corrected_text=NULL,result_path=NULL,error=NULL,processing_method=NULL,quality_status=NULL,warnings_json=NULL,attempt_manifest_path=NULL WHERE path=?",
                        (digest, width, height, path_text),
                    )
                    changed_count += 1
            db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('source_folder',?)", (str(root),))
            # New or changed membership invalidates confirmation.
            if new_count or changed_count:
                db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('order_confirmed','false')")
        return {"new": new_count, "changed": changed_count, "unreadable": unreadable_count, "duplicates": duplicate_count}

    def list_photos(self) -> list[Photo]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM photos ORDER BY order_index, path").fetchall()
        return [Photo(**{**dict(row), "included": bool(row["included"])}) for row in rows]

    def set_order(self, paths: list[str]) -> None:
        with self._connect() as db:
            current = {row[0] for row in db.execute("SELECT path FROM photos")}
            if set(paths) != current or len(paths) != len(current):
                raise ValueError("Порядок должен содержать каждую фотографию ровно один раз")
            db.executemany("UPDATE photos SET order_index=? WHERE path=?", [(i, p) for i, p in enumerate(paths)])
            db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('order_confirmed','true')")

    def save_order(self, paths: list[str]) -> None:
        with self._connect() as db:
            current = {row[0] for row in db.execute("SELECT path FROM photos")}
            if set(paths) != current or len(paths) != len(current):
                raise ValueError("Порядок должен содержать каждую фотографию ровно один раз")
            db.executemany("UPDATE photos SET order_index=? WHERE path=?", [(i, p) for i, p in enumerate(paths)])

    def move(self, path: str, delta: int) -> None:
        photos = self.list_photos()
        index = next(i for i, photo in enumerate(photos) if photo.path == path)
        target = max(0, min(len(photos) - 1, index + delta))
        photos.insert(target, photos.pop(index))
        with self._connect() as db:
            db.executemany("UPDATE photos SET order_index=? WHERE path=?", [(i, p.path) for i, p in enumerate(photos)])

    def set_included(self, path: str, included: bool) -> None:
        with self._connect() as db:
            db.execute("UPDATE photos SET included=? WHERE path=?", (int(included), path))

    def setting(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def eligible_photos(self) -> list[Photo]:
        if self.setting("order_confirmed") != "true":
            raise ValueError("Сначала проверьте и подтвердите порядок фотографий")
        return [p for p in self.list_photos() if p.included and p.status != "complete"]

    def save_result(
        self,
        photo_path: str,
        text: str,
        result_path: str,
        manual_review: bool,
        *,
        processing_method: str | None = None,
        quality_status: str = "usable",
        warnings: tuple[str, ...] = (),
        attempt_manifest_path: str | None = None,
    ) -> None:
        status = "complete" if quality_status == "usable" else "needs_retry"
        with self._connect() as db:
            db.execute(
                "UPDATE photos SET status=?,original_text=?,result_path=?,error=NULL,processing_method=?,quality_status=?,warnings_json=?,attempt_manifest_path=? WHERE path=?",
                (
                    status,
                    text,
                    result_path,
                    processing_method,
                    quality_status,
                    json.dumps(warnings, ensure_ascii=False),
                    attempt_manifest_path,
                    photo_path,
                ),
            )
            if status == "complete":
                db.execute(
                    "UPDATE job SET succeeded=succeeded+1,review_count=review_count+? WHERE id=1",
                    (int(manual_review),),
                )

    def save_error(self, photo_path: str, error: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE photos SET status='error',error=? WHERE path=?", (error, photo_path))
            db.execute("UPDATE job SET errors=errors+1 WHERE id=1")

    def save_corrected_text(self, photo_path: str, text: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE photos SET corrected_text=? WHERE path=?", (text, photo_path))

    def completed_records(self) -> list[Photo]:
        return [p for p in self.list_photos() if p.status == "complete"]

    def retry_records(self) -> list[Photo]:
        return [p for p in self.list_photos() if p.status == "needs_retry"]

    def job_state(self) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM job WHERE id=1").fetchone()
        return {**dict(row), "samples": json.loads(row["samples_json"])}

    def start_job(self, total: int) -> None:
        with self._connect() as db:
            db.execute("UPDATE job SET status='running',started_at=strftime('%s','now'),stop_requested=0,current_path=NULL,total=?,processed=0,succeeded=0,errors=0,review_count=0,samples_json='[]' WHERE id=1", (total,))

    def request_stop(self) -> None:
        with self._connect() as db:
            db.execute("UPDATE job SET stop_requested=1 WHERE id=1")

    def update_job(self, **values: Any) -> None:
        if not values:
            return
        assignments = ",".join(f"{key}=?" for key in values)
        params = list(values.values())
        with self._connect() as db:
            db.execute(f"UPDATE job SET {assignments} WHERE id=1", params)

    def reset_interrupted_job(self) -> None:
        with self._connect() as db:
            db.execute("UPDATE job SET status='interrupted',stop_requested=1 WHERE id=1 AND status='running'")
            db.execute("UPDATE photos SET status='pending',error=NULL WHERE status='processing'")

