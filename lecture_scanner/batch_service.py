"""Resumable, per-image OCR worker used by the local Streamlit UI."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from .storage import Photo, Store


class BatchRunner:
    def __init__(self, store: Store, data_dir: str | Path):
        self.store = store
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.results_dir = self.data_dir / "results"
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.store.reset_interrupted_job()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            photos = self.store.eligible_photos()
            if not photos:
                raise ValueError("Нет включённых фотографий, ожидающих обработки")
            self.store.start_job(len(photos))
            self._thread = threading.Thread(target=self._run, args=(photos,), daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.store.request_stop()

    def _run(self, photos: list[Photo]) -> None:
        samples: list[float] = []
        start_time = time.time()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Heavy runtime modules are imported only after explicit user start.
            from ocr_prototype import PaddleOCREngine
            from ocr_quality import run_quality_control
            from .processing import process_photo

            engine = PaddleOCREngine()
            for index, photo in enumerate(photos):
                job = self.store.job_state()
                if job["stop_requested"]:
                    break
                self.store.update_job(current_path=photo.path)
                item_started = time.perf_counter()
                try:
                    source = Path(photo.path)
                    outcome = process_photo(
                        source,
                        source_sha256=photo.sha256,
                        results_dir=self.results_dir,
                        recognizer=engine,
                    )
                    qc_path = self.results_dir / f"{photo.sha256}.quality.json"
                    flagged, _, _ = run_quality_control(
                        outcome.primary.report_path,
                        image_path=outcome.primary.image_path,
                        output_path=qc_path,
                        recognizer_factory=lambda: engine,
                    )
                    manual_review = (
                        outcome.primary.manual_review_required
                        or any(region.review_status == "pending" for region in flagged)
                    )
                    self.store.save_result(
                        photo.path,
                        outcome.primary.original_text,
                        str(outcome.primary.report_path),
                        manual_review,
                        processing_method=outcome.primary.method,
                        quality_status=outcome.quality_status,
                        warnings=outcome.warnings,
                        attempt_manifest_path=str(outcome.manifest_path),
                    )
                except Exception as error:
                    self.store.save_error(photo.path, f"{type(error).__name__}: {error}")
                elapsed = time.perf_counter() - item_started
                # The first image is model initialization; exclude it from ETA.
                if index > 0:
                    samples.append(elapsed)
                processed = self.store.job_state()["processed"] + 1
                self.store.update_job(processed=processed, samples_json=json.dumps(samples[-20:]))
            final_state = "stopped" if self.store.job_state()["stop_requested"] else "complete"
            self.store.update_job(status=final_state, current_path=None)
        except Exception as error:
            self.store.update_job(status="failed", current_path=None)
            # Keep collection resumable and preserve the failure in the job status.

