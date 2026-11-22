from pathlib import Path

from PIL import Image

from lecture_scanner.storage import Store


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 24), color).save(path)


def test_scan_deduplicates_hashes_and_persists_user_order(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    make_image(source / "lecture_b.jpg", (255, 0, 0))
    make_image(source / "lecture_a.jpg", (0, 255, 0))
    # Same bytes under a separate name should not be added twice.
    (source / "duplicate.jpg").write_bytes((source / "lecture_b.jpg").read_bytes())
    store = Store(tmp_path / "app.sqlite3")

    report = store.scan_folder(source)
    photos = store.list_photos()

    assert report["new"] == 2
    assert report["duplicates"] == 1
    assert len(photos) == 2
    assert store.setting("order_confirmed") == "false"

    user_order = [photos[1].path, photos[0].path]
    store.set_order(user_order)
    reopened = Store(tmp_path / "app.sqlite3")
    assert [photo.path for photo in reopened.list_photos()] == user_order
    assert reopened.setting("order_confirmed") == "true"


def test_unchanged_photo_is_not_eligible_after_completion(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    image = source / "slide.jpg"
    make_image(image, (10, 20, 30))
    store = Store(tmp_path / "app.sqlite3")
    store.scan_folder(source)
    photo = store.list_photos()[0]
    store.set_order([photo.path])
    store.save_result(photo.path, "raw OCR", str(tmp_path / "raw.json"), False)
    assert store.eligible_photos() == []

    make_image(image, (30, 20, 10))
    store.scan_folder(source)
    assert store.list_photos()[0].status == "pending"
    store.set_order([store.list_photos()[0].path])
    assert len(store.eligible_photos()) == 1


def test_insufficient_ocr_is_saved_for_retry_not_as_complete(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    image = source / "slide.jpg"
    make_image(image, (10, 20, 30))
    store = Store(tmp_path / "app.sqlite3")
    store.scan_folder(source)
    photo = store.list_photos()[0]
    store.set_order([photo.path])

    store.save_result(
        photo.path,
        "title",
        str(tmp_path / "raw.json"),
        False,
        processing_method="full_image",
        quality_status="needs_retry",
        warnings=("Слишком мало текста",),
        attempt_manifest_path=str(tmp_path / "attempts.json"),
    )

    assert store.completed_records() == []
    retry = store.retry_records()
    assert len(retry) == 1
    assert retry[0].processing_method == "full_image"
    assert retry[0].quality_status == "needs_retry"
    assert "Слишком мало текста" in (retry[0].warnings_json or "")

