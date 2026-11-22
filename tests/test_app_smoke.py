from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_lists_synthetic_demo_photos(tmp_path, monkeypatch):
    monkeypatch.setenv("LECTURE_SCANNER_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("LECTURE_SCANNER_INPUT_DIR", str(ROOT / "tests" / "assets"))
    from app import get_runner, get_store

    get_runner.clear()
    get_store.clear()
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Добавить фотографии").click().run(timeout=60)
    assert not app.exception
    assert "2" in [metric.value for metric in app.metric]
    assert len(app.image) >= 1
    assert any(button.label == "Подтвердить текущий порядок" for button in app.button)
    assert get_store(str(tmp_path / "app_data" / "lecture_scanner.sqlite3")).job_state()["processed"] == 0
