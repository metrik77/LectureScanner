"""Локальный интерфейс Lecture Scanner. OCR запускается только вручную."""

from __future__ import annotations

import os
import json
import subprocess
import time
from pathlib import Path

import streamlit as st

from lecture_scanner.batch_service import BatchRunner
from lecture_scanner.exports import build_docx, build_organized_docx
from lecture_scanner.organizer import SourceSlide, organize, to_markdown, to_text
from lecture_scanner.storage import Store


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LECTURE_SCANNER_DATA_DIR", ROOT / "app_data")).expanduser().resolve()
DB_PATH = DATA_DIR / "lecture_scanner.sqlite3"
PAGE_SIZE = 24
DEFAULT_INPUT_DIR = Path(os.environ.get("LECTURE_SCANNER_INPUT_DIR", ROOT / "input_photos")).expanduser().resolve()


@st.cache_resource
def get_store(database: str) -> Store:
    return Store(database)


@st.cache_resource
def get_runner(database: str, data_dir: str) -> BatchRunner:
    return BatchRunner(get_store(database), data_dir)


def _select_folder() -> str | None:
    if os.name != "posix" or not Path("/usr/bin/osascript").exists():
        st.info("Выбор папки через системное окно доступен на macOS. Укажите путь ниже.")
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", 'POSIX path of (choose folder with prompt "Выберите папку с фотографиями лекции")'],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _pretty_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"около {seconds} с"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"около {minutes} мин {remainder // 10 * 10} с"
    hours, minutes = divmod(minutes, 60)
    return f"около {hours} ч {minutes} мин"


def main() -> None:
    st.set_page_config(page_title="Lecture Scanner", page_icon="📚", layout="wide")
    st.title("Lecture Scanner")
    st.caption("Локальная обработка фотографий лекций · данные остаются на этом Mac")
    store = get_store(str(DB_PATH))
    runner = get_runner(str(DB_PATH), str(DATA_DIR))

    upload_tab, order_tab, process_tab, results_tab = st.tabs(
        ["Загрузка фотографий", "Порядок фотографий", "Обработка лекции", "Результаты и экспорт"]
    )

    with upload_tab:
        st.subheader("Добавить папку с фотографиями")
        saved_folder = store.setting("source_folder", str(DEFAULT_INPUT_DIR))
        folder_path = st.text_input("Путь к папке на этом Mac", value=saved_folder or "", key="folder_path")
        choose_col, scan_col, count_col = st.columns([1, 1, 2])
        with choose_col:
            if st.button("Выбрать папку…", width="stretch"):
                chosen = _select_folder()
                if chosen:
                    st.session_state.folder_path = chosen
                    st.rerun()
        with scan_col:
            if st.button("Добавить фотографии", type="primary", width="stretch"):
                try:
                    result = store.scan_folder(folder_path)
                    st.success(
                        f"Добавлено: {result['new']}; изменено: {result['changed']}; "
                        f"нечитаемых: {result['unreadable']}; дубликатов пропущено: {result['duplicates']}."
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Не удалось просканировать папку: {type(error).__name__}: {error}")
        photos = store.list_photos()
        with count_col:
            st.metric("Фотографий в коллекции", len(photos))
        if photos:
            st.caption("Повторное добавление того же пути и файлов с одинаковым содержимым пропускается.")
            st.info("Порядок из файловой системы считается предварительным. Проверьте его во вкладке «Порядок фотографий».")

    photos = store.list_photos()
    with order_tab:
        st.subheader("Порядок и включение снимков")
        total_included = sum(photo.included for photo in photos)
        st.write(f"В коллекции **{len(photos)}** фотографий; включено в обработку: **{total_included}**.")
        order_confirmed = store.setting("order_confirmed") == "true"
        if photos and not order_confirmed:
            st.warning("Порядок пока не подтверждён. Обработка будет недоступна до проверки и подтверждения.")
        if photos:
            page_count = max(1, (len(photos) + PAGE_SIZE - 1) // PAGE_SIZE)
            page = st.number_input("Страница", min_value=1, max_value=page_count, value=1, step=1, key="photo_page")
            start = (int(page) - 1) * PAGE_SIZE
            page_photos = photos[start : start + PAGE_SIZE]
            st.caption(f"Показаны фотографии {start + 1}–{start + len(page_photos)} из {len(photos)}. Перемещение сохраняется сразу.")
            for index, photo in enumerate(page_photos, start=start):
                with st.container(border=True):
                    left, details, controls = st.columns([1, 3, 2])
                    with left:
                        st.image(photo.path, width=190, caption=f"№ {index + 1}")
                    with details:
                        st.markdown(f"**{index + 1}. {Path(photo.path).name}**")
                        st.caption(f"{photo.width} × {photo.height} · {photo.status}")
                        if photo.error:
                            st.error(photo.error)
                    with controls:
                        checked = st.checkbox("Обрабатывать", value=photo.included, key=f"include_{photo.sha256}")
                        if checked != photo.included:
                            store.set_included(photo.path, checked)
                        up, down = st.columns(2)
                        with up:
                            if st.button("↑ Выше", key=f"up_{photo.sha256}", disabled=index == 0, width="stretch"):
                                store.move(photo.path, -1)
                                st.rerun()
                        with down:
                            if st.button("↓ Ниже", key=f"down_{photo.sha256}", disabled=index == len(photos) - 1, width="stretch"):
                                store.move(photo.path, 1)
                                st.rerun()
            if st.button("Подтвердить текущий порядок", type="primary"):
                store.set_order([photo.path for photo in store.list_photos()])
                st.success("Порядок сохранён и подтверждён. Он будет использоваться при обработке.")
                st.rerun()
        else:
            st.info("Добавьте папку с фотографиями во вкладке «Загрузка фотографий».")

    with process_tab:
        st.subheader("Панель обработки")
        eligible_count = sum(photo.included and photo.status != "complete" for photo in photos)
        if store.setting("order_confirmed") != "true":
            st.warning("Сначала задайте и подтвердите порядок фотографий.")
        else:
            st.caption("Обработка запускается только по кнопке. Результаты записываются после каждого снимка; остановка срабатывает после текущей фотографии.")

        @st.fragment(run_every="2s")
        def progress_panel() -> None:
            current = store.job_state()
            action_col, stop_col, status_col = st.columns([1, 1, 2])
            with action_col:
                if st.button("Начать обработку / продолжить", type="primary", disabled=store.setting("order_confirmed") != "true" or eligible_count == 0):
                    try:
                        runner.start()
                        st.rerun()
                    except Exception as error:
                        st.error(str(error))
            with stop_col:
                if st.button("Остановить", disabled=current["status"] != "running"):
                    runner.stop()
                    st.rerun()
            with status_col:
                st.write(f"Состояние: **{current['status']}**")
            done = int(current["processed"])
            total = int(current["total"])
            st.progress(min(done / total, 1.0) if total else 0.0, text=f"Обработано фотографий: {done} / {total}")
            elapsed = max(0.0, time.time() - float(current["started_at"] or time.time()))
            durations = current["samples"]
            speed = (len(durations) / sum(durations)) if len(durations) >= 2 and sum(durations) else None
            eta = ((total - done) / speed) if speed and total >= done else None
            metric_cols = st.columns(4)
            metric_cols[0].metric("Успешно", current["succeeded"])
            metric_cols[1].metric("Ошибки", current["errors"])
            metric_cols[2].metric("Ручная проверка", current["review_count"])
            metric_cols[3].metric("Средняя скорость", f"{speed:.2f} фото/с" if speed else "считается")
            st.write(f"Текущая фотография: **{Path(current['current_path']).name if current['current_path'] else '—'}**")
            st.write(f"Время с начала: {_pretty_seconds(elapsed)}")
            st.write(f"Осталось: {_pretty_seconds(eta) if eta is not None else 'оценка появится после трёх фотографий'}")

        progress_panel()

    with results_tab:
        st.subheader("Результаты и экспорт")
        if st.button("Обновить результаты", key="refresh_results"):
            st.rerun()
        all_photos = store.list_photos()
        completed = store.completed_records()
        retry_records = store.retry_records()
        excluded_records = [photo for photo in all_photos if photo.status != "complete"]
        status_labels = {
            "pending": "ожидает OCR",
            "processing": "обрабатывается",
            "needs_retry": "требует проверки / повтора",
            "error": "ошибка OCR",
        }
        manifest_lines = ["ФОТОГРАФИЯ\\tСТАТУС\\tВКЛЮЧЕНА\\tПРИЧИНА"]
        for photo in all_photos:
            reason = photo.error or (
                "; ".join(json.loads(photo.warnings_json or "[]"))
                if photo.warnings_json else ""
            )
            manifest_lines.append(
                f"{Path(photo.path).name}\\t{status_labels.get(photo.status, photo.status)}"
                f"\\t{'да' if photo.included else 'нет'}\\t{reason}"
            )
        st.caption(
            f"Всего фотографий: {len(all_photos)} · успешно распознано: {len(completed)} · "
            f"не вошло в чистовой экспорт: {len(excluded_records)}"
        )
        st.download_button(
            "Скачать ведомость всех фотографий и статусов",
            "\\n".join(manifest_lines) + "\\n",
            file_name="lecture_sources_status.tsv",
            mime="text/tab-separated-values",
            key="download_source_manifest",
        )
        if retry_records:
            st.warning(
                f"Фотографий, требующих повторной обработки: {len(retry_records)}. "
                "Исходные OCR-варианты сохранены и не попадают в экспорт до проверки качества."
            )
            with st.expander("Показать фотографии для повторной обработки"):
                for photo in retry_records:
                    st.markdown(f"**{Path(photo.path).name}** · {photo.processing_method or 'способ не указан'}")
                    if photo.warnings_json:
                        for warning in json.loads(photo.warnings_json):
                            st.caption(warning)
                    if photo.result_path and Path(photo.result_path).is_file():
                        st.download_button(
                            "Скачать исходный OCR JSON",
                            Path(photo.result_path).read_text(encoding="utf-8"),
                            file_name=f"{Path(photo.path).stem}.ocr.json",
                            mime="application/json",
                            key=f"retry_download_{photo.sha256}",
                        )
        if not completed:
            st.info("Завершённых OCR-результатов пока нет. До запуска обработки экспорт недоступен.")
        else:
            st.markdown("**Экспорт всей обработанной лекции**")
            combined_text = "\n\n".join(
                f"## {Path(photo.path).name}\n\n{(photo.corrected_text if photo.corrected_text is not None else photo.original_text) or ''}"
                for photo in completed
            )
            combined_plain = "\n\n".join(
                f"[{Path(photo.path).name}]\n{(photo.corrected_text if photo.corrected_text is not None else photo.original_text) or ''}"
                for photo in completed
            )
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st.download_button("Скачать лекцию Markdown", combined_text + "\n", file_name="lecture.md", mime="text/markdown")
            with export_col2:
                st.download_button("Скачать лекцию TXT", combined_plain + "\n", file_name="lecture.txt", mime="text/plain")
            docx_content = build_docx(
                (
                    Path(photo.path).name,
                    (photo.corrected_text if photo.corrected_text is not None else photo.original_text) or "",
                )
                for photo in completed
            )
            st.download_button("Скачать лекцию DOCX", docx_content, file_name="lecture.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            organized = organize(
                SourceSlide(
                    Path(photo.path).name,
                    (photo.corrected_text if photo.corrected_text is not None else photo.original_text) or "",
                )
                for photo in completed
            )
            st.markdown("**Тематическая группировка (локальная, по ключевым словам)**")
            st.caption(f"Тем: {len(organized)}. Группировка эвристическая: объединяются соседние слайды с общими словами; исходный OCR и имя источника сохраняются.")
            organizer_col1, organizer_col2, organizer_col3 = st.columns(3)
            with organizer_col1:
                st.download_button("Organizer Markdown", to_markdown(organized), file_name="lecture_organized.md", mime="text/markdown")
            with organizer_col2:
                st.download_button("Organizer TXT", to_text(organized), file_name="lecture_organized.txt", mime="text/plain")
            with organizer_col3:
                st.download_button("Organizer DOCX", build_organized_docx(organized), file_name="lecture_organized.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            options = {f"{i + 1}. {Path(photo.path).name}": photo for i, photo in enumerate(completed)}
            selected_label = st.selectbox("Фотография", list(options))
            selected = options[selected_label]
            quality_path = DATA_DIR / "results" / f"{selected.sha256}.quality.json"
            if quality_path.is_file():
                try:
                    quality = json.loads(quality_path.read_text(encoding="utf-8"))
                    flagged_regions = [region for region in quality.get("regions", []) if region.get("flags")]
                    st.write(f"Областей для ручной проверки: **{len(flagged_regions)}**")
                    for region in flagged_regions:
                        st.markdown(f"**Область {region['region_id']}** · confidence {region['original_confidence']:.3f} · {', '.join(region['flags'])}")
                        st.write(f"Исходный OCR: {region['original_text']}")
                        if region.get("retry_text") is not None:
                            st.write(f"Повторный OCR: {region['retry_text']} (confidence {region.get('retry_confidence')})")
                except (OSError, ValueError, KeyError, TypeError):
                    st.warning("Не удалось прочитать сохранённые предложения QC.")
            st.markdown("**Исходный OCR — сохранён без изменений**")
            st.text_area("Распознанный текст", value=selected.original_text or "", height=220, disabled=True, key=f"original_{selected.sha256}")
            corrected = st.text_area("Исправленный текст (сохраняется отдельно)", value=selected.corrected_text or selected.original_text or "", height=220, key=f"corrected_{selected.sha256}")
            if st.button("Сохранить ручную правку", key=f"save_corrected_{selected.sha256}"):
                store.save_corrected_text(selected.path, corrected)
                st.success("Ручная версия сохранена отдельно от исходного OCR.")
            result_json = Path(selected.result_path).read_text(encoding="utf-8") if selected.result_path and Path(selected.result_path).is_file() else "{}"
            stem = Path(selected.path).stem
            st.download_button("Скачать исходный OCR JSON", result_json, file_name=f"{stem}.ocr.json", mime="application/json")
            md = (selected.corrected_text or selected.original_text or "").strip()
            st.download_button("Скачать текущий текст Markdown", f"# {stem}\n\n{md}\n", file_name=f"{stem}.md", mime="text/markdown")
            st.download_button("Скачать текущий текст TXT", md + "\n", file_name=f"{stem}.txt", mime="text/plain")


if __name__ == "__main__":
    main()

