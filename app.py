from __future__ import annotations

# import os  # <-- 안 쓰면 지워도 됩니다
import uuid
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import gradio as gr

from watermark.configs import UPLOAD_DIR, OUTPUT_DIR
from watermark.embedder import embed_watermark, decode_watermark
from watermark.edit.presets import PRESETS, FIT_STRATEGIES
from watermark.edit.image_ops import edit_pipeline


ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}


# ---------------------------
# 유틸
# ---------------------------
def _ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_outputs():
    if not OUTPUT_DIR.exists():
        return
    for p in OUTPUT_DIR.iterdir():
        # ZIP까지 지울지 여부는 필요시 조정
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass


def _copy_to_uploads(files: List[Path]) -> List[Path]:
    saved = []
    for f in files:
        if f is None:
            continue
        p = Path(f.name if hasattr(f, "name") else f)
        ext = p.suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        dst = UPLOAD_DIR / p.name
        shutil.copyfile(p, dst)
        saved.append(dst)
    return saved


# ---------------------------
# Embed 핸들러
# ---------------------------
def do_embed(
    files: List,                # gr.File(file_count="multiple")
    use_uuid_per_file: bool,    # 파일별 UUID 생성
    custom_payload: str,        # 사용자가 직접 입력(모든 파일 동일)
    clean_prev: bool,           # 실행 전 outputs 비우기
) -> Tuple[str, str, List[str]]:
    """
    반환: (zip 경로, 로그 텍스트, 갤러리 미리보기 경로들)
    """
    _ensure_dirs()
    if clean_prev:
        _clean_outputs()

    # 업로드 저장
    file_paths = _copy_to_uploads(files or [])
    if not file_paths:
        return ("", "이미지를 선택해 주세요(PNG/JPG).", [])

    mappings = []
    previews = []

    for in_path in file_paths:
        stem, suffix = in_path.stem, in_path.suffix

        if use_uuid_per_file:
            payload = uuid.uuid4().hex
        else:
            payload = custom_payload.strip() or uuid.uuid4().hex  # fallback

        out_name = f"{stem}_watermarked{suffix}"
        out_path = OUTPUT_DIR / out_name

        embed_watermark(str(in_path), str(out_path), payload)

        mappings.append((out_name, payload))
        previews.append(str(out_path))

    # 로그 저장 + 화면용 문자열
    log_path = OUTPUT_DIR / "watermarks.txt"
    with log_path.open("w", encoding="utf-8") as f:
        for fname, u in mappings:
            f.write(f"{fname} -> {u}\n")
    log_text = "\n".join(f"{fname} -> {u}" for fname, u in mappings)

    # ZIP 묶기
    zip_base = OUTPUT_DIR / "watermarked_results"
    shutil.make_archive(str(zip_base), "zip", root_dir=str(OUTPUT_DIR))
    zip_path = str(zip_base) + ".zip"

    return (zip_path, log_text, previews)


# ---------------------------
# Decode 핸들러
# ---------------------------
def do_decode(file) -> str:
    if not file:
        return "이미지를 선택해 주세요."
    try:
        msg = decode_watermark(file.name if hasattr(file, "name") else str(file))
        return msg if msg else "(복원된 문자열이 비었습니다)"
    except Exception as e:
        return f"복원 실패: {e}"


def _safe_size(w: Optional[float], h: Optional[float]) -> Optional[Tuple[int, int]]:
    if w is None or h is None:
        return None
    try:
        return int(w), int(h)
    except Exception:
        return None

def do_embed_with_edit(
    files: List,                # 업로드 파일들
    use_uuid_per_file: bool,    # 파일별 UUID 생성
    custom_payload: str,        # 사용자 지정 페이로드(모든 파일 동일)
    clean_prev: bool,           # 실행 전 outputs 비우기
    preset_name: str,           # 프리셋 키
    cw: Optional[float], ch: Optional[float],  # 맞춤 폭/높이
    fit_mode: str,              # 'crop' | 'pad'
    mw: Optional[float], mh: Optional[float],  # 최대 가로/세로
    sharp: float,               # 샤프닝 강도
) -> Tuple[str, str, List[str]]:
    """
    반환: (zip 경로, 로그 텍스트, 미리보기 경로들)
    """
    _ensure_dirs()
    if clean_prev:
        _clean_outputs()

    # 프리셋/입력 파라미터 해석
    preset_size = PRESETS.get(preset_name).size if preset_name in PRESETS else None
    custom_size = _safe_size(cw, ch)
    max_side   = _safe_size(mw, mh)

    file_paths = _copy_to_uploads(files or [])
    if not file_paths:
        return ("", "이미지를 선택해 주세요(PNG/JPG).", [])

    mappings, previews = [], []
    edited_dir = OUTPUT_DIR / "edited"
    edited_dir.mkdir(parents=True, exist_ok=True)

    for in_path in file_paths:
        stem, suffix = in_path.stem, in_path.suffix

        # 1) 편집
        img = cv2.imread(str(in_path))
        if img is None:
            continue

        edited = edit_pipeline(
            img,
            preset_size=preset_size,
            custom_size=custom_size,
            fit_strategy=fit_mode,
            max_side=max_side,
            sharpen_amount=float(sharp or 0.0),
        )
        edited_path = edited_dir / f"{stem}_edited{suffix}"
        cv2.imwrite(str(edited_path), edited)

        # 2) 워터마크
        payload = (uuid.uuid4().hex if use_uuid_per_file
                   else (custom_payload.strip() or uuid.uuid4().hex))
        out_path = OUTPUT_DIR / f"{stem}_watermarked{suffix}"
        embed_watermark(str(edited_path), str(out_path), payload)

        mappings.append((in_path.name, edited_path.name, out_path.name, payload))
        previews.append(str(out_path))

    # 로그 저장
    log_path = OUTPUT_DIR / "watermarks.txt"
    with log_path.open("w", encoding="utf-8") as f:
        for orig, ed, wm, u in mappings:
            f.write(f"{orig} -> {ed} -> {wm} -> {u}\n")

    log_text = "\n".join(f"{orig} -> {ed} -> {wm} -> {u}" for orig, ed, wm, u in mappings)

    # ZIP 묶기
    zip_base = OUTPUT_DIR / "watermarked_results"
    shutil.make_archive(str(zip_base), "zip", root_dir=str(OUTPUT_DIR))
    return (str(zip_base) + ".zip", log_text, previews)


# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="보이지 않는 워터마크 적용기") as iface:
    gr.Markdown(
        "## 보이지 않는 워터마크 적용기\n"
        "이미지를 업로드해서 **워터마크 임베드**하거나, 워터마크가 들어간 이미지에서 **디코드(복원)** 할 수 있습니다."
    )

    with gr.Tabs():
        # ------------------- Embed 탭 -------------------
        with gr.Tab("Embed (워터마크 삽입)"):
            with gr.Row():
                with gr.Column(scale=2):
                    in_files = gr.File(
                        label="이미지 업로드 (PNG/JPG, 여러 장 가능)",
                        file_count="multiple",
                        file_types=["image"],
                    )

                    # --- 편집 옵션 ---
                    preset = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="원본 유지",
                        label="비율/해상도 프리셋",
                    )
                    with gr.Row():
                        custom_w = gr.Number(value=None, label="맞춤 가로(px)", precision=0)
                        custom_h = gr.Number(value=None, label="맞춤 세로(px)", precision=0)

                    fit = gr.Radio(
                        choices=list(FIT_STRATEGIES),
                        value="crop",
                        label="비율 맞춤 방식",
                    )
                    with gr.Row():
                        max_w = gr.Number(value=1920, label="최대 가로(px)", precision=0)
                        max_h = gr.Number(value=1920, label="최대 세로(px)", precision=0)
                    sharpen = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="샤프닝 강도")

                    # --- 워터마크 옵션 ---
                    use_uuid = gr.Checkbox(value=True, label="파일별 UUID 자동 생성")
                    clean_prev = gr.Checkbox(value=True, label="실행 전 이전 결과 삭제")
                    custom_payload = gr.Textbox(
                        label="사용자 지정 페이로드(모든 파일 동일 적용, 비우면 UUID 자동 생성)",
                        placeholder="예) OPENMARK-UUID-2025-....",
                    )

                    run_btn = gr.Button("워터마크 적용", variant="primary")

                with gr.Column(scale=3):
                    out_zip = gr.File(label="결과 ZIP 다운로드")
                    out_log = gr.Textbox(
                        label="워터마크 매핑 로그",
                        lines=12,
                        interactive=False,
                    )
                    gallery = gr.Gallery(
                        label="미리보기",
                        show_label=True,
                        height=340,
                        columns=4,
                    )

            # 핸들러 연결: do_embed_with_edit(files, use_uuid, custom_payload, clean_prev,
            #                                 preset_name, cw, ch, fit_mode, mw, mh, sharp)
            run_btn.click(
                fn=do_embed_with_edit,
                inputs=[
                    in_files, use_uuid, custom_payload, clean_prev,
                    preset, custom_w, custom_h, fit, max_w, max_h, sharpen
                ],
                outputs=[out_zip, out_log, gallery],
            )

        # ------------------- Decode 탭 -------------------
        with gr.Tab("Decode (워터마크 복원)"):
            with gr.Row():
                dec_in = gr.File(
                    label="워터마크가 삽입된 이미지 한 장 선택",
                    file_count="single",
                    file_types=["image"],
                )
                dec_btn = gr.Button("복원 실행", variant="primary")
            dec_out = gr.Textbox(
                label="복원된 페이로드",
                lines=3,
                interactive=False,
            )

            # 핸들러 연결: do_decode(file)
            dec_btn.click(fn=do_decode, inputs=[dec_in], outputs=[dec_out])

    gr.Markdown(
        "💡 **Tip**\n"
        "- Embed 탭 ZIP에는 워터마크 이미지와 `watermarks.txt` 로그가 포함됩니다.\n"
        "- Decode에서 결과가 비어 있으면 JPEG 재압축/크기변경 등의 영향일 수 있어요."
    )



if __name__ == "__main__":
    iface.launch()