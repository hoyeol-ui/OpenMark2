from __future__ import annotations
import gradio as gr
from gradio.themes import Soft
from .pipeline import run_one_shot, do_decode
from watermark.configs import human_size_policy

KOREAN_FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
* { font-family: 'Noto Sans KR', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Apple SD Gothic Neo', 'Noto Sans KR', '맑은 고딕', 'Malgun Gothic', sans-serif; }
:root { --brand-blue: #2563eb; }
.gradio-container { color: #111827; }
.footer { opacity: .9; }
"""

def launch(share: bool = False):
    theme = Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="OpenMark — 보이지 않는 보호 + 방해 (원샷)", theme=theme, css=KOREAN_FONT_CSS) as demo:
        gr.Markdown(
            "## OpenMark — **원샷 보호(워터마크+방해)**  \n"
            "이미지를 한 장 업로드하고 **원샷 적용**을 누르면 끝!  \n"
            f"**이미지 크기 가이드:** {human_size_policy()}"
        )

        with gr.Row():
            with gr.Column(scale=2):
                in_file = gr.File(label="이미지 업로드 (PNG/JPG)", file_count="single", file_types=["image"])
                btn_run = gr.Button("원샷 적용", variant="primary")
            with gr.Column(scale=3):
                out_img = gr.Image(label="결과 이미지", interactive=False)
                out_uuid, uuid_text = gr.File(label="UUID 다운로드 (.txt)"), gr.Textbox(label="UUID (기록용)", interactive=False)
                diag = gr.Gallery(label="Residual Heatmap / FFT / Overlay", columns=3, height=320)
                info = gr.Markdown("")
        btn_run.click(run_one_shot, [in_file], [out_img, out_uuid, diag, info, uuid_text])

        gr.Markdown("---\n### 디코드 확인 (선택) — 결과 이미지에서 UUID 복원")
        dec_in = gr.File(label="OpenMark 보호가 적용된 이미지 선택", file_count="single", file_types=["image"])
        gr.Button("디코드 실행", variant="secondary").click(do_decode, [dec_in], [gr.Textbox(label="복원된 UUID", interactive=False)])

    demo.launch(share=share)