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

HELP_TEXT = """
💡 **Tip**: 지키고 싶은 이미지를 한 장씩 업로드 해주세요. 
"""

def launch():
    theme = Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="OpenMark — 보이지 않는 보호 + 방해 (원샷)", theme=theme, css=KOREAN_FONT_CSS) as demo:
        gr.Markdown("## OpenMark — **원샷 보호(워터마크+방해)**")
        with gr.Row():
            with gr.Column(scale=2):
                in_file = gr.File(label="이미지 업로드 (PNG/JPG)", file_count="single", file_types=["image"])
                btn_run = gr.Button("원샷 적용", variant="primary")
                gr.Markdown(f"**이미지 크기 가이드** — {human_size_policy()}")
            with gr.Column(scale=3):
                out_img = gr.Image(label="결과 이미지", interactive=False)
                with gr.Row():
                    out_uuid = gr.File(label="UUID 다운로드 (.txt)")
                    uuid_text = gr.Textbox(label="UUID (기록용)", interactive=False)
                diag = gr.Gallery(label="Residual Heatmap / FFT / Overlay", columns=3, height=320)
                info = gr.Markdown("")

        btn_run.click(fn=run_one_shot, inputs=[in_file],
                      outputs=[out_img, out_uuid, diag, info, uuid_text])

        gr.Markdown("---")
        gr.Markdown("### 디코드 확인 (선택) — 결과 이미지에서 UUID 복원")
        dec_in  = gr.File(label="OpenMark 보호가 적용된 이미지 선택", file_count="single", file_types=["image"])
        dec_btn = gr.Button("디코드 실행", variant="secondary")
        dec_out = gr.Textbox(label="복원된 UUID", interactive=False)
        dec_btn.click(fn=do_decode, inputs=[dec_in], outputs=[dec_out])

    demo.launch()