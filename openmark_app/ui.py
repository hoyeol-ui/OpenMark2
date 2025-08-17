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

/* --- 설명 태그/문단 스타일 --- */
.tag {
  display:inline-block;
  background-color: var(--brand-blue);
  color:#fff;
  font-weight:700;
  font-size:13px;
  padding:4px 10px;
  border-radius:8px;
  letter-spacing:.2px;
}
.desc {
  display:block;
  margin:6px 0 14px;
  line-height:1.45;
}
#guide-box { max-width: 420px; }  /* 줄 길이 제한 */
"""

def launch(share: bool = False):
    theme = Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="OpenMark — 보이지 않는 보호 + 방해 (원샷)", theme=theme, css=KOREAN_FONT_CSS) as demo:
        gr.Markdown(
            "## OpenMark — **원샷(워터마크+방해)으로 내 컨텐츠를 지킨다.**\n"
            "내 이미지를 한 장 업로드하고, **원샷 적용**을 누르면 끝!<br>"
            f"**이미지 크기 가이드:** {human_size_policy()}"
        )

        with gr.Row():
            # --- 좌측: 업로드/버튼/설명 ---
            with gr.Column(scale=2):
                in_file = gr.File(label="이미지 업로드 (PNG/JPG)", file_count="single", file_types=["image"])
                btn_run = gr.Button("원샷 적용", variant="primary")

                # ✅ 버튼 바로 아래: 키워드 박스 + 짧은 설명(줄바꿈)
                gr.Markdown(
                    """
                            <span class="tag">UUID</span>
                            <span class="desc">
                            이미지에 숨겨진 고유 식별번호.<br>
                            소유자 인증·추적에 쓰이며,<br>
                            결과 이미지에서 숨겨저 있습니다.
                            </span>
                            
                            <span class="tag">Residual Heatmap</span>
                            <span class="desc">
                            워터마크/방해로 생긴<br>
                            <strong>미세한 차이</strong>를 색으로 표시.<br>
                            사람 눈엔 거의 안 보이지만<br>
                            컴퓨터는 쉽게 감지.
                            </span>
                            
                            <span class="tag">FFT</span>
                            <span class="desc">
                            이미지를 <strong>주파수</strong>로 본 모습.<br>
                            워터마크가 숨어 있는 규칙적 패턴을 확인.
                            </span>
                            
                            <span class="tag">Overlay</span>
                            <span class="desc">
                            인공지능이 학습을 할때 인식하는 모습<br>
                            사람 눈에는 안보이지만,<br>
                            인공지능은 이 모습으로 내 컨텐츠를 받아들입니다.<br>
                            (인공지능이 보는 내 이미지 모습)
                            </span>
                    """,
                    elem_id="guide-box"
                )

            # --- 우측: 결과/진단 ---
            with gr.Column(scale=3):
                out_img = gr.Image(label="결과 이미지", interactive=False)
                with gr.Row():
                    out_uuid = gr.File(label="UUID 다운로드 (.txt)")
                    uuid_text = gr.Textbox(label="UUID (기록용)", interactive=False)
                diag = gr.Gallery(label="Residual Heatmap / FFT / Overlay", columns=3, height=320)
                info = gr.Markdown("")  # 처리 후 동적 안내(PSNR 등)

        # 원샷 실행 바인딩
        btn_run.click(run_one_shot, [in_file], [out_img, out_uuid, diag, info, uuid_text])

        gr.Markdown("---\n### 디코드 확인 (선택) — 결과 이미지에서 UUID 복원")
        dec_in = gr.File(label="OpenMark 보호가 적용된 이미지 선택", file_count="single", file_types=["image"])
        dec_out = gr.Textbox(label="복원된 UUID", interactive=False)
        gr.Button("디코드 실행", variant="secondary").click(do_decode, [dec_in], [dec_out])

    demo.launch(share=share)