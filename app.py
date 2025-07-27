import os
import uuid
import shutil
import gradio as gr

# 원본 이미지가 위치한 폴더
UPLOAD_DIR = "uploads"
# 워터마크 이미지를 저장할 폴더
OUTPUT_DIR = "outputs"

# 우리가 만든 embedder 모듈에서 함수만 import
from watermark.embedder import embed_watermark

def process_images() -> str:
    """
    uploads 폴더 안의 모든 이미지 파일에 워터마크를 삽입하고,
    결과물(zip 파일 경로)을 반환합니다.
    """
    # outputs 디렉토리가 없다면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mappings = []      # (파일명, UUID)를 저장할 리스트
    output_paths = []  # 생성된 워터마크 이미지 경로 리스트

    # uploads 폴더 내 모든 파일 순회
    for original_name in os.listdir(UPLOAD_DIR):
        # 이미지 파일 확장자만 처리
        if not original_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(UPLOAD_DIR, original_name)
        name, ext = os.path.splitext(original_name)

        # UUID 생성
        watermark_uuid = uuid.uuid4().hex
        # 출력 파일명/경로
        output_filename = f"{name}_watermarked{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # 워터마크 삽입
        embed_watermark(input_path, output_path, watermark_uuid)

        # 매핑 및 경로 저장
        mappings.append((output_filename, watermark_uuid))
        output_paths.append(output_path)

    # 워터마크 로그 파일 생성
    log_path = os.path.join(OUTPUT_DIR, "watermarks.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        for fname, uuid_str in mappings:
            log_file.write(f"{fname} -> {uuid_str}\n")
    output_paths.append(log_path)

    # 결과물 ZIP으로 묶기
    zip_base = os.path.join(OUTPUT_DIR, "watermarked_results")
    shutil.make_archive(zip_base, 'zip', OUTPUT_DIR)
    zip_path = zip_base + ".zip"

    return zip_path

# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=process_images,
    inputs=[],  # uploads 폴더를 직접 참조하므로 파일 입력 UI는 비워둠
    outputs=gr.File(label="워터마크 적용 결과 (ZIP 파일)"),
    title="보이지 않는 워터마크 적용기",
    description=(
        "uploads 폴더 안의 이미지에 UUID 워터마크를 삽입합니다.\n"
        "실행 버튼을 클릭하면 ZIP 파일을 다운로드할 수 있습니다."
    )
)

if __name__ == "__main__":
    iface.launch()