import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder

import uuid


# 워터마크 삽입 함수
def embed_watermark(input_path: str, output_path: str, watermark_uuid: str) -> None:
    """
    주어진 이미지 파일(input_path)에 watermark_uuid 문자열 워터마크를 삽입하여
    output_path 위치에 저장합니다. 워터마크는 눈에 보이지 않으며, watermark_uuid는 별도 저장/관리됩니다.
    """
    # 이미지 읽기 (BGR 형식으로 로드)
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # 워터마크 인코더 초기화 및 설정
    encoder = WatermarkEncoder()
    # 워터마크를 bytes 형태로 설정 (UTF-8 인코딩)
    encoder.set_watermark('bytes', watermark_uuid.encode('utf-8'))

    # DWT-DCT 알고리즘으로 워터마크 삽입 수행
    watermarked_image = encoder.encode(image, 'dwtDct')

    # 워터마크가 삽입된 이미지 파일 저장
    cv2.imwrite(output_path, watermarked_image)


# (선택) 워터마크 추출 함수 - 디버그 또는 향후 검증용
def extract_watermark(image_path: str, watermark_length: int) -> str:
    """
    워터마크가 삽입된 이미지에서 숨겨진 워터마크를 추출하여 문자열로 반환합니다.
    watermark_length는 숨긴 메시지 비트 길이(또는 문자의 총 비트 수)를 지정해야 합니다.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found for decoding: {image_path}")
    # 디코더 초기화 ('bytes' 모드, 길이는 비트 단위)
    decoder = WatermarkDecoder('bytes', watermark_length)
    extracted_bytes = decoder.decode(image, 'dwtDct')
    try:
        return extracted_bytes.decode('utf-8')
    except Exception as e:
        # UTF-8 디코딩 실패 시 바이너리 데이터 그대로 문자열 변환
        return str(extracted_bytes)