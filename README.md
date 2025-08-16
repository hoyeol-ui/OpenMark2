

<h1 align="center">OpenMark — Protect + Disrupt (원샷)</h1>

<p align="center">
  <strong>눈에 안 보이는 워터마크(UUID) + 학습 방해(Disrupt)</strong>를 버튼 한 번으로.<br/>
  원본과 육안상 거의 동일한 품질을 유지하면서, 모델 학습 정렬을 흔들어줍니다.
</p>



---

## ✨ 특징

- **원샷 처리**: 이미지 업로드 → *Protect(워터마크)* → *Disrupt* → 결과/UUID/진단 출력
- **워터마크 복원 보장**: 삽입 직후 내부 디코딩으로 검증하고, 필요시 자동 보정
- **육안 품질 유지**: 평균 PSNR ≈ 44dB(기본값), 고주파 노이즈 억제
- **학습 방해(Disrupt)**: CLIP 정렬을 흔드는 경량 EOT-PGD, JPEG/리사이즈에 일부 강건
- **진단 시각화**: Residual Heatmap / FFT / Overlay로 “모델이 본 변화”를 확인

<p align="center">
  <img src="./docs/20250816%20result.png" alt="demo result" width="85%" />
</p>

---

## 📦 설치

```bash
# Python 3.10+ 권장, 가상환경 추천
pip install -r requirements.txt

# (선택) GPU/MPS 환경이 있다면 torch 설치 가이드에 맞춰 설치
# Mac(MPS) / CUDA 환경 모두 자동 감지