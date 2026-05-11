"""노이즈 억제 전처리 모듈 — M2에서 WebRTC NS 구현 예정."""

# TODO(M2): webrtc-noise-gain 또는 noisereduce 기반 WebRTC NS 통합
# TODO(M2): aggressiveness 파라미터 설정 파일 연동
# TODO(M2): NS on/off A/B 비교 실험 스크립트 작성
# 참조: docs/development-plan.md §4.2


# M1 단계에서는 외부에서 suppress()를 호출해도 입력을 그대로 돌려준다 (no-op).
# M2 이후 WebRTC NS가 통합되면 이 함수가 실제 노이즈 억제를 수행한다.
def suppress(audio_16k_mono, aggressiveness: int = 1):
    """노이즈 억제를 수행한다 (M2 구현 후 동작).

    Args:
        audio_16k_mono: float32 ndarray, 16kHz mono.
        aggressiveness: 0~3, 높을수록 강한 억제 (WebRTC NS 기준).

    Returns:
        억제된 float32 ndarray (M2 이전에는 입력을 그대로 반환).
    """
    # M1: 패스스루 — 노이즈 처리 없이 그대로 반환
    return audio_16k_mono
