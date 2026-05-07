"""UART/Serial 기반 임베디드 알림 송신 모듈 — M5에서 구현 예정."""

# TODO(M5): pyserial을 이용한 UART JSON 라인 프로토콜 구현
# TODO(M5): heartbeat 5초 주기 전송
# TODO(M5): seq 단조 증가, 재전송 큐 구현
# 페이로드 형식: {"ts": 1714000000.123, "event": "danger", "class": "...",
#                 "score": 0.87, "duration_ms": 960, "seq": 142}


class UARTSender:
    """M5 구현 플레이스홀더."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        raise NotImplementedError("UARTSender는 M5 마일스톤에서 구현됩니다.")

    def send_event(self, payload: dict) -> None:
        raise NotImplementedError

    def send_heartbeat(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
