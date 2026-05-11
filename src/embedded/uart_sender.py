"""UART/Serial 기반 임베디드 알림 송신 모듈 — M5에서 구현 예정."""

# TODO(M5): pyserial을 이용한 UART JSON 라인 프로토콜 구현
# TODO(M5): heartbeat 5초 주기 전송
# TODO(M5): seq 단조 증가, 재전송 큐 구현
# 페이로드 형식: {"ts": 1714000000.123, "event": "danger", "class": "...",
#                 "score": 0.87, "duration_ms": 960, "seq": 142}


class UARTSender:
    """M5 구현 플레이스홀더."""

    # 현재는 인터페이스만 정의된 스텁이다. 호출 시 NotImplementedError로 명확히 알린다.

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        # 시리얼 포트 이름(예: "COM3", "/dev/ttyUSB0")과 통신 속도 지정.
        raise NotImplementedError("UARTSender는 M5 마일스톤에서 구현됩니다.")

    def send_event(self, payload: dict) -> None:
        # 위험 이벤트 JSON 라인을 UART로 송신.
        raise NotImplementedError

    def send_heartbeat(self) -> None:
        # 연결 상태 확인용 주기적 heartbeat 패킷 송신.
        raise NotImplementedError

    def close(self) -> None:
        # 시리얼 포트 정리.
        raise NotImplementedError
