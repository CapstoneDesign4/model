# ESP32 2대 - Jetson Orin 위험음 분류 시스템 개발 기획서

> 작성일: 2026-05-19  
> 기반 문서: `docs/esp32-jetson-orin-development-analysis.md`  
> 목표: ESP32 2대가 각자 마이크 음성을 수집해 Jetson Orin으로 보내고, Jetson이 위험음을 분류한 뒤 ESP32로 결과를 반환해 진동 모터를 제어하는 MVP를 개발한다.

---

## 1. 개발 목표

ESP32-A와 ESP32-B가 각각 I2S MEMS 마이크로 주변 음성을 수집한다. 각 ESP32는 16kHz / 16bit / mono PCM 오디오를 0.48초 청크 단위로 Jetson Orin에 전송한다. Jetson Orin은 두 장치의 오디오 스트림을 `device_id`로 분리해 YAMNet 기반 위험음 분류를 수행하고, 위험음 이벤트가 발생하면 해당 ESP32 또는 양쪽 ESP32에 JSON 결과를 보낸다. ESP32는 수신한 `severity` 또는 `vibration_pattern`에 따라 진동 모터를 구동한다.

### MVP 성공 기준

| 지표 | 목표 |
|---|---|
| ESP32 동시 연결 | ESP32 2대 동시 오디오 송신 |
| 오디오 포맷 | 16kHz, int16 little-endian, mono |
| 분석 윈도우 | 0.96초 window / 0.48초 hop |
| 위험음 분류 | 기존 YAMNet + whitelist + debounce/cooldown 파이프라인 재사용 |
| 결과 반환 | Jetson -> ESP32 JSON 이벤트 송신 |
| 진동 제어 | `high`, `medium`, `low` 3단계 패턴 동작 |
| end-to-end latency | 평균 1초 이내 목표 |
| 안정성 | ESP32 2대 10분 이상 연속 스트리밍 |

---

## 2. 시스템 범위

### 2.1 포함 범위

| 영역 | 개발 범위 |
|---|---|
| ESP32 펌웨어 | I2S 마이크 캡처, PCM 청크 생성, UDP 송신, 결과 JSON 수신, 진동 PWM 제어 |
| Jetson 수신부 | UDP packet 수신, header 검증, 장치별 ring buffer, 0.96초 윈도우 생성 |
| Jetson 추론부 | YAMNet 추론, 위험 클래스 추출, threshold, debounce, cooldown 적용 |
| Jetson 송신부 | 위험 이벤트 JSON 송신, heartbeat 송신 |
| 검증 도구 | WAV -> UDP PCM sender, latency 측정, packet loss 로그 |
| 문서화 | 실행 절차, 패킷 규격, 통합 테스트 절차 |

### 2.2 MVP 비범위

| 항목 | 제외 이유 |
|---|---|
| ESP32 내부 위험음 추론 | ESP32 연산/메모리로 YAMNet 실시간 추론은 부적합 |
| Opus 등 오디오 압축 | Wi-Fi 대역폭은 충분하고 ESP32 구현 복잡도 증가 |
| 모바일 앱/웹 대시보드 | MVP 핵심 흐름 이후 확장 |
| 다수 사용자 1:N broadcast 정책 고도화 | 우선 ESP32 2대 고정 구성 검증 |
| TFLite/TensorRT 최적화 | Orin에서 기본 YAMNet 추론 성능 측정 후 판단 |

---

## 3. 전체 아키텍처

```text
[ESP32-A]
  I2S Mic -> PCM chunk -> UDP packet(device_id=1)
                                             \
                                              -> [Jetson Orin]
                                             /      UDP receiver
[ESP32-B]                                  /        per-device ring buffer
  I2S Mic -> PCM chunk -> UDP packet(device_id=2)   YAMNet inference
                                                    danger trigger
                                                    JSON sender
                                                       |
                         +-----------------------------+-----------------------------+
                         |                                                           |
                  result JSON                                                   result JSON
                         |                                                           |
                    [ESP32-A]                                                   [ESP32-B]
                  vibration PWM                                               vibration PWM
```

### 3.1 통신 규격

#### ESP32 -> Jetson: UDP PCM

| 항목 | 값 |
|---|---|
| 프로토콜 | UDP |
| 헤더 엔디언 | network byte order, big-endian |
| 헤더 struct | `!HBBIIHH` |
| 샘플레이트 | 16,000 Hz |
| 샘플 포맷 | signed int16 little-endian |
| 채널 | mono |
| 청크 길이 | 0.48초 |
| 청크 샘플 수 | 7,680 |
| payload 크기 | 15,360 bytes |
| ESP32 2대 총 대역폭 | 약 512 kbps + overhead |

권장 UDP header:

```text
magic        uint16  0xA501
version      uint8   1
device_id    uint8   1 or 2
seq          uint32  장치별 단조 증가
timestamp_ms uint32  ESP32 부팅 후 ms
payload_len  uint16  15360
flags        uint16  reserved
payload      int16[7680]
```

#### Jetson -> ESP32: JSON 이벤트

위험 이벤트:

```json
{
  "event": "danger",
  "seq": 142,
  "target_device_id": 1,
  "source_device_id": 1,
  "ts": 1779132000.123,
  "class": "glass_shatter",
  "score": 0.87,
  "severity": "high",
  "vibration_pattern": "high_continuous",
  "duration_ms": 960
}
```

heartbeat:

```json
{
  "event": "heartbeat",
  "seq": 143,
  "ts": 1779132005.000
}
```

---

## 4. 개발 마일스톤

### M0. Jetson Orin 실행 환경 고정

목적: 현재 저장소의 로컬 YAMNet 파이프라인이 Jetson Orin에서 정상 동작하는지 먼저 확정한다.

| 항목 | 내용 |
|---|---|
| 구현/작업 | Python/TensorFlow/TensorFlow Hub 환경 구성, `TFHUB_CACHE_DIR` 고정, 샘플 WAV 추론 |
| 산출물 | Jetson 환경 세팅 메모, 정상 추론 로그 |
| 완료 기준 | `scripts/verify_inference.py` 성공, `python -m src.cli --input data/sample/...wav --verbose` 성공 |
| 리스크 | TF-Hub 캐시 불완전, Jetson용 TensorFlow 설치 문제 |

### M1. Jetson UDP 네트워크 입력 파이프라인

목적: ESP32 없이도 UDP PCM 입력을 받아 기존 YAMNet 파이프라인에 연결한다.

| 항목 | 내용 |
|---|---|
| 구현/작업 | `src/audio_io/network_stream.py` 추가, UDP header 파싱, int16 -> float32 변환, 장치별 ring buffer |
| CLI 변경 | `--input network`, `--listen-port`, `--device-count` 추가 |
| 산출물 | 네트워크 오디오 입력 모듈 |
| 완료 기준 | UDP로 받은 WAV PCM이 기존 파일 입력과 유사한 위험 클래스 score 출력 |
| 테스트 | 패킷 손실, 중복, 순서 역전 로그 확인 |

### M2. PC 기반 UDP sender 검증 도구

목적: ESP32 펌웨어 개발 전에 PC에서 ESP32 2대 상황을 시뮬레이션한다.

| 항목 | 내용 |
|---|---|
| 구현/작업 | `scripts/send_pcm_udp.py` 추가, WAV 파일을 0.48초 PCM packet으로 전송 |
| 산출물 | UDP 테스트 sender |
| 완료 기준 | `device_id=1`, `device_id=2` 동시 전송 시 Jetson이 장치별 score를 분리 출력 |
| 검증 명령 예 | `python scripts/send_pcm_udp.py --file ... --device-id 1 --host <jetson-ip> --port 5005` |

### M3. ESP32 1대 오디오 송신 PoC

목적: 실제 I2S 마이크 음성을 Jetson으로 보내는 최소 펌웨어를 만든다.

| 항목 | 내용 |
|---|---|
| 구현/작업 | I2S MEMS 마이크 캡처, 16kHz int16 mono 변환, UDP packet 생성, `device_id=1` 송신 |
| 권장 하드웨어 | ESP32, INMP441 또는 SPH0645, 안정적인 USB 전원 |
| 산출물 | ESP32 audio node 1대 펌웨어 |
| 완료 기준 | Jetson에서 ESP32 실시간 오디오 score 출력, 위험음 샘플 재생 시 score 상승 |
| 검증 | 조용한 환경 false positive 확인, packet loss 확인 |

### M4. ESP32 2대 동시 송신 통합

목적: 두 ESP32의 오디오 스트림을 동시에 안정적으로 처리한다.

| 항목 | 내용 |
|---|---|
| 구현/작업 | ESP32-A/B `device_id` 분리, 두 장치 동시 송신, Jetson 장치별 stream state 유지 |
| 산출물 | ESP32 2대 송신 펌웨어, Jetson 장치별 로그 |
| 완료 기준 | 10분 이상 동시 스트리밍, 장치별 score/event 분리, 한 장치 손실이 다른 장치에 영향 없음 |
| 검증 | ESP32-A에는 siren, ESP32-B에는 glass_shatter를 보내 장치별 이벤트 확인 |

### M5. Jetson -> ESP32 결과 송신

목적: Jetson이 위험음 분류 결과를 ESP32로 되돌려 보낸다.

| 항목 | 내용 |
|---|---|
| 구현/작업 | `src/embedded/wifi_sender.py` 추가, JSON 이벤트 송신, heartbeat 송신, seq 증가 |
| ESP32 작업 | JSON 수신 서버, 중복 seq 제거, heartbeat timeout 처리 |
| 산출물 | Jetson Wi-Fi sender, ESP32 result receiver |
| 완료 기준 | Jetson에서 임의 danger JSON을 보내면 ESP32가 이벤트를 파싱하고 상태를 갱신 |
| 기본 정책 | MVP는 위험음이 감지된 source ESP32에만 결과 송신 |

### M6. 진동 모터 제어

목적: ESP32가 Jetson 결과에 따라 진동 모터를 구동한다.

| severity | PWM duty | 패턴 |
|---|---:|---|
| `high` | 100% | 1.5초 연속 |
| `medium` | 70% | 0.3초 ON x 3 |
| `low` | 40% | 0.2초 단발 |

| 항목 | 내용 |
|---|---|
| 구현/작업 | PWM 출력, MOSFET/트랜지스터 드라이버 회로, 보호 다이오드 적용 |
| 산출물 | 진동 제어 펌웨어 및 회로 연결표 |
| 완료 기준 | `high/medium/low` JSON에 따라 서로 다른 진동 패턴 동작 |
| 주의 | ESP32 GPIO로 모터를 직접 구동하지 않는다 |

### M7. End-to-End 통합 검증

목적: 실제 목표 흐름 전체를 측정한다.

| 항목 | 내용 |
|---|---|
| 검증 흐름 | 위험음 재생 -> ESP32 마이크 수집 -> UDP 송신 -> Jetson 분류 -> JSON 반환 -> ESP32 진동 |
| 측정 지표 | 평균 latency, 95p latency, packet loss, false alarm, missed detection, CPU/GPU/RAM |
| 산출물 | 통합 테스트 결과표, 로그 파일, 문제 목록 |
| 완료 기준 | 평균 latency 1초 이내, ESP32 2대 10분 이상 안정 동작, 위험음 재생 시 진동 발생 |

---

## 5. 담당 영역 분리

| 영역 | 담당 구현 |
|---|---|
| ESP32 펌웨어 | I2S 설정, DMA/audio buffer, UDP packet 생성, Wi-Fi 재연결, JSON 수신, heartbeat 감시, PWM 진동 제어 |
| Jetson 네트워크 | UDP socket, packet header 검증, 장치별 ring buffer, 손실/중복/역전 packet 로그 |
| Jetson 모델 | 기존 `YAMNetWrapper`, `DangerFilter`, `Trigger` 재사용, network input 연결, 이벤트 생성 |
| Jetson 송신 | `wifi_sender.py`, JSON schema, heartbeat, target device routing |
| 모델 품질 | threshold 재측정, noise suppression A/B, 위험/negative 데이터셋 확장 |
| 검증 | PC UDP sender, latency 측정, 장시간 스트리밍 테스트, 테스트 결과 문서화 |
| 하드웨어 | 마이크 배선, 모터 드라이버, 전원 안정화, 진동 모터 보호 회로 |

---

## 6. 우선순위 작업 계획

### P0: 지금 바로 착수

1. Jetson Orin에서 현재 YAMNet 로컬 추론 성공
2. TF-Hub 캐시 문제 해결 및 `TFHUB_CACHE_DIR` 고정
3. UDP packet header 최종 확정
4. `network_stream.py` 설계
5. `send_pcm_udp.py` 테스트 도구 설계

### P1: 네트워크 입력 MVP

1. `network_stream.py` 구현
2. `src/cli.py --input network` 추가
3. PC sender로 `device_id=1/2` 동시 송신 테스트
4. 장치별 ring buffer와 trigger 상태 분리
5. JSONL 로그에 `device_id`, `seq`, packet loss 정보 추가

### P2: 실제 ESP32 통합

1. ESP32 1대 I2S 마이크 송신 PoC
2. ESP32 2대 동시 송신
3. Jetson -> ESP32 결과 송신
4. ESP32 진동 제어
5. end-to-end latency 측정

### P3: 품질 개선

1. 실제 환경 negative 1시간 이상 수집
2. 클래스별 threshold 재보정
3. noise suppression 구현 및 A/B 검증
4. Orin 추론 시간 벤치마크
5. 필요 시 TFLite/ONNX/TensorRT 검토

---

## 7. 검증 계획

### 7.1 단위 검증

| 대상 | 검증 방법 |
|---|---|
| UDP header parser | 정상/잘못된 magic/version/payload_len 테스트 |
| int16 -> float32 변환 | 범위가 `[-1.0, 1.0]`인지 확인 |
| 장치별 ring buffer | device 1/2 packet이 섞이지 않는지 확인 |
| Trigger 상태 | 장치별 debounce/cooldown이 독립인지 확인 |
| JSON sender | danger/heartbeat schema 확인 |

### 7.2 통합 검증

| 단계 | 검증 방법 |
|---|---|
| PC sender -> Jetson | WAV를 UDP로 보내 기존 파일 입력 score와 비교 |
| ESP32 1대 -> Jetson | 실제 마이크 입력 score 확인 |
| ESP32 2대 -> Jetson | 두 장치 동시 송신 및 분리 로그 확인 |
| Jetson -> ESP32 | 임의 danger JSON으로 진동 패턴 확인 |
| 전체 흐름 | 위험음 재생부터 진동까지 latency 측정 |

### 7.3 현장 검증

| 환경 | 확인 내용 |
|---|---|
| 조용한 실내 | false positive 여부 |
| 소음 있는 실내 | TV/대화/생활소음에서 오탐 여부 |
| 위험음 재생 | siren, glass_shatter, baby_cry, vehicle_horn 감지 여부 |
| 두 ESP32 거리 차이 | 가까운 장치와 먼 장치의 score 차이 |
| 장시간 테스트 | 10분, 30분, 1시간 스트리밍 안정성 |

---

## 8. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| Jetson YAMNet 로딩 실패 | 개발 시작 불가 | 캐시 삭제 후 재다운로드, `TFHUB_CACHE_DIR` 고정 |
| ESP32 I2S 캡처 품질 불량 | 위험음 score 불안정 | WAV dump 기능 추가, 파형/볼륨 확인, 배선 재점검 |
| UDP packet 손실 | 분석 윈도우 품질 저하 | seq 로그, 오래된 packet drop, 실시간성 우선 |
| 두 ESP32 packet 혼선 | 잘못된 장치에 진동 | `device_id` 필수, 장치별 ring buffer와 trigger 분리 |
| debounce 지연 | 1초 목표 초과 | 단발 위험음은 K/N 완화 또는 `--no-debounce` 비교 |
| 데이터셋 부족 | 실제 환경 성능 불확실 | 자체 위험음/negative 수집 후 threshold 재보정 |
| 진동 모터 전류 과부하 | ESP32 손상 | MOSFET/트랜지스터 드라이버와 보호 다이오드 사용 |
| Wi-Fi 재연결 문제 | 스트리밍 중단 | ESP32 재연결 루프, Jetson timeout 상태 표시 |

---

## 9. 산출물 목록

| 산출물 | 경로/형태 |
|---|---|
| Jetson UDP 수신 모듈 | `src/audio_io/network_stream.py` |
| Jetson Wi-Fi 결과 송신 모듈 | `src/embedded/wifi_sender.py` |
| CLI network 모드 | `src/cli.py` |
| UDP PCM 테스트 sender | `scripts/send_pcm_udp.py` |
| latency 측정 도구 | `scripts/measure_latency.py` |
| ESP32 오디오 노드 펌웨어 | `firmware/esp32_audio_node/` |
| ESP32 진동 수신/제어 펌웨어 | `firmware/esp32_audio_node/` |
| 패킷 규격 문서 | 본 문서 및 후속 protocol 문서 |
| 통합 테스트 결과 | `experiments/` 또는 `output/` |

---

## 10. 바로 다음 작업 상세

가장 먼저 구현할 것은 ESP32 펌웨어가 아니라 Jetson의 network input 경로다. 이 순서가 좋은 이유는 PC sender로 ESP32 2대 상황을 먼저 재현할 수 있고, 이후 ESP32는 이미 검증된 packet 규격에 맞춰 오디오만 보내면 되기 때문이다.

첫 번째 개발 목표:

```powershell
python -m src.cli --input network --listen-port 5005 --device-count 2 --verbose
```

두 번째 개발 목표:

```powershell
python scripts\send_pcm_udp.py --file data\sample\siren\1-31482-A-42.wav --device-id 1 --host <jetson-ip> --port 5005
python scripts\send_pcm_udp.py --file data\sample\glass_shatter\1-20133-A-39.wav --device-id 2 --host <jetson-ip> --port 5005
```

완료되면 다음 판단이 가능해진다.

- 현재 YAMNet 파이프라인이 네트워크 입력에서도 그대로 동작하는지
- ESP32 2대의 stream state 분리가 충분한지
- 실제 ESP32 펌웨어가 맞춰야 할 packet 규격이 안정적인지
- end-to-end MVP 구현을 바로 시작할 수 있는지
