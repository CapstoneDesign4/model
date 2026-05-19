# ESP32 2대 - Jetson Orin 위험음 분류 개발 분석

> 작성일: 2026-05-19  
> 목적: ESP32 2대가 마이크 음성을 Jetson Orin으로 전송하고, Jetson의 위험음 분류 결과를 다시 ESP32로 보내 진동 모터를 제어하는 구조를 현재 저장소 상태 기준으로 분석한다.

---

## 1. 목표 시스템 요약

목표 구조는 다음과 같다.

```text
ESP32-A + I2S 마이크 ──┐
                       ├─ Wi-Fi UDP PCM 오디오 ──► Jetson Orin ── 위험음 분류
ESP32-B + I2S 마이크 ──┘                                  │
                                                          ▼
                           Wi-Fi JSON 결과 ◄──── ESP32-A / ESP32-B
                                                          │
                                                          ▼
                                               진동 모터 PWM 제어
```

핵심 역할 분담은 아래처럼 잡는 것이 현실적이다.

| 장치 | 담당 |
|---|---|
| ESP32 2대 | I2S MEMS 마이크 캡처, 16kHz/16bit mono PCM 청크 전송, Jetson 결과 수신, 진동 모터 구동 |
| Jetson Orin | 두 ESP32 스트림 수신, 0.96초 윈도우 조립, YAMNet 위험음 분류, debounce/cooldown, 결과 JSON 송신 |

MVP에서는 두 ESP32의 오디오를 독립 스트림으로 처리한다. 즉, `device_id=esp32_a`, `device_id=esp32_b`를 구분하고 각 장치별로 별도 ring buffer와 trigger 상태를 유지한다. 이후 필요하면 두 장치 중 하나라도 위험을 감지하면 양쪽 모두 진동시키는 broadcast 정책이나, 두 장치 score를 합치는 fusion 정책을 추가한다.

---

## 2. 현재 구현 상태

현재 저장소는 Jetson/PC 로컬에서 WAV 파일 또는 마이크 입력을 분석하는 모델 파이프라인 중심으로 구현되어 있다. ESP32 네트워크 연동은 아직 구현 전이다.

### 2.1 구현된 것

| 영역 | 상태 | 근거 파일 |
|---|---|---|
| CLI 실행 진입점 | `--input mic` 또는 WAV 파일 경로 입력 지원. JSONL 로그, verbose, debounce 옵션 지원 | `src/cli.py` |
| WAV 입력 | `librosa`로 16kHz mono 로드 후 0.96초 윈도우 / 0.48초 hop 생성 | `src/audio_io/file_reader.py` |
| 로컬 마이크 입력 | `sounddevice.InputStream`으로 16kHz mono 캡처 후 동일한 윈도우 생성 | `src/audio_io/mic_stream.py` |
| YAMNet 추론 | TF-Hub YAMNet v1 로드, 521 클래스 score 평균 산출 | `src/model/yamnet_wrapper.py` |
| 위험 클래스 필터 | 12개 위험 이벤트 추출. `glass_shatter`는 YAMNet 인덱스 `[435, 437]` max 통합 | `src/model/danger_filter.py`, `config/whitelist.yaml` |
| 후처리 | K/N debounce 구현됨. 기본 `window=3`, `k=2`, cooldown은 클래스별 적용 | `src/postprocess/trigger.py`, `config/whitelist.yaml` |
| 평가 스크립트 | WAV 샘플 평가, threshold sweep, summary 생성 | `scripts/evaluate_samples.py` |
| 검증 스크립트 | YAMNet 로딩 및 더미/파일 추론 확인 | `scripts/verify_inference.py` |
| 테스트 | debounce, synthetic signal, YAMNet shape 관련 테스트 존재 | `tests/` |

### 2.2 현재 샘플/평가 결과

현재 `data/sample/`에는 18개 WAV가 있다.

| 라벨 | 개수 |
|---|---:|
| `baby_cry` | 3 |
| `glass_shatter` | 3 |
| `siren` | 3 |
| `vehicle_horn` | 3 |
| `negative` | 6 |

최근 평가 결과는 `experiments/eval_20260511_193757/summary.md`와 `per_class_metrics.json`에 있다.

| 평가 방식 | 결과 |
|---|---|
| 단일 threshold 0.5 | Precision 0.9167, Recall 0.9167, F1 0.9167, FAR 0.1667 |
| `whitelist.yaml` 클래스별 threshold | Precision 1.0, Recall 1.0, F1 1.0, FAR 0.0 |

주의할 점은 샘플 수가 매우 작다는 것이다. 현재 결과는 파이프라인 동작 확인에는 의미가 있지만, 실제 환경 성능 근거로 쓰기에는 부족하다.

### 2.3 테스트 확인

이번 분석 중 실행한 테스트 결과는 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_debounce_trigger.py -q
```

결과: `29 passed`

전체 테스트는 YAMNet 캐시 문제로 실패했다.

```text
ValueError: ... tfhub_modules\9616fd04... contains neither saved_model.pb nor saved_model.pbtxt
```

즉, 코드 로직 전체가 깨졌다기보다 현재 PC의 TF-Hub 캐시 디렉터리에 불완전한 YAMNet 다운로드가 남아 있는 상태로 보인다. Jetson Orin에서 최초 세팅할 때는 `TFHUB_CACHE_DIR`를 명시하고 YAMNet 캐시를 새로 받아 검증해야 한다.

---

## 3. 아직 빠진 구현

목표 시스템을 만들기 위해 빠진 부분은 명확하다.

| 빠진 항목 | 필요 이유 | 권장 파일/위치 |
|---|---|---|
| ESP32 펌웨어 | I2S 마이크 캡처, UDP 송신, 결과 수신, 진동 PWM 제어가 필요 | 별도 `firmware/esp32_*` 또는 PlatformIO/Arduino 프로젝트 |
| Jetson 네트워크 오디오 수신 | 현재 `src/cli.py`는 `mic`/파일만 처리한다. ESP32 UDP PCM 입력 경로가 없다 | `src/audio_io/network_stream.py` |
| `--input network` CLI 모드 | Jetson에서 네트워크 스트림으로 기존 YAMNet 파이프라인을 재사용해야 한다 | `src/cli.py` |
| 두 ESP32 장치 식별 | 장치별 ring buffer, seq, trigger 상태를 분리해야 한다 | UDP 헤더 `device_id`, `seq`, `timestamp_ms` |
| Jetson 결과 송신 모듈 | 위험 이벤트를 ESP32로 JSON 전송해야 한다 | `src/embedded/wifi_sender.py` |
| ESP32 결과 수신 서버 | Jetson JSON을 받아 진동 패턴으로 변환해야 한다 | ESP32 펌웨어 |
| 노이즈 억제 | `src/preprocess/noise_suppress.py`는 현재 no-op이고 CLI에서도 호출되지 않는다 | `src/preprocess/noise_suppress.py`, `src/cli.py` |
| Jetson Orin 성능 검증 | Orin에서 실제 TensorFlow/YAMNet 추론 시간, CPU/GPU 사용량 확인 필요 | 벤치마크 스크립트 추가 |
| end-to-end 테스트 | 위험음 재생부터 진동까지 지연과 신뢰성 측정 필요 | 통합 테스트/실험 문서 |

현재 `src/embedded/uart_sender.py`는 placeholder라서 생성자와 메서드가 모두 `NotImplementedError`다. 이번 목표 구조는 Wi-Fi 기반이므로 UART는 유선 디버그/백업용으로만 유지하고, 실제 구현은 `wifi_sender.py`를 새로 두는 것이 맞다.

---

## 4. 권장 통신 설계

### 4.1 ESP32 -> Jetson Orin: UDP raw PCM

권장 포맷은 16kHz / 16bit signed little-endian / mono PCM이다.

| 항목 | 값 |
|---|---|
| UDP 헤더 엔디언 | network byte order, big-endian |
| UDP 헤더 struct | `!HBBIIHH` |
| 샘플레이트 | 16,000 Hz |
| 샘플 포맷 | int16 little-endian |
| 채널 | mono |
| 청크 길이 | 0.48초 |
| 청크 샘플 수 | 7,680 samples |
| payload 크기 | 15,360 bytes |
| ESP32 1대 대역폭 | 약 256 kbps |
| ESP32 2대 대역폭 | 약 512 kbps + UDP/IP overhead |

ESP32 2대 기준으로도 Wi-Fi 대역폭은 충분하다. 압축(Opus 등)은 ESP32 구현 복잡도와 위험음 transient 손실 위험이 커서 MVP에서는 쓰지 않는 편이 낫다.

### 4.2 UDP 패킷 헤더

두 ESP32를 구분하려면 기존 문서의 8바이트 헤더보다 장치 식별 필드가 필요하다. MVP에서는 아래 16바이트 고정 헤더를 권장한다.

```text
magic        uint16  0xA501
version      uint8   1
device_id    uint8   1 또는 2
seq          uint32  장치별 단조 증가
timestamp_ms uint32  ESP32 부팅 후 ms
payload_len  uint16  15360
flags        uint16  reserved
payload      int16[7680]
```

Jetson은 `(device_id, seq)`로 손실/중복/역전 패킷을 감지한다. timestamp는 디버그와 지연 측정용으로 쓰고, 판정 시각은 Jetson 수신 시각을 기준으로 삼는 것이 안전하다.

### 4.3 Jetson Orin 내부 처리

Jetson 쪽은 장치별 상태를 독립적으로 유지한다.

```text
DeviceStreamState
  device_id
  last_seq
  ring_buffer float32[]
  trigger Trigger
  last_packet_ts
```

처리 흐름:

```text
UDP packet 수신
  -> header 검증
  -> int16 PCM을 float32 [-1.0, 1.0]로 변환
  -> device_id별 ring buffer에 append
  -> 0.96초(15360 samples) 이상이면 window 생성
  -> YAMNet infer_mean_scores()
  -> DangerFilter.extract()
  -> Trigger.evaluate()
  -> 이벤트 발생 시 wifi_sender로 해당 ESP32 또는 양쪽 ESP32에 JSON 송신
```

MVP 정책은 “위험음이 들어온 ESP32에만 진동”이다. 착용형/양방향 알림이 목적이면 이후 “한쪽이 감지해도 양쪽 모두 진동”으로 정책을 바꿀 수 있다.

### 4.4 Jetson -> ESP32: JSON 결과

ESP32가 TCP 서버를 열고 Jetson이 각 ESP32에 연결하는 구조를 권장한다. 이벤트 유실을 줄이고 구현도 단순하다.

위험 이벤트 예시:

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

heartbeat 예시:

```json
{
  "event": "heartbeat",
  "seq": 143,
  "ts": 1779132005.000
}
```

---

## 5. 개발 순서

### 1단계: Jetson Orin 환경 고정

가장 먼저 Jetson Orin에서 현재 로컬 파이프라인이 정상 동작해야 한다.

완료 조건:

- Python 3.11 또는 Jetson에서 호환되는 TensorFlow 환경 확정
- `scripts/verify_inference.py` 성공
- `python -m src.cli --input data/sample/...wav --verbose` 성공
- TF-Hub 캐시 경로 고정: 예를 들어 `TFHUB_CACHE_DIR=/home/<user>/tfhub_modules`

### 2단계: PC/Jetson UDP 수신기 먼저 구현

ESP32 펌웨어를 바로 붙이기 전에 Jetson 쪽 `network_stream.py`를 먼저 만든다. 테스트용 Python UDP sender가 WAV 파일을 int16 PCM 청크로 쏘게 하면 ESP32 없이도 네트워크 수신 파이프라인을 검증할 수 있다.

구현 대상:

- `src/audio_io/network_stream.py`
- `scripts/send_pcm_udp.py` 같은 더미 송신 스크립트
- `src/cli.py`에 `--input network`, `--listen-port`, `--device-count` 옵션 추가

완료 조건:

- WAV 파일을 UDP로 송신했을 때 Jetson이 기존 파일 입력과 비슷한 score를 출력
- 패킷 손실/중복/역전 seq 로그 출력
- 장치 2대 시뮬레이션 가능

### 3단계: ESP32 한 대 PoC

한 대부터 붙인다. 두 대 동시 구현보다 한 대의 I2S 캡처 품질과 네트워크 안정성을 먼저 확인해야 한다.

ESP32 구현 항목:

- INMP441 또는 SPH0645 I2S 마이크 캡처
- 16kHz, 16bit, mono 변환
- 0.48초마다 UDP 패킷 송신
- `device_id=1`, `seq` 증가

완료 조건:

- Jetson에서 ESP32 오디오를 수신해 YAMNet score 출력
- 조용한 환경에서 false positive가 거의 없음
- 위험음 샘플 재생 시 해당 클래스 score 상승 확인

### 4단계: ESP32 두 대 동시 수신

두 장치가 동시에 Jetson으로 송신하게 만든다.

검증할 것:

- 두 장치의 `device_id`가 정확히 분리되는지
- 장치별 ring buffer가 섞이지 않는지
- 한 장치 패킷 손실이 다른 장치 판정에 영향이 없는지
- Jetson CPU/GPU 사용량과 큐 누적이 없는지

완료 조건:

- ESP32-A/B가 동시에 송신해도 각 장치별 score와 이벤트가 별도로 기록됨
- 10분 이상 스트리밍 중 수신 루프가 멈추지 않음

### 5단계: Jetson -> ESP32 결과 송신

`src/embedded/wifi_sender.py`를 구현하고 ESP32는 TCP 또는 UDP로 JSON을 받는다.

구현 대상:

- Jetson 결과 송신 모듈
- ESP32 JSON 파서
- heartbeat 처리
- 중복 이벤트 제거: `seq` 기준

완료 조건:

- Jetson에서 임의 danger JSON을 보내면 ESP32 진동 모터가 동작
- heartbeat 끊김 시 ESP32가 연결 상태 오류를 감지

### 6단계: 진동 모터 제어

MVP는 severity 기반 3단계로 충분하다.

| severity | PWM duty | 패턴 |
|---|---:|---|
| `high` | 100% | 1.5초 연속 |
| `medium` | 70% | 0.3초 ON x 3 |
| `low` | 40% | 0.2초 단발 |

완료 조건:

- JSON의 `severity`에 따라 진동 강도/패턴이 달라짐
- 모터 구동 회로가 ESP32 GPIO에 직접 과부하를 주지 않음
- MOSFET 또는 트랜지스터 드라이버와 역기전력 보호 다이오드 사용

### 7단계: end-to-end 검증

최종적으로 위험음 재생부터 진동까지 측정한다.

측정 항목:

- 평균 latency
- 95 percentile latency
- 패킷 손실률
- 장치별 false alarm
- 장치별 missed detection
- Jetson CPU/GPU/RAM 사용량
- ESP32 재연결 성공 여부

목표는 1초 이내로 잡는 것이 현실적이다. 현재 0.48초 hop과 2/3 debounce 구조에서는 200ms 수준은 어렵다.

---

## 6. 우선순위별 작업 목록

### P0: 바로 해야 하는 작업

- Jetson Orin에서 현재 YAMNet 로컬 추론 성공시키기
- TF-Hub 캐시 문제 해결 및 캐시 경로 고정
- `network_stream.py` 설계/구현
- UDP 패킷 헤더에 `device_id` 포함
- PC 더미 sender로 ESP32 2대 시뮬레이션

### P1: 통합에 필요한 작업

- ESP32 한 대 I2S 마이크 캡처 구현
- ESP32 UDP 송신 구현
- `src/cli.py --input network` 추가
- `wifi_sender.py` 구현
- ESP32 JSON 수신 및 진동 PWM 구현

### P2: 성능/품질 개선

- `noise_suppress.suppress()` 실제 구현 및 CLI 연결
- 클래스별 threshold 재측정
- 실제 환경 negative 1시간 이상 수집
- Jetson Orin 추론 시간 벤치마크
- 필요 시 TFLite/ONNX/TensorRT 검토

### P3: 확장 작업

- 두 ESP32 score fusion 정책
- 위험 감지 시 양쪽 ESP32 동시 진동 broadcast 정책
- 환경 프로파일(Home/Street/Public)
- 웹 대시보드 또는 로그 시각화

---

## 7. 주요 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| TF-Hub/YAMNet 캐시 불완전 | Jetson에서 모델 로딩 실패 | 캐시 삭제 후 재다운로드, `TFHUB_CACHE_DIR` 고정 |
| ESP32 I2S 캡처 품질 불량 | 분류 score 불안정 | WAV 저장 후 파형/볼륨 확인, gain/마이크 배선 점검 |
| 두 ESP32 패킷 혼선 | 장치별 이벤트 오판 | `device_id`, `seq`, 장치별 ring buffer 필수 |
| UDP 손실/순서 역전 | 윈도우 품질 저하 | seq로 감지, 오래된 패킷 drop, 실시간성 우선 |
| debounce 지연 | 위험음 후 진동까지 1초 이상 | 단발성 위험음은 `--no-debounce` 또는 클래스별 K/N 완화 검토 |
| 평가 데이터 부족 | 실제 환경 오탐/미탐 | 자체 negative/위험음 데이터 수집 후 threshold 재보정 |
| 노이즈 억제가 위험음을 깎음 | 유리 파손/총성 recall 저하 | NS on/off A/B 필수 |
| ESP32 전력 소비 | 배터리 운용 시간 감소 | MVP는 USB/상시전원 가정, 배터리는 후속 최적화 |

---

## 8. 구현 시 권장 파일 구조

```text
model/
├── src/
│   ├── audio_io/
│   │   ├── network_stream.py       # 신규: UDP PCM 수신, 장치별 ring buffer
│   │   ├── mic_stream.py           # 기존: 로컬 마이크
│   │   └── file_reader.py          # 기존: WAV
│   ├── embedded/
│   │   ├── wifi_sender.py          # 신규: Jetson -> ESP32 JSON 송신
│   │   └── uart_sender.py          # 기존: 유선 폴백 placeholder
│   ├── model/
│   ├── postprocess/
│   └── preprocess/
├── scripts/
│   ├── send_pcm_udp.py             # 신규 권장: WAV를 UDP PCM으로 송신하는 테스트 도구
│   └── measure_latency.py          # 신규 권장: end-to-end 지연 측정
├── firmware/
│   ├── esp32_audio_node_a/         # 신규 권장
│   └── esp32_audio_node_b/         # 신규 권장
└── docs/
```

---

## 9. 다음 액션 제안

바로 다음 구현은 `network_stream.py`부터 시작하는 것이 좋다. 이유는 ESP32 펌웨어가 없어도 PC에서 UDP 더미 오디오를 쏴서 Jetson 수신 파이프라인을 검증할 수 있고, 이후 ESP32는 같은 패킷 규격만 맞추면 되기 때문이다.

첫 구현 목표:

```powershell
python -m src.cli --input network --listen-port 5005 --device-count 2 --verbose
```

그리고 별도 sender로 아래처럼 검증한다.

```powershell
python scripts\send_pcm_udp.py --file data\sample\siren\1-31482-A-42.wav --device-id 1 --host <jetson-ip> --port 5005
python scripts\send_pcm_udp.py --file data\sample\glass_shatter\1-20133-A-39.wav --device-id 2 --host <jetson-ip> --port 5005
```

이 단계가 통과되면 ESP32 펌웨어는 “같은 패킷을 실제 마이크에서 만들어 보내는 코드”로 범위가 좁아진다.
