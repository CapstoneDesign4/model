# Jetson Orin P0 실행 Runbook

> 작성일: 2026-05-19  
> 목적: ESP32 2대 - Jetson Orin 위험음 분류 MVP의 P0 작업을 수행하기 위한 실행 절차를 정리한다.

---

## 1. P0 완료 기준

P0는 실제 ESP32 펌웨어 개발 전에 Jetson Orin의 모델 실행 환경과 네트워크 입력 준비 상태를 고정하는 단계다.

완료 기준:

- Python / TensorFlow / TensorFlow Hub 환경 확인
- `TFHUB_CACHE_DIR` 고정
- YAMNet 로컬 추론 성공
- 로컬 WAV 파일 추론 성공
- UDP packet 규격 확정
- `--input network` 실행 경로 확인
- `scripts/send_pcm_udp.py`로 WAV 기반 UDP 송신 가능

---

## 2. Jetson 환경 확인

Jetson Orin에서 저장소 루트로 이동한다.

```bash
cd /path/to/model
```

Python 실행 경로와 버전을 확인한다.

```bash
python --version
python -c "import sys; print(sys.executable); print(sys.version)"
```

필수 패키지 import를 확인한다.

```bash
python -c "import tensorflow as tf; print('tensorflow', tf.__version__)"
python -c "import tensorflow_hub as hub; print('tensorflow_hub ok')"
python -c "import numpy as np; print('numpy', np.__version__)"
python -c "import librosa; print('librosa ok')"
```

TensorFlow가 인식하는 장치를 확인한다.

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

CPU만 보여도 P0의 로컬 WAV 추론은 가능하다. GPU/TensorRT 최적화는 P0 이후 성능 단계에서 별도로 판단한다.

---

## 3. TF-Hub 캐시 고정

YAMNet 캐시 위치를 명시적으로 고정한다.

```bash
mkdir -p "$HOME/tfhub_modules"
export TFHUB_CACHE_DIR="$HOME/tfhub_modules"
python -c "import os; print(os.environ.get('TFHUB_CACHE_DIR'))"
```

매 터미널마다 적용하기 싫다면 쉘 설정에 추가한다.

```bash
echo 'export TFHUB_CACHE_DIR="$HOME/tfhub_modules"' >> ~/.bashrc
source ~/.bashrc
```

YAMNet 캐시가 불완전하면 다음 오류가 날 수 있다.

```text
contains neither saved_model.pb nor saved_model.pbtxt
```

이 경우 기존 캐시를 백업하고 새로 받는다.

```bash
mv "$TFHUB_CACHE_DIR" "${TFHUB_CACHE_DIR}_broken_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TFHUB_CACHE_DIR"
python scripts/verify_inference.py
```

---

## 4. 로컬 YAMNet 검증

먼저 더미 입력으로 YAMNet 로딩과 추론을 확인한다.

```bash
python scripts/verify_inference.py
```

성공 기준:

```text
PASS: YAMNet inference OK
```

샘플 WAV가 있는지 확인한다.

```bash
find data/sample -name "*.wav" | head
```

대표 샘플로 CLI 추론을 실행한다.

```bash
python -m src.cli --input data/sample/siren/1-31482-A-42.wav --verbose
python -m src.cli --input data/sample/glass_shatter/1-20133-A-39.wav --verbose
python -m src.cli --input data/sample/baby_cry/1-187207-A-20.wav --verbose
python -m src.cli --input data/sample/vehicle_horn/1-17124-A-43.wav --verbose
```

샘플 평가를 다시 돌릴 수도 있다.

```bash
python scripts/evaluate_samples.py --data-dir data/sample --per-class-threshold
```

---

## 5. UDP 패킷 규격

현재 구현 기준은 다음과 같다.

| 항목 | 값 |
|---|---|
| header byte order | network byte order, big-endian |
| Python struct format | `!HBBIIHH` |
| header size | 16 bytes |
| payload format | signed int16 little-endian PCM |
| sample rate | 16,000 Hz |
| channel | mono |
| samples per packet | 7,680 |
| payload size | 15,360 bytes |
| packet size | 15,376 bytes |

Header:

```text
magic        uint16  0xA501
version      uint8   1
device_id    uint8   1 or 2
seq          uint32  device-local monotonic sequence
timestamp_ms uint32  ESP32 uptime milliseconds
payload_len  uint16  15360
flags        uint16  0
```

Payload:

```text
int16[7680] little-endian PCM
```

Python 수신부 구현 위치:

- `src/audio_io/network_stream.py`

테스트 sender 구현 위치:

- `scripts/send_pcm_udp.py`

---

## 6. Network Receiver 실행

Jetson에서 network mode를 실행한다.

```bash
python -m src.cli --input network --listen-port 5005 --device-count 2 --verbose
```

특정 바인드 주소가 필요하면 지정한다.

```bash
python -m src.cli --input network --listen-host 0.0.0.0 --listen-port 5005 --device-count 2 --verbose
```

로그 파일도 남길 수 있다.

```bash
python -m src.cli --input network --listen-port 5005 --device-count 2 --verbose --log output/network_run.jsonl
```

---

## 7. WAV 기반 UDP Sender 실행

다른 터미널 또는 다른 PC에서 sender를 실행한다.

장치 1 시뮬레이션:

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/siren/1-31482-A-42.wav \
  --device-id 1 \
  --host <jetson-ip> \
  --port 5005
```

장치 2 시뮬레이션:

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/glass_shatter/1-20133-A-39.wav \
  --device-id 2 \
  --host <jetson-ip> \
  --port 5005
```

반복 송신:

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/siren/1-31482-A-42.wav \
  --device-id 1 \
  --host <jetson-ip> \
  --port 5005 \
  --loop
```

빠른 기능 테스트만 할 때는 실시간 sleep 없이 전송할 수 있다.

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/siren/1-31482-A-42.wav \
  --device-id 1 \
  --host 127.0.0.1 \
  --port 5005 \
  --no-realtime
```

---

## 8. 성공 기준

Network receiver 콘솔에서 다음을 확인한다.

- `device=1`, `device=2`가 분리되어 출력됨
- `seq`가 증가함
- WAV 파일에 맞는 위험 클래스 score가 상승함
- `DANGER: siren`, `DANGER: glass_shatter` 같은 이벤트가 발생함
- `output/network_run.jsonl`에 `device_id`, `packet_seq`, `network_stats`가 기록됨

---

## 9. 실패 대응

| 증상 | 가능 원인 | 대응 |
|---|---|---|
| `ModuleNotFoundError: tensorflow` | 가상환경 미활성화 또는 TensorFlow 미설치 | Python 실행 경로 확인, Jetson용 TensorFlow 환경 재구성 |
| `saved_model.pb` 없음 오류 | TF-Hub 캐시 불완전 | `TFHUB_CACHE_DIR` 확인, 캐시 백업/삭제 후 재다운로드 |
| WAV 파일 경로 오류 | 실제 파일명이 다름 | `find data/sample -name "*.wav"`로 파일명 확인 |
| receiver가 UDP를 못 받음 | IP/port/방화벽 문제 | `<jetson-ip>`, `--listen-port`, 같은 네트워크 여부 확인 |
| `bad_packets` 증가 | header 형식 또는 payload 크기 불일치 | `!HBBIIHH`, payload 15,360 bytes 확인 |
| `unknown_device_packets` 증가 | `device_id` 범위 오류 | `--device-id 1` 또는 `--device-id 2` 사용 |
| 장치별 score가 섞임 | device state 분리 오류 | `device_id` 로그와 JSONL의 `device_id` 확인 |
| latency가 큼 | YAMNet 로딩/추론 또는 debounce 지연 | `--no-debounce` 비교, 추론 시간 별도 측정 |

---

## 10. P0 체크리스트

- [ ] Jetson에서 Python/TensorFlow/TensorFlow Hub import 성공
- [ ] `TFHUB_CACHE_DIR` 고정
- [ ] `python scripts/verify_inference.py` 성공
- [ ] 로컬 WAV CLI 추론 성공
- [ ] `python -m src.cli --input network --listen-port 5005 --device-count 2 --verbose` 실행 확인
- [ ] `scripts/send_pcm_udp.py`로 `device_id=1` 송신 확인
- [ ] `scripts/send_pcm_udp.py`로 `device_id=2` 송신 확인
- [ ] JSONL 로그에 `device_id`, `packet_seq`, `network_stats` 기록 확인

