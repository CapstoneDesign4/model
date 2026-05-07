# M1 베이스라인 초기 모델 스펙

> 대상: 모델 개발 에이전트  
> 버전: v0.1 (2026-05-07)  
> 참조: `docs/development-plan.md` §6.2, §8 (M1 단계)

---

## 1. 목표 (Scope / Non-Scope)

### 1.1 Scope — M1에서 만드는 것

| 항목 | 내용 |
|---|---|
| YAMNet 추론 파이프라인 | TF-Hub에서 YAMNet 로드, 0.96초 윈도우 단위 추론 |
| 위험 클래스 필터링 | 12종 화이트리스트 score 추출 및 출력 |
| 단순 임계값 후처리 | 클래스별 단일 고정 임계값 + cooldown(단순) |
| 파일/마이크 입력 지원 | WAV 파일 및 실시간 마이크 두 모드 |
| CLI 진입점 | `python -m src.cli` 형태로 실행 |
| 추론 결과 콘솔 출력 | 트리거된 클래스, score, 타임스탬프 출력 |
| 추론 동작 검증 스크립트 | 짧은 WAV 1개로 파이프라인 통과 확인 |

### 1.2 Non-Scope — M1에서 만들지 않는 것

| 항목 | 이유 / 담당 마일스톤 |
|---|---|
| 노이즈 캔슬링(WebRTC NS) | M2에서 통합 |
| 경량 분류 헤드 학습 | M3에서 FSD50K 기반 학습 |
| Debounce(K/N 다수결) | M2로 미룸. M1은 단순 cooldown만 |
| TFLite 변환/양자화 | M4 |
| 임베디드 UART 송신 | M5 |
| 환경별 프로파일 분기 | M3 이후 |

---

## 2. 입력 사양

| 항목 | 값 | 근거 |
|---|---|---|
| 샘플레이트 | 16,000 Hz | YAMNet 표준 요구사항 |
| 채널 | Mono (1ch) | YAMNet 단채널 입력 |
| 비트 깊이 | 16-bit PCM | 일반 마이크/파일 호환 |
| 윈도우 길이 | 0.96 s (15,360 samples) | YAMNet 내부 프레임 구조 표준 |
| Hop 길이 | 0.48 s (7,680 samples) | 50% 오버랩, 실시간 지연 최소화 |
| 입력 소스 (파일) | WAV 형식, 16kHz/mono 기준. 다른 포맷은 librosa로 리샘플링 후 처리 | |
| 입력 소스 (마이크) | sounddevice 또는 pyaudio로 스트림 캡처, 링 버퍼에 0.96s 누적 후 추론 | |

### 2.1 입력 전처리 흐름 (의사코드)

```
[파일 모드]
  audio, sr = librosa.load(path, sr=16000, mono=True)
  frames = sliding_window(audio, window=15360, hop=7680)
  for frame in frames:
      run_inference(frame)

[마이크 모드]
  stream = open_mic(sr=16000, channels=1, chunk=7680)
  ring_buffer = RingBuffer(size=15360)
  loop:
      chunk = stream.read(7680)
      ring_buffer.push(chunk)
      if ring_buffer.is_full():
          run_inference(ring_buffer.snapshot())
```

---

## 3. 모델 구성

### 3.1 YAMNet 로딩

| 항목 | 값 |
|---|---|
| 소스 | TensorFlow Hub: `https://tfhub.dev/google/yamnet/1` |
| 고정(freeze) 여부 | Backbone 전체 freeze (M1은 파인튜닝 없음) |
| 출력 선택 | `scores` (521개 클래스별 확률) 사용 |
| 임베딩 사용 | M1에서는 미사용. M3 헤드 학습 시 `embeddings` 출력 사용 예정 |

### 3.2 scores vs embeddings 선택 근거

| | scores | embeddings |
|---|---|---|
| M1 적합성 | 바로 사용 가능, 추가 학습 불필요 | 헤드 학습 필요 |
| 성능 | 사전학습 분류 성능 그대로 | 헤드 튜닝 시 더 높음 |
| 구현 복잡도 | 낮음 | 중간 |
| M1 선택 | **scores 선택** | M3에서 전환 |

### 3.3 YAMNet 입력 형식

- 입력: float32 numpy array, shape `(N_samples,)`, 값 범위 `[-1.0, 1.0]`
- YAMNet 내부에서 mel spectrogram 계산 수행 (외부에서 spectrogram 계산 불필요)
- 0.96s 윈도우 1개 입력 시 YAMNet은 내부적으로 0.48s 패치 2개를 처리하고 프레임별 scores를 반환. M1에서는 프레임별 scores를 평균 내어 윈도우 단위 1개의 scores 벡터로 사용

```
yamnet_input  : float32, shape (15360,)
yamnet_output :
  scores      : shape (num_patches, 521)  → mean(axis=0) → shape (521,)
  embeddings  : shape (num_patches, 1024)  [M1에서는 미사용]
  spectrogram : shape (num_patches, 64)    [M1에서는 미사용]
```

---

## 4. 위험 클래스 화이트리스트 (12종)

> `yamnet_class_map.csv` 기준 확인된 정확한 인덱스

| 순번 | 클래스 키 (내부 enum) | YAMNet 인덱스 | AudioSet MID | display_name (CSV 기준) | 시나리오 | 우선순위 |
|---|---|---|---|---|---|---|
| 1 | `screaming` | **11** | /m/03qc9zr | Screaming | 가정/공공 | 필수 |
| 2 | `baby_cry` | **20** | /t/dd00002 | Baby cry, infant cry | 가정 | 필수 |
| 3 | `glass_shatter` | **437** | /m/07rn7sz | Shatter | 가정/공공 | 필수 |
| 4 | `glass` | **435** | /m/039jq | Glass | 가정/공공 | 보조(437과 함께 사용) |
| 5 | `breaking` | **464** | /m/07pc8lb | Breaking | 가정/공공 | 권장 |
| 6 | `gunshot` | **421** | /m/032s66 | Gunshot, gunfire | 실외/공공 | 필수 |
| 7 | `explosion` | **420** | /m/014zdl | Explosion | 실외 | 필수 |
| 8 | `fire_alarm` | **394** | /m/0c3f7m | Fire alarm | 가정/공공 | 필수 |
| 9 | `smoke_alarm` | **393** | /m/01y3hg | Smoke detector, smoke alarm | 가정 | 필수 |
| 10 | `siren` | **390** | /m/030rvx | Siren | 실외 | 필수 |
| 11 | `civil_defense_siren` | **391** | /m/0dgbq | Civil defense siren | 실외 | 필수 |
| 12 | `car_alarm` | **304** | /m/02mfyn | Car alarm | 실외 | 권장 |
| 13 | `vehicle_horn` | **302** | /m/0912c9 | Vehicle horn, car horn, honking | 실외 | 권장 |

> 인덱스 총 13개를 사용하지만 `glass`(435)와 `glass_shatter`(437)은 논리적으로 묶어 "유리 파손" 이벤트 1개로 처리. 화이트리스트 설정 파일에 13개 인덱스를 모두 등록하되, 알림 이벤트는 12종으로 그룹핑.

### 4.1 화이트리스트 설정 파일 형식 (config/whitelist.yaml, 의사코드)

```yaml
danger_classes:
  - key: screaming
    yamnet_index: 11
    threshold: 0.5
    cooldown_sec: 5
  - key: baby_cry
    yamnet_index: 20
    threshold: 0.5
    cooldown_sec: 5
  - key: glass_shatter
    yamnet_indices: [435, 437]   # 복수 인덱스 max() 취합
    threshold: 0.5
    cooldown_sec: 5
  - key: breaking
    yamnet_index: 464
    threshold: 0.5
    cooldown_sec: 5
  - key: gunshot
    yamnet_index: 421
    threshold: 0.5
    cooldown_sec: 5
  - key: explosion
    yamnet_index: 420
    threshold: 0.5
    cooldown_sec: 5
  - key: fire_alarm
    yamnet_index: 394
    threshold: 0.5
    cooldown_sec: 5
  - key: smoke_alarm
    yamnet_index: 393
    threshold: 0.5
    cooldown_sec: 5
  - key: siren
    yamnet_index: 390
    threshold: 0.5
    cooldown_sec: 5
  - key: civil_defense_siren
    yamnet_index: 391
    threshold: 0.5
    cooldown_sec: 5
  - key: car_alarm
    yamnet_index: 304
    threshold: 0.5
    cooldown_sec: 5
  - key: vehicle_horn
    yamnet_index: 302
    threshold: 0.5
    cooldown_sec: 5
```

---

## 5. 출력 사양

### 5.1 추론 1회 출력 (윈도우 단위)

| 필드 | 타입 | 설명 |
|---|---|---|
| `timestamp` | float | 윈도우 시작 시각 (Unix epoch, 초) |
| `window_duration_ms` | int | 고정 960 |
| `scores` | dict[str, float] | 화이트리스트 12종의 클래스별 score (0.0~1.0) |
| `triggered` | list[str] | 임계값 초과 + cooldown 통과한 클래스 키 목록 |
| `top_score` | float | triggered 중 최고 score (triggered 없으면 null) |

### 5.2 콘솔 출력 형식 (M1 단순 버전)

```
[2026-05-07 14:32:01.123] DANGER: screaming (score=0.82)
[2026-05-07 14:32:01.123] DANGER: glass_shatter (score=0.71)
[2026-05-07 14:32:03.610] -- no danger (top: siren=0.22)
```

- 위험 클래스가 트리거되지 않은 윈도우는 `-- no danger` + 최고 score 클래스 1개 표시.
- `--verbose` 플래그 시 12종 전체 score 표시.

---

## 6. 후처리 (M1 단순 버전)

> M1은 단순하게 구현. 복잡한 debounce(K/N 다수결)는 M2에서 추가.

### 6.1 처리 순서

```
1. scores_521 = yamnet(frame).mean(axis=0)          # shape (521,)
2. for each class in whitelist:
       score = max(scores_521[class.yamnet_indices])  # 복수 인덱스는 max
3. for each class:
       if score >= class.threshold:
           if now - last_trigger[class] >= class.cooldown_sec:
               emit_event(class, score, timestamp)
               last_trigger[class] = now
```

### 6.2 M1 후처리 파라미터

| 파라미터 | M1 기본값 | 비고 |
|---|---|---|
| 클래스별 threshold | 0.5 (전체 동일) | CLI `--threshold`로 오버라이드 가능 |
| cooldown | 5초 | 동일 클래스 연속 알림 억제 |
| debounce (K/N) | 미구현 | M2로 미룸 |
| 다중 클래스 처리 | 독립 평가 (multi-label) | argmax 아님 |

### 6.3 M2로 미루는 이유

- Debounce(다수결)는 직전 N개 윈도우 결과를 저장하는 상태가 필요하며, M1에서는 파이프라인 정합성 검증이 우선이므로 단순 cooldown만 구현.
- Cooldown만으로도 동일 이벤트의 연속 발화는 억제 가능.

---

## 7. CLI 사양

### 7.1 진입점

```
python -m src.cli [옵션]
```

### 7.2 옵션 목록

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--input` | str | 필수 | `mic` 또는 WAV 파일 경로 |
| `--threshold` | float | 0.5 | 전체 클래스 공통 임계값 오버라이드 |
| `--config` | str | `config/whitelist.yaml` | 화이트리스트 설정 파일 경로 |
| `--hop` | float | 0.48 | hop 길이(초). 실시간 지연 조정 용 |
| `--verbose` | flag | False | 12종 전체 score 매 윈도우 출력 |
| `--log` | str | None | 결과를 JSONL 파일로 저장할 경로 |
| `--device` | int | 0 | 마이크 장치 인덱스 (mic 모드 한정) |

### 7.3 사용 예시

```
# WAV 파일 분석
python -m src.cli --input data/sample/test.wav --threshold 0.5 --verbose

# 마이크 실시간 분석
python -m src.cli --input mic --threshold 0.4 --log output/run.jsonl

# 특정 설정 파일 사용
python -m src.cli --input mic --config config/whitelist_home.yaml
```

---

## 8. 디렉터리 구조 (M1 생성 대상)

> `model-developer.md`의 컨벤션을 따른다. M1에서 생성할 파일에 `[M1]` 표시.

```
model/
├── src/
│   ├── __init__.py                  [M1]
│   ├── cli.py                       [M1]  진입점
│   ├── audio_io/
│   │   ├── __init__.py              [M1]
│   │   ├── file_reader.py           [M1]  WAV 로드 + librosa 리샘플링
│   │   └── mic_stream.py            [M1]  마이크 스트림 + 링 버퍼
│   ├── preprocess/
│   │   ├── __init__.py              [M1]  (M1은 빈 스켈레톤)
│   │   └── noise_suppress.py        [M2]  WebRTC NS (M2에서 구현)
│   ├── model/
│   │   ├── __init__.py              [M1]
│   │   ├── yamnet_wrapper.py        [M1]  TF-Hub 로드, 추론 래퍼
│   │   └── danger_filter.py         [M1]  화이트리스트 필터링 + score 추출
│   ├── postprocess/
│   │   ├── __init__.py              [M1]
│   │   └── trigger.py               [M1]  임계값 비교 + cooldown
│   └── embedded/
│       ├── __init__.py              [M1]  (M1은 빈 스켈레톤)
│       └── uart_sender.py           [M5]  UART 송신 (M5에서 구현)
├── config/
│   └── whitelist.yaml               [M1]  화이트리스트 + 임계값 설정
├── data/
│   └── sample/                      [M1]  검증용 WAV 샘플 보관
├── experiments/                     [M1]  실험 결과 노트 (빈 디렉터리)
├── tests/
│   ├── __init__.py                  [M1]
│   └── test_yamnet_load.py          [M1]  YAMNet 로딩 검증 테스트
├── scripts/
│   └── verify_inference.py          [M1]  단일 WAV 추론 검증 스크립트
└── requirements.txt                 [M1]
```

---

## 9. 의존성 목록

### 9.1 requirements.txt (M1 기준)

| 패키지 | 권장 버전 | 용도 | 비고 |
|---|---|---|---|
| `tensorflow` | `>=2.13, <2.17` | YAMNet 추론 | GPU 있으면 `tensorflow[and-cuda]` |
| `tensorflow-hub` | `>=0.15` | TF-Hub YAMNet 로딩 | |
| `numpy` | `>=1.24, <2.0` | 배열 연산 | TF 버전과 호환 확인 필요 |
| `librosa` | `>=0.10` | 오디오 로드, 리샘플링 | |
| `soundfile` | `>=0.12` | WAV read/write | librosa 백엔드 |
| `scipy` | `>=1.11` | 필터, 신호처리 | |
| `sounddevice` | `>=0.4.6` | 마이크 스트림 | PortAudio 설치 필요 |
| `pyyaml` | `>=6.0` | whitelist.yaml 로드 | |
| `pytest` | `>=7.4` | 단위 테스트 | dev 의존성 |

### 9.2 버전 호환 주의 사항

- TensorFlow 2.16+는 Keras 3 기본 사용 → 기존 코드와 충돌 가능. `TF_USE_LEGACY_KERAS=1` 환경변수 또는 `tf-keras` 패키지 별도 설치 필요.
- `sounddevice`는 PortAudio C 라이브러리 의존. Windows: `pip install sounddevice`로 자동 설치, Linux/라즈베리파이: `sudo apt install libportaudio2` 필요.
- `librosa` 0.10+는 `audioread` 대신 `soundfile` 우선 사용. WAV 외 포맷 처리 시 `ffmpeg` 설치 권장.

### 9.3 M2 이후 추가 예정 패키지 (M1에서는 불필요)

| 패키지 | 마일스톤 | 용도 |
|---|---|---|
| `noisereduce` 또는 `webrtc-noise-gain` | M2 | 노이즈 캔슬링 |
| `pyserial` | M5 | UART 통신 |
| `paho-mqtt` | M5 | MQTT 통신 |

---

## 10. 초기 설정 체크리스트

> 모델 개발 에이전트가 첫 커밋/PR에서 처리해야 할 항목

### 10.1 환경 구성

- [ ] Python 3.10 이상 가상환경 생성 (`python -m venv .venv`)
- [ ] `requirements.txt` 작성 및 `pip install -r requirements.txt` 확인
- [ ] Windows 환경에서 PortAudio 설치 확인 (`python -c "import sounddevice"`)
- [ ] TensorFlow 설치 확인 (`python -c "import tensorflow as tf; print(tf.__version__)"`)

### 10.2 YAMNet 로딩 검증

- [ ] `scripts/verify_inference.py` 작성 및 실행
  - TF-Hub에서 YAMNet 다운로드 성공 확인
  - 0.96s 더미 오디오(`numpy.zeros(15360)`) 입력 → scores shape `(2, 521)` 반환 확인
  - 최초 실행 시 모델 캐시 경로(`~/.cache/tfhub_modules`) 확인
- [ ] `class_map.csv` 접근 가능 여부 확인 (TF-Hub 모델 내 포함)

### 10.3 화이트리스트 설정

- [ ] `config/whitelist.yaml` 생성 (12종 인덱스 + 기본 threshold 0.5)
- [ ] `src/model/danger_filter.py`에서 YAML 로드 후 인덱스 매핑 단위 테스트

### 10.4 스켈레톤 파일

- [ ] `src/` 하위 디렉터리 및 `__init__.py` 일괄 생성
- [ ] `experiments/`, `data/sample/`, `tests/` 디렉터리 생성
- [ ] M2/M5 미구현 모듈은 빈 파일 + `NotImplementedError` 플레이스홀더 추가

### 10.5 샘플 데이터

- [ ] 검증용 WAV 파일 1개 이상 `data/sample/`에 배치
  - 권장: ESC-50 또는 FSD50K에서 fire alarm 또는 siren 클립 1개 (공개 라이선스 확인)
  - 대안: `scipy.io.wavfile`로 생성한 합성 tone 파일 (1kHz 사인파 0.96s)

### 10.6 문서

- [ ] `CLAUDE.md`의 "시작하기" 및 "명령어" 섹션 업데이트
- [ ] 첫 실행 시 필요한 커맨드 라인 전체를 README 또는 CLAUDE.md에 기록

---

## 11. 검증 방법

### 11.1 단계별 검증 체크리스트

| 단계 | 검증 방법 | 성공 기준 |
|---|---|---|
| 1. YAMNet 로드 | `scripts/verify_inference.py` 실행 | 오류 없이 scores 출력, shape `(*, 521)` 확인 |
| 2. 화이트리스트 필터 | 더미 scores 배열 입력 후 12종 추출 확인 | 지정 인덱스의 값이 정확히 추출됨 |
| 3. 파일 추론 | `python -m src.cli --input data/sample/test.wav --verbose` | 콘솔에 각 윈도우 score 출력, 크래시 없음 |
| 4. 임계값 트리거 | `--threshold 0.0` 으로 전체 클래스 강제 트리거 | `DANGER:` 로그가 각 윈도우마다 출력됨 |
| 5. cooldown 동작 | 동일 파일 반복 실행, 5초 내 재트리거 없는지 확인 | 같은 클래스 연속 출력이 cooldown_sec 이내 억제됨 |
| 6. 마이크 모드 | `python -m src.cli --input mic` 실행 후 박수 소리 입력 | 마이크 스트림 시작 로그 출력, 크래시 없음 |

### 11.2 verify_inference.py 의사코드

```
[입력]
  mode: "dummy" | "file"
  path: WAV 파일 경로 (file 모드)

[처리]
  1. tensorflow_hub.load("https://tfhub.dev/google/yamnet/1") → yamnet
  2. if mode == "dummy":
       waveform = zeros(15360, dtype=float32)
     else:
       waveform = librosa.load(path, sr=16000, mono=True)[0][:15360]
  3. scores, embeddings, spectrogram = yamnet(waveform)
  4. print("scores shape:", scores.shape)         # 기대: (2, 521)
  5. print("embeddings shape:", embeddings.shape) # 기대: (2, 1024)
  6. danger_indices = [11, 20, 437, 435, 464, 421, 420, 394, 393, 390, 391, 304, 302]
  7. avg_scores = mean(scores, axis=0)
  8. for idx in danger_indices:
       print(f"  [{idx}] {class_names[idx]}: {avg_scores[idx]:.4f}")
  9. print("PASS: YAMNet inference OK")
```

### 11.3 합격/불합격 기준

| 항목 | 합격 | 불합격 시 조치 |
|---|---|---|
| YAMNet 로드 | 오류 없이 완료 | TF-Hub 캐시 초기화, 네트워크 확인 |
| scores shape | `(2, 521)` 또는 `(N, 521)` | TF 버전 확인, 입력 dtype float32 확인 |
| 위험 클래스 score 출력 | 13개 인덱스 모두 출력 | 인덱스 오타 확인, class_map.csv 대조 |
| 파일 추론 | 크래시 없이 완료 | 샘플레이트/채널 확인, librosa 재설치 |
| CLI 실행 | 도움말 출력 (`--help`) | argparse 설정 확인 |

---

## 12. M1 완료 조건 (Exit Criteria)

> `development-plan.md` §8 M1 Exit Criteria 기준

| 조건 | 측정 방법 |
|---|---|
| 자체 평가셋 위험 클래스 F1 ≥ 0.6 | ESC-50 siren/glass_shatter/fire 클립 대상 threshold=0.5 기준 |
| Latency 측정 완료 | 파일 모드 100 윈도우 추론 시간 측정, 95퍼센타일 기록 |
| 마이크 모드 크래시 없이 1분 동작 | 직접 테스트 |
| CLI `--help` 정상 출력 | 실행 확인 |
| `verify_inference.py` PASS | 스크립트 실행 확인 |

---

*문서 버전: v0.1 (2026-05-07). M2 착수 전 noise-cancelling 전략 확정 후 본 문서 §6 업데이트 예정.*
