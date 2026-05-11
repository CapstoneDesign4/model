# 프로젝트 파일 요약

이 문서는 `C:\CapstoneDesign\model` 프로젝트의 현재 구조와 관련 파일 내용을 요약합니다. 기준 시점은 M1 베이스라인 구현 상태입니다.

## 1. 프로젝트 개요

이 프로젝트는 YAMNet 기반 위험 소리 감지 모델입니다. WAV 파일 또는 마이크 입력을 16kHz mono 오디오로 처리하고, 0.96초 윈도우와 0.48초 hop 단위로 YAMNet 추론을 수행합니다. YAMNet의 521개 AudioSet score 중 위험 소리 12종만 추출한 뒤 threshold와 cooldown 조건으로 위험 이벤트를 출력합니다.

전체 처리 흐름:

```text
WAV 파일 또는 마이크 입력
  -> 16kHz mono 오디오 프레임 생성
  -> TF-Hub YAMNet 추론
  -> 위험 클래스 12종 score 추출
  -> threshold + cooldown 판정
  -> 콘솔 출력 또는 JSONL 로그 저장
```

## 2. 현재 구현 범위

구현된 기능:

- TF-Hub YAMNet 모델 로드
- WAV 파일 입력 처리
- `sounddevice` 기반 마이크 실시간 입력
- 0.96초 윈도우 / 0.48초 hop 슬라이딩 처리
- 위험 소리 12종 화이트리스트 score 추출
- 클래스별 threshold 및 cooldown 트리거
- CLI 실행 진입점
- JSONL 로그 저장
- YAMNet 로딩 검증 스크립트
- 주요 단위 테스트

아직 구현되지 않은 기능:

- 실제 노이즈 캔슬링 전처리
- K/N debounce 후처리
- YAMNet embedding 기반 경량 분류 헤드 학습
- TFLite 변환 및 양자화
- UART/Serial 임베디드 송신

## 3. 루트 파일 요약

| 파일 | 내용 |
|---|---|
| `README.md` | 처음 설치하는 사용자를 위한 설치 튜토리얼입니다. Python 버전, 가상환경, 의존성 설치, YAMNet 검증, WAV/마이크 실행, 테스트, 문제 해결을 설명합니다. |
| `CLAUDE.md` | 개발 에이전트용 지침 문서입니다. 프로젝트 목표, 시작 명령어, 주요 CLI 사용법, 아키텍처와 설계 결정을 요약합니다. |
| `requirements.txt` | M1 베이스라인 실행에 필요한 Python 패키지 목록입니다. Python 3.11 기준으로 TensorFlow `>=2.13,<2.16`을 사용하며, `tensorflow-hub`, `numpy<2.0`, `librosa`, `sounddevice`, `pyyaml`, `pytest` 등이 포함됩니다. |
| `.gitignore` | 가상환경, Python 캐시, 빌드 산출물, 대용량 데이터, 오디오 파일, 모델 가중치, TF-Hub 캐시, 로그와 IDE 파일을 제외합니다. `data/sample/` 디렉터리는 허용하지만 오디오 확장자는 기본적으로 제외됩니다. |

## 4. 설정 파일

### `config/whitelist.yaml`

YAMNet 521개 클래스 중 위험 소리로 볼 클래스를 정의합니다. 각 항목은 내부 key, 표시 이름, YAMNet index, threshold, cooldown을 가집니다.

현재 위험 클래스:

| key | YAMNet index | 설명 |
|---|---:|---|
| `screaming` | 11 | 비명 |
| `baby_cry` | 20 | 영유아 울음 |
| `glass_shatter` | 435, 437 | 유리 및 파손음. 두 index score 중 max 사용 |
| `breaking` | 464 | 파손음 |
| `gunshot` | 421 | 총성 |
| `explosion` | 420 | 폭발음 |
| `fire_alarm` | 394 | 화재 경보 |
| `smoke_alarm` | 393 | 연기 감지기 경보 |
| `siren` | 390 | 사이렌 |
| `civil_defense_siren` | 391 | 민방위 사이렌 |
| `car_alarm` | 304 | 차량 경보 |
| `vehicle_horn` | 302 | 차량 경적 |

기본 threshold는 `0.5`, cooldown은 `5.0`초입니다.

## 5. 소스 코드 요약

### `src/cli.py`

CLI 실행 진입점입니다.

실행 형태:

```powershell
python -m src.cli --input mic
python -m src.cli --input data\sample\test.wav
```

주요 역할:

- `argparse`로 CLI 옵션 파싱
- `DangerFilter`로 화이트리스트 설정 로드
- `--threshold` 값이 기본값과 다르면 모든 클래스 threshold 오버라이드
- `YAMNetWrapper`로 TF-Hub YAMNet 로드
- 입력이 `mic`이면 마이크 모드 실행
- 입력이 파일 경로이면 파일 모드 실행
- 위험 이벤트는 `DANGER: key (score=...)` 형식으로 출력
- 이벤트가 없고 `--verbose`가 꺼져 있으면 최고 score 클래스만 출력
- `--log`가 있으면 윈도우별 결과를 JSONL로 저장

주요 CLI 옵션:

| 옵션 | 설명 |
|---|---|
| `--input` | 필수. `mic` 또는 오디오 파일 경로 |
| `--threshold` | 전체 클래스 공통 threshold |
| `--config` | 화이트리스트 YAML 경로 |
| `--hop` | hop 길이, 초 단위 |
| `--verbose` | 매 윈도우의 12종 score 출력 |
| `--log` | JSONL 로그 저장 경로 |
| `--device` | 마이크 장치 인덱스 |

### `src/audio_io/file_reader.py`

WAV 등 파일 입력을 처리합니다.

주요 내용:

- `librosa.load(..., sr=16000, mono=True)`로 파일을 16kHz mono float32 배열로 로드
- 기본 윈도우 길이는 15360 sample, 즉 0.96초
- 기본 hop은 7680 sample, 즉 0.48초
- 마지막 프레임이 짧으면 zero padding
- `iter_file_frames()`가 `(timestamp_sec, frame)` 형태로 프레임을 생성

### `src/audio_io/mic_stream.py`

마이크 실시간 입력을 처리합니다.

주요 내용:

- `sounddevice.InputStream`을 사용해 16kHz, mono, float32 입력 스트림 생성
- 콜백에서 들어온 오디오 chunk를 queue에 저장
- 내부 버퍼에 chunk를 누적한 뒤 0.96초 이상 모이면 프레임 생성
- 프레임 생성 후 hop 길이만큼 버퍼를 이동
- 입력 overflow 등 `sounddevice` 상태 경고는 stderr로 출력하고 계속 동작

### `src/model/yamnet_wrapper.py`

TF-Hub YAMNet 모델 래퍼입니다.

주요 내용:

- 모델 URL은 `https://tfhub.dev/google/yamnet/1`
- `hub.load()`로 모델 로드
- `infer()`는 `scores`, `embeddings`, `spectrogram`을 numpy 배열로 반환
- `infer_mean_scores()`는 패치별 `scores`를 평균내어 shape `(521,)` 벡터로 반환
- M1에서는 embedding 학습을 하지 않고 YAMNet score를 직접 사용

### `src/model/danger_filter.py`

화이트리스트 기반 위험 클래스 score 추출 모듈입니다.

주요 내용:

- 기본 설정 파일은 `config/whitelist.yaml`
- `DangerClassEntry`가 key, display name, YAMNet index 목록, threshold, cooldown을 보관
- `DangerFilter.extract()`는 shape `(521,)` score 벡터를 받아 위험 클래스 12종 score dict를 반환
- `glass_shatter`처럼 복수 index가 있는 클래스는 해당 index들의 max score를 사용
- 입력 shape가 `(521,)`가 아니면 `ValueError` 발생
- `override_threshold()`로 모든 클래스 threshold를 일괄 변경 가능

### `src/postprocess/trigger.py`

threshold와 cooldown으로 위험 이벤트를 판정합니다.

주요 내용:

- `TriggerEvent` dataclass는 `key`, `display_name`, `score`, `timestamp`를 가짐
- `Trigger.evaluate()`는 score가 클래스 threshold 이상이고 마지막 트리거 이후 cooldown이 지났을 때 이벤트 생성
- 클래스별 마지막 트리거 시각을 `_last_trigger`에 저장
- M1은 단순 threshold + cooldown만 사용하며, debounce는 이후 단계로 남아 있음

### `src/preprocess/noise_suppress.py`

노이즈 억제 전처리 placeholder입니다.

현재 상태:

- `suppress(audio_16k_mono, aggressiveness=1)` 함수가 존재
- M1에서는 입력 오디오를 그대로 반환
- M2에서 WebRTC NS 또는 유사 노이즈 억제 방식 통합 예정

### `src/embedded/uart_sender.py`

임베디드 UART 송신 placeholder입니다.

현재 상태:

- `UARTSender` 클래스 구조만 있음
- 생성자, `send_event()`, `send_heartbeat()`, `close()` 모두 `NotImplementedError`
- M5에서 `pyserial` 기반 JSON line 프로토콜 구현 예정

### `__init__.py` 파일들

각 디렉터리를 Python 패키지로 인식시키기 위한 파일입니다. 대부분 패키지 설명 주석만 포함합니다.

## 6. 검증 스크립트와 테스트

### `scripts/verify_inference.py`

설치와 YAMNet 추론 환경을 확인하는 스크립트입니다.

실행 예:

```powershell
python scripts\verify_inference.py
python scripts\verify_inference.py --file data\sample\test.wav
```

주요 동작:

- TF-Hub에서 YAMNet 로드
- 입력이 없으면 0.96초 길이의 zero dummy audio 생성
- `--file`이 있으면 오디오 파일을 16kHz mono로 로드
- YAMNet 추론 실행
- `scores` shape가 `(N, 521)`인지 확인
- `embeddings` shape가 `(N, 1024)`인지 확인
- 위험 클래스 index들의 평균 score 출력
- 성공 시 `PASS: YAMNet inference OK` 출력

### `tests/test_yamnet_load.py`

pytest 기반 테스트 파일입니다.

테스트 범위:

- YAMNet 로드 성공 여부
- scores shape가 `(N, 521)`인지 확인
- embeddings shape가 `(N, 1024)`인지 확인
- 평균 score shape가 `(521,)`인지 확인
- score 값 범위가 `[0, 1]`인지 확인
- `DangerFilter.extract()`가 12종 key를 모두 반환하는지 확인
- `glass_shatter`가 index 435, 437 중 max 값을 쓰는지 확인
- score shape 오류 시 `ValueError`가 나는지 확인
- threshold override 동작 확인
- cooldown 내 동일 클래스 재트리거 억제 확인
- cooldown 경과 후 재트리거 허용 확인

TensorFlow 또는 TensorFlow Hub가 설치되어 있지 않으면 YAMNet 관련 테스트는 skip됩니다. `DangerFilter`와 `Trigger` 테스트는 네트워크 없이 실행됩니다.

## 7. 문서 파일 요약

| 파일 | 내용 |
|---|---|
| `docs/development-plan.md` | 전체 개발 계획서입니다. 프로젝트 목표, KPI, 데이터셋 전략, 노이즈 캔슬링, 아키텍처, 임베디드 연동, M1~M6 마일스톤, 평가 방법과 리스크를 설명합니다. |
| `docs/m1-initial-model-spec.md` | M1 베이스라인 상세 스펙입니다. Scope/Non-Scope, 입력 사양, YAMNet 모델 구성, 위험 클래스 화이트리스트, 출력 포맷, CLI 사양, 검증 기준을 정의합니다. |
| `docs/mic-quickstart.md` | 노트북 마이크 실시간 분석 가이드입니다. 가상환경, 의존성 설치, YAMNet 검증, 마이크 장치 확인, 실시간 실행, threshold 튜닝, 트러블슈팅을 다룹니다. |
| `docs/project-file-summary.md` | 현재 문서입니다. 프로젝트 관련 파일별 역할과 내용을 요약합니다. |

## 8. 보조 디렉터리와 메타 파일

| 경로 | 내용 |
|---|---|
| `data/sample/` | 사용자가 직접 검증용 오디오를 넣는 위치입니다. 현재 저장소에는 샘플 오디오가 포함되어 있지 않습니다. |
| `experiments/` | 향후 실험 결과나 분석 산출물을 저장할 디렉터리입니다. |
| `.github/PR_BODY.md` | M1 베이스라인 구현 내용을 설명하는 PR 본문 초안입니다. |
| `.claude/agents/` | Claude Code용 서브에이전트 정의 파일들이 있습니다. 모델 개발, 계획, PR 관리 역할을 분리합니다. |
| `.claude/settings.local.json` | 로컬 Claude Code 권한 설정입니다. 사용자 환경에 종속될 수 있습니다. |

## 9. 실행 및 검증 흐름

처음 설치 후 권장 확인 순서:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
python scripts\verify_inference.py
python -m pytest tests -v
python -m src.cli --input mic --threshold 0.4 --verbose
```

파일 입력 확인:

```powershell
python -m src.cli --input data\sample\test.wav --threshold 0.5 --verbose
```

마이크 장치 확인:

```powershell
python -c "import sounddevice; print(sounddevice.query_devices())"
```

특정 마이크 사용:

```powershell
python -m src.cli --input mic --device 1 --threshold 0.4
```

## 10. 문서와 코드 간 주의사항

- `docs/mic-quickstart.md`에는 `--device` 옵션이 아직 CLI에 노출되지 않았을 수 있다는 표현이 있으나, 현재 `src/cli.py`에는 `--device`가 구현되어 있습니다.
- `docs/mic-quickstart.md`의 cooldown 설명에는 `cooldown_seconds`라는 이름이 나오지만, 실제 `config/whitelist.yaml` 필드명은 `cooldown_sec`입니다.
- `requirements.txt`는 Python 3.11 기준 TensorFlow 2.13~2.15 범위를 사용합니다. Python 3.12 가상환경에서는 해당 TensorFlow 범위가 설치되지 않으므로 `.venv`를 Python 3.11로 다시 만들어야 합니다.
- 저장소에는 오디오 샘플과 모델 캐시가 포함되어 있지 않습니다. 최초 YAMNet 실행에는 인터넷 또는 사전 캐시가 필요합니다.
