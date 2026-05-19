# Docker 도입 계획서: YAMNet 위험 소리 감지 시스템

> 버전: v0.1 (2026-05-14)
> 참조: `CLAUDE.md`, `docs/development-plan.md`, `requirements.txt`

---

## 1. 목표와 비목표

### 1.1 목표 (Docker로 해결할 것)

| 항목 | 내용 |
|---|---|
| 환경 재현성 | Python 3.11, TF 2.13~2.15, librosa, soundfile, scipy 등 모든 의존성을 고정 버전으로 캡슐화 |
| 단일 명령 실행 | `docker run ...` 한 줄로 WAV 파일 분석 및 pytest 실행 가능 |
| YAMNet 모델 캐시 | 빌드 타임에 TF-Hub로부터 YAMNet (~17MB)을 미리 다운로드하여 컨테이너 안에 포함 |
| OS 무관 파일 모드 | Windows / macOS / Linux 어느 환경에서도 WAV 파일 분석이 동일하게 동작 |
| CI 재사용 | 같은 이미지를 GitHub Actions에서 pytest 실행에 재사용 가능 |

### 1.2 비목표 (Docker로 해결하지 않을 것)

| 항목 | 결정 근거 |
|---|---|
| **마이크 모드 완전 지원** | sounddevice → PortAudio → 호스트 오디오 디바이스 접근이 필요. Linux에서는 `/dev/snd` 바인드나 PulseAudio over TCP로 우회 가능하나 설정 복잡도가 높고 macOS/Windows는 사실상 불가. 마이크 모드는 기존 venv 워크플로(`--input mic`)를 사용하도록 안내. |
| UART/직렬 포트 연동 | 임베디드 통신(M5)은 호스트 직렬 포트 접근(`/dev/ttyUSB0` 등)을 요구하므로 Docker 범위 밖. |
| GPU 가속 | TF-Hub YAMNet 추론은 CPU-only로도 충분. GPU 지원 이미지는 크기가 수 GB 증가하므로 포함하지 않음. |
| 학습(헤드 파인튜닝) | M3 헤드 학습은 Colab/GPU 머신 환경을 사용. 학습용 이미지는 별도 필요 시 정의. |

---

## 2. 이미지 전략

### 2.1 베이스 이미지 선택

**선택: `python:3.11-slim` (Debian Bookworm 기반)**

| 후보 | 크기(압축) | 비고 | 채택 여부 |
|---|---|---|---|
| `python:3.11` (full) | ~350MB | 불필요한 도구 포함, 크기 낭비 | 미채택 |
| `python:3.11-slim` | ~45MB | 필수 시스템 라이브러리만 포함, apt 추가 가능 | **채택** |
| `python:3.11-alpine` | ~10MB | musl libc — librosa/scipy의 C 확장 빌드 실패 위험 매우 높음 | 미채택 |
| `tensorflow/tensorflow:2.15.0` | ~1.2GB | TF 공식 이미지, 크기 과대 | 미채택 |

`python:3.11-slim`을 선택하는 이유: librosa, scipy, soundfile은 glibc 기반 C 확장을 사용하므로 Alpine은 위험하다. full 이미지는 불필요한 패키지가 많다. TF 공식 이미지는 이미 1GB 이상이라 추가 최적화 여지가 없다.

### 2.2 빌드 단계 전략

**단일스테이지 빌드 채택**

멀티스테이지를 고려했으나, TensorFlow 자체가 런타임 라이브러리(libc, libstdc++ 등)에 의존하므로 build stage에서 컴파일한 바이너리만 복사하는 방식이 효과가 제한적이다. pip 캐시 레이어 최적화만으로 충분한 효과를 얻을 수 있으므로 단일스테이지로 유지한다.

단, 추후 학습(M3)용 이미지가 필요해지면 `FROM yamnet-danger:inference AS base` 형태의 다단계 분리를 검토한다.

### 2.3 YAMNet 모델 빌드 타임 캐시

**채택: 빌드 타임에 `scripts/verify_inference.py` 실행하여 TF-Hub 모델 다운로드**

- TF-Hub는 기본적으로 `~/.cache/tfhub_modules/` (또는 `TFHUB_CACHE_DIR` 환경변수 경로)에 모델을 저장한다.
- Dockerfile 내에서 `RUN python scripts/verify_inference.py`를 실행하면 빌드 레이어에 YAMNet 파일이 포함된다.
- 이렇게 하면 컨테이너 첫 실행 시 네트워크 접근 없이 즉시 추론이 가능하다.
- 경로 통일을 위해 `ENV TFHUB_CACHE_DIR=/opt/tfhub_cache`로 명시적 고정을 권장한다.
- 빌드 시 네트워크 접근이 필요하지만, 이는 일회성이고 이후 모든 실행에서는 캐시가 재사용된다.

### 2.4 이미지 크기 추정

| 레이어 | 추정 크기 |
|---|---|
| python:3.11-slim 베이스 | ~45MB |
| 시스템 의존성 (libsndfile1, ffmpeg 등) | ~80MB |
| TensorFlow 2.15 + tensorflow-hub | ~850MB |
| numpy, scipy, librosa, soundfile | ~120MB |
| sounddevice (PortAudio, 실행 안 해도 설치됨) | ~5MB |
| pyyaml, pytest 등 | ~10MB |
| YAMNet TF-Hub 캐시 | ~17MB |
| 소스 코드 + 설정 | ~1MB |
| **합계 (압축 전, 추정)** | **~1.1GB** |

> 실측: 2026-05-14 빌드 기준 `docker image ls` 가 표시하는 디스크 점유량은 **약 3.77GB**. 위 표는 패키지 자체 크기 합산 추정이고, 실제 이미지에는 apt 시스템 라이브러리(libsndfile1/ffmpeg/libportaudio2 등 약 466MB)와 베이스 OS, 빌드 산출물이 더해진다. 팀원 안내 시 **~5GB 디스크 여유 권장**.

TF 자체가 1GB에 육박하므로 이미지 크기를 극적으로 줄이는 것은 불가능하다. 이 점을 팀원에게 미리 안내해야 한다. 첫 `docker pull` 또는 `docker build`에 시간이 걸리지만 이후 레이어 캐시로 재빌드는 빠르다.

### 2.5 레이어 최적화 원칙

1. `pip install` 명령을 `requirements.txt` 기반 단일 `RUN`으로 묶어 레이어 수를 최소화한다.
2. `--no-cache-dir` 옵션을 사용해 pip 다운로드 캐시를 레이어에 남기지 않는다.
3. `apt-get` 설치 후 `/var/lib/apt/lists/*` 삭제를 동일 `RUN` 블록 안에서 수행한다.
4. `.dockerignore`로 불필요한 파일이 빌드 컨텍스트에 들어가지 않도록 한다.

### 2.6 .dockerignore 포함 권장 목록

아래 항목은 `.dockerignore`에 반드시 포함해야 한다. 누락 시 빌드 컨텍스트 크기가 수백 MB 이상 증가할 수 있다.

```
# 가상환경 (수백 MB)
.venv/
venv/
env/

# Python 바이트코드
__pycache__/
*.pyc
*.pyo

# 데이터 파일 (WAV 등 대용량)
data/raw/
data/processed/
*.wav
*.mp3
*.flac

# 모델 가중치 (이미지에 넣지 않음, 런타임 다운로드 또는 볼륨 마운트)
*.tflite
*.h5
*.keras
checkpoints/
tfhub_modules/

# 출력/실험 로그
output/
experiments/
*.log
*.jsonl

# OS / IDE 파일
.DS_Store
Thumbs.db
.vscode/
.idea/

# Git
.git/

# pytest 캐시
.pytest_cache/

# 빌드 산출물
dist/
build/
*.egg-info/
```

**주의**: `data/sample/` 디렉터리는 `.gitignore`에서 예외(`!data/sample/`)로 허용되어 있다. `.dockerignore`에서도 `data/sample/`을 명시적으로 허용해야 검증용 WAV 파일이 컨테이너 이미지에 포함된다. 단, 샘플 파일이 크다면 포함하지 않고 볼륨 마운트로 제공하는 방식이 더 유연하다.

---

## 3. 실행 시나리오별 권장 명령어

이하 모든 명령어는 사용자가 프로젝트 루트 디렉터리에서 실행하는 것을 기준으로 한다.

### 3.1 이미지 빌드

```bash
docker build -t yamnet-danger:latest .
```

빌드 시 YAMNet 다운로드가 포함되므로 첫 빌드는 네트워크 속도에 따라 5~15분 소요된다. 이후 재빌드는 레이어 캐시로 수분 내 완료된다.

### 3.2 파일 모드 — WAV 파일 분석

```bash
# Linux / macOS
docker run --rm \
  -v "$(pwd)/data/sample:/app/data/sample:ro" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/config:/app/config:ro" \
  yamnet-danger:latest \
  python -m src.cli --input data/sample/test.wav --threshold 0.5 --verbose

# Windows PowerShell
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  -v "${PWD}/output:/app/output" `
  -v "${PWD}/config:/app/config:ro" `
  yamnet-danger:latest `
  python -m src.cli --input data/sample/test.wav --threshold 0.5 --verbose
```

### 3.3 임계값 0 강제 트리거 (동작 확인용)

```bash
# Linux / macOS
docker run --rm \
  -v "$(pwd)/data/sample:/app/data/sample:ro" \
  yamnet-danger:latest \
  python -m src.cli --input data/sample/test.wav --threshold 0.0

# Windows PowerShell
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  yamnet-danger:latest `
  python -m src.cli --input data/sample/test.wav --threshold 0.0
```

### 3.4 YAMNet 로딩 검증

```bash
docker run --rm yamnet-danger:latest python scripts/verify_inference.py
```

네트워크 없이 실행 가능 (빌드 타임에 캐시된 모델 사용).

### 3.5 테스트 실행 (pytest)

```bash
# 네트워크 불필요 테스트만 (YAMNet 로드 없음)
docker run --rm yamnet-danger:latest pytest tests/ -v

# YAMNet 로딩 포함 테스트 (빌드 타임 캐시 덕분에 네트워크 불필요)
docker run --rm yamnet-danger:latest pytest tests/ -v -k "yamnet"
```

### 3.6 로그 출력과 함께 파일 분석

```bash
# Linux / macOS
docker run --rm \
  -v "$(pwd)/data/sample:/app/data/sample:ro" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/config:/app/config:ro" \
  yamnet-danger:latest \
  python -m src.cli \
    --input data/sample/test.wav \
    --threshold 0.4 \
    --log output/run.jsonl \
    --verbose

# Windows PowerShell
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  -v "${PWD}/output:/app/output" `
  -v "${PWD}/config:/app/config:ro" `
  yamnet-danger:latest `
  python -m src.cli `
    --input data/sample/test.wav `
    --threshold 0.4 `
    --log output/run.jsonl `
    --verbose
```

### 3.7 마이크 모드 (`--input mic`) — OS별 대응

#### Linux (제한적 지원)

Linux에서는 `/dev/snd` 디바이스 바인드 또는 PulseAudio over TCP를 통해 우회가 가능하다. 단, 호스트 오디오 설정(ALSA/PulseAudio/PipeWire 혼용)에 따라 실패할 수 있으므로 "실험적 지원" 수준으로 안내한다.

```bash
# Linux — /dev/snd 바인드 방식 (ALSA 직접 접근)
docker run --rm \
  --device /dev/snd \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/config:/app/config:ro" \
  yamnet-danger:latest \
  python -m src.cli --input mic --threshold 0.4

# Linux — PulseAudio over TCP 방식 (더 안정적이나 설정 필요)
# 호스트에서 먼저: pactl load-module module-native-protocol-tcp auth-anonymous=1
docker run --rm \
  -e PULSE_SERVER=tcp:host.docker.internal:4713 \
  -v "$(pwd)/output:/app/output" \
  yamnet-danger:latest \
  python -m src.cli --input mic --threshold 0.4
```

#### macOS

macOS에서는 Docker Desktop이 호스트 오디오 디바이스를 컨테이너에 직접 노출하는 방법을 공식 지원하지 않는다. PulseAudio 설치 후 over TCP 방식이 이론적으로 가능하나 설정 복잡도가 매우 높고 안정성을 보장하기 어렵다.

**권장**: macOS에서 마이크 모드는 venv 워크플로를 사용한다.

```powershell
# macOS에서 마이크 모드 — Docker 대신 venv 사용
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli --input mic --threshold 0.4
```

#### Windows

Windows에서는 Docker Desktop(WSL2 백엔드)이 호스트 오디오 디바이스를 컨테이너에 전달하는 공식 경로가 없다. WSL2에서 PulseAudio 설정이 가능하다는 비공식 방법이 존재하지만 Docker Desktop 버전과 WSL2 배포판에 따라 동작 여부가 달라 재현성을 보장할 수 없다.

**권장**: Windows에서 마이크 모드는 venv 워크플로를 사용한다.

```powershell
# Windows에서 마이크 모드 — Docker 대신 venv 사용
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.cli --input mic --threshold 0.4
```

---

## 4. docker-compose.yml 도입 여부

**결정: 도입하지 않음 (현재 단계)**

근거:
- 현재 이 프로젝트에서 Docker로 실행하는 시나리오는 `app`(파일 분석), `test`(pytest) 두 가지이다.
- 두 서비스 모두 단일 컨테이너이고 상호 의존성이 없다. 데이터베이스나 메시지 브로커 같은 사이드카 서비스가 필요 없다.
- docker-compose.yml을 도입하면 볼륨 경로를 미리 고정해야 하는데, 팀원마다 로컬 데이터 경로가 달라 오히려 혼란을 초래할 수 있다.
- 명령어가 길어지는 문제는 `Makefile`(Linux/macOS) 또는 PowerShell 별칭으로 해결하는 것이 더 가볍다.

추후 재검토 조건: MQTT 브로커(`mosquitto`)나 학습 데이터 서버 등 사이드카 서비스가 추가되면 그 시점에 도입을 재고한다.

---

## 5. 데이터/출력/로그 볼륨 매핑

| 경로 | 마운트 여부 | 접근 권한 | 근거 |
|---|---|---|---|
| `data/sample/` | 마운트 (권장) | `:ro` (읽기 전용) | WAV 파일은 이미지에 포함하지 않고 호스트에서 주입. `data/raw/`, `data/processed/`는 대용량이므로 제외. |
| `output/` | 마운트 (권장) | `:rw` (읽기/쓰기) | `--log output/run.jsonl` 사용 시 컨테이너 밖으로 로그를 남기기 위해 필요. 마운트 안 하면 컨테이너 종료 시 소멸. |
| `config/` | 마운트 (선택) | `:ro` | `whitelist.yaml` 임계값 또는 debounce 설정 변경 실험 시 유용. 기본값으로 실행하면 생략 가능. |
| `data/raw/`, `data/processed/` | 마운트 안 함 | - | 용량이 크고 CI/검증 목적에 불필요. 학습(M3) 시는 별도 논의. |
| `tfhub_modules/` | 마운트 안 함 | - | 빌드 타임 캐시를 이미지 안에 포함하는 전략이므로, 런타임 볼륨 필요 없음. |

`output/` 디렉터리는 컨테이너 실행 전 호스트에 미리 생성해 두어야 한다. 없으면 Docker가 root 소유로 만들어 이후 호스트에서 파일 접근이 불편해질 수 있다.

```bash
mkdir -p output   # Linux/macOS
New-Item -ItemType Directory -Force output  # Windows PowerShell
```

---

## 6. CI 연동 가능성

같은 이미지를 GitHub Actions의 `container:` 키워드 또는 `docker run` 스텝에서 재사용하면, 로컬 빌드와 CI 환경의 Python/TF 버전 불일치로 인한 "로컬에서는 통과, CI에서는 실패" 문제를 제거할 수 있다. GitHub Container Registry(ghcr.io)에 이미지를 푸시해 두면 Actions에서 매번 재빌드 없이 바로 사용할 수 있어 파이프라인 시간을 단축할 수 있다.

---

## 7. 검증 체크리스트

빌드 후 아래 순서로 동작을 확인한다.

### 7.1 빌드

```bash
docker build -t yamnet-danger:latest .
```

확인 항목:
- [ ] 빌드가 오류 없이 완료됨
- [ ] `RUN python scripts/verify_inference.py` 단계에서 YAMNet 다운로드 로그가 출력됨
- [ ] 최종 이미지 크기 실측(`docker images yamnet-danger`) — 2026-05-14 기준 ~3.77GB

### 7.2 기본 동작 확인

```bash
# 1. YAMNet 로딩만 확인 (네트워크 없이)
docker run --rm yamnet-danger:latest python scripts/verify_inference.py
```

기대: `YAMNet 로드 성공` 등의 메시지, 오류 없이 종료.

```bash
# 2. 임계값 0 강제 트리거 (WAV 파일 필요)
docker run --rm \
  -v "$(pwd)/data/sample:/app/data/sample:ro" \
  yamnet-danger:latest \
  python -m src.cli --input data/sample/test.wav --threshold 0.0
```

기대: 모든 위험 클래스가 트리거되어 콘솔에 출력됨.

### 7.3 pytest 통과 확인

```bash
docker run --rm yamnet-danger:latest pytest tests/ -v
```

기대: `test_debounce_trigger.py` TC-1~TC-5 전체 통과.

### 7.4 로그 파일 생성 확인

```bash
mkdir -p output
docker run --rm \
  -v "$(pwd)/data/sample:/app/data/sample:ro" \
  -v "$(pwd)/output:/app/output" \
  yamnet-danger:latest \
  python -m src.cli --input data/sample/test.wav --threshold 0.0 --log output/run.jsonl
```

기대: `output/run.jsonl`이 호스트에 생성되고 JSON Lines 형식으로 이벤트가 기록됨.

### 7.5 config 오버라이드 확인 (선택)

```bash
docker run --rm \
  -v "$(pwd)/data/sample:/app/data/sample:ro" \
  -v "$(pwd)/config:/app/config:ro" \
  yamnet-danger:latest \
  python -m src.cli --input data/sample/test.wav --threshold 0.5 --verbose
```

기대: `--verbose` 출력에 whitelist 클래스별 score가 표시됨.

---

## 8. 위험 요소와 대응

### 8.1 이미지 크기 (실측 ~3.77GB)

**위험**: TensorFlow 단독으로 850MB 이상을 차지해 이미지가 크다. 팀원이 처음 `docker pull`하거나 빌드할 때 시간이 오래 걸릴 수 있다.

**대응**:
- 빌드 후 레이어 캐시가 유지되면 재빌드는 빠르다. `requirements.txt`가 바뀌지 않으면 pip 설치 레이어는 캐시 히트.
- GitHub Container Registry에 이미 빌드된 이미지를 올려두면 팀원은 `docker pull`만 하면 된다.
- TF 2.x CPU-only 휠을 사용하면 크기를 다소 줄일 수 있으나, `requirements.txt` 변경이 필요하고 추후 GPU 실험 시 다시 교체해야 하는 트레이드오프가 있다.

### 8.2 첫 빌드 시간

**위험**: TF 설치 + YAMNet 다운로드로 첫 빌드에 10분 이상 소요될 수 있다.

**대응**:
- `requirements.txt` 변경 전 소스 코드 COPY, 변경 후 pip install을 배치해 소스 변경 시 pip 레이어 재빌드를 피한다. (Dockerfile 레이어 순서 설계 책임은 모델 개발 에이전트)
- YAMNet 다운로드를 위한 `RUN python scripts/verify_inference.py` 레이어는 TF 설치 직후에 배치해 소스 변경 시 이 레이어가 무효화되지 않도록 한다.
- CI에서는 Docker layer cache를 actions/cache로 저장해 매 실행마다 전체 재빌드를 피한다.

### 8.3 마이크 모드 OS 비호환성

**위험**: `--input mic` 모드가 Docker에서 동작하지 않아 팀원이 혼란을 느낄 수 있다.

**대응**:
- README의 Docker 섹션에 "마이크 모드는 venv 워크플로를 사용하세요"라고 명시한다.
- `docker run ... --input mic` 실행 시 sounddevice가 오디오 디바이스를 찾지 못해 오류를 내뱉는다. 이 오류 메시지는 명확하므로 추가 처리 불필요.

### 8.4 .dockerignore 누락 위험

**위험**: `.venv/`(수백 MB), `data/`(WAV 파일 대용량), `output/`, `__pycache__/`가 빌드 컨텍스트에 포함되면 빌드가 수 분 이상 지연되고 이미지 크기도 불필요하게 커진다.

**대응**: `.dockerignore` 파일을 Dockerfile과 동시에 작성하며, §2.6에 정의한 목록을 빠짐없이 포함한다. 빌드 시 `Sending build context to Docker daemon` 크기가 1MB 이하인지 확인한다.

### 8.5 Windows 경로 마운트 주의사항

**위험**: PowerShell에서 `${PWD}`가 Windows 경로 형식(`C:\Users\...`)으로 확장되어 Docker에 전달될 수 있다. Docker Desktop은 내부적으로 WSL2를 통해 변환하지만, 경로에 공백이 있거나 네트워크 드라이브인 경우 실패할 수 있다.

**대응**:
- PowerShell에서는 `-v "${PWD}/...:/app/..."` 형태로 사용하며, 경로에 공백이 있으면 `"${PWD}/.../..."` 전체를 따옴표로 묶는다.
- Windows에서 실행이 안 될 경우 `docker run`에 `-v "C:/Users/user/model/data/sample:/app/data/sample"` 형태로 절대경로를 직접 입력한다.
- Docker Desktop의 Settings > Resources > File Sharing에 프로젝트 드라이브(예: `C:`)가 공유 설정되어 있어야 한다.

### 8.6 data/sample/ 처리 정책

**위험**: `.gitignore`는 `*.wav`를 제외하면서 `!data/sample/`로 예외를 두고 있다. `.dockerignore`에서 같은 정책을 명시하지 않으면 검증용 샘플이 빌드 컨텍스트에서 제외될 수 있다.

**대응**: `.dockerignore`에 `*.wav` 제외와 `!data/sample/*.wav` 허용을 순서대로 명시한다. 단, 샘플 파일을 이미지에 넣는 것보다 볼륨 마운트로 주입하는 방식이 이미지 크기와 유연성 면에서 낫다.

### 8.7 출력 디렉터리 권한 문제

**위험**: `output/`가 호스트에 없는 상태에서 `-v "${PWD}/output:/app/output"`을 사용하면 Docker가 해당 경로를 root 소유로 만들고, 이후 호스트에서 일반 사용자가 파일을 접근하기 어려워질 수 있다 (Linux 주로 해당).

**대응**: `docker run` 전에 `mkdir -p output`으로 호스트에서 먼저 생성하도록 README에 명시한다.

---

## 9. 다음 단계 — 모델 개발 에이전트가 만들 파일 목록

아래 파일들을 본 계획서를 기반으로 작성한다.

| 파일 | 내용 요약 | 우선순위 |
|---|---|---|
| `Dockerfile` | python:3.11-slim 베이스, 시스템 의존성 apt 설치, pip requirements, YAMNet 캐시 빌드 (`RUN python scripts/verify_inference.py`), `WORKDIR /app`, `ENTRYPOINT`/`CMD` 정의 | P0 |
| `.dockerignore` | §2.6의 목록 기반, `data/sample/*.wav` 예외 포함 여부 명시 | P0 |
| `README.md` Docker 섹션 | 빌드 명령, 파일 분석 명령, pytest 명령, 마이크 모드 불가 안내, Windows 주의사항 포함 | P1 |
| `docker-compose.yml` | 현재 단계에서는 불필요. MQTT 등 사이드카 추가 시 재검토. | 보류 |
| `.github/workflows/ci.yml` | (M3 이후) `docker build` + `docker run pytest`로 CI 구성. 현재는 선택 사항. | 보류 |

### Dockerfile 레이어 순서 지침 (모델 개발 에이전트 참고)

소스 코드 변경 시 pip 레이어 캐시가 무효화되지 않도록 아래 순서를 권장한다:

```
1. FROM python:3.11-slim
2. RUN apt-get install ... (시스템 의존성: libsndfile1, libportaudio2 등)
3. WORKDIR /app
4. COPY requirements.txt .
5. RUN pip install --no-cache-dir -r requirements.txt
6. COPY scripts/ scripts/          (YAMNet 캐시 다운로드에만 필요한 파일)
7. ENV TFHUB_CACHE_DIR=/opt/tfhub_cache
8. RUN python scripts/verify_inference.py   (YAMNet 모델 캐시)
9. COPY . .                         (소스 코드 전체 — 가장 자주 변경되므로 마지막)
10. CMD ["python", "-m", "src.cli", "--help"]
```

이 순서대로 레이어를 배치하면 소스 코드 수정 시 1~8 레이어는 캐시 히트되고 9번 이후만 재실행된다.

---

## 10. 결정 사항 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| 베이스 이미지 | `python:3.11-slim` | glibc 호환성 + 최소 크기 균형 |
| 빌드 단계 | 단일스테이지 | TF 런타임 의존성으로 인해 멀티스테이지 효과 미미 |
| YAMNet 캐시 | 빌드 타임 포함 (`ENV TFHUB_CACHE_DIR` + `RUN verify_inference.py`) | 첫 실행 시 네트워크 의존성 제거 |
| 마이크 모드 Docker 지원 | 비목표. macOS/Windows는 venv 워크플로 사용 | OS별 오디오 디바이스 접근 제약 |
| docker-compose.yml | 현재 단계 불필요 | 단일 컨테이너, 사이드카 서비스 없음 |
| 볼륨 마운트 | `data/sample`(ro), `output`(rw), `config`(ro, 선택) | 이미지 크기/유연성 균형 |

---

*문서 버전: v0.1 (2026-05-14 초안). Dockerfile 구현 후 실측 이미지 크기와 빌드 시간으로 수치 갱신 필요.*
