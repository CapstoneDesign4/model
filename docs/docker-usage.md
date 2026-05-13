# Docker 사용 가이드

## 1. 이 문서를 봐야 하는 사람

Python / TensorFlow / PortAudio 환경을 직접 구성하지 않고 **WAV 파일 분석**과 **pytest**를 바로 실행하고 싶은 팀원 또는 검토자를 위한 문서입니다.

단, **마이크 실시간 분석(`--input mic`)은 Docker에서 지원하지 않습니다.** 마이크 모드가 필요하면 `README.md` 상단의 호스트 venv 워크플로를 사용하세요.

---

## 2. 사전 요구사항

| 항목 | 내용 |
|---|---|
| Docker Desktop (Windows/macOS) 또는 Docker Engine (Linux) | [docker.com/get-started](https://www.docker.com/get-started) |
| 디스크 여유 공간 | 약 5GB (이미지 약 3.77GB + 빌드 캐시) |
| WSL2 백엔드 | Windows 사용자에게 권장. Docker Desktop 설정에서 활성화 |
| 인터넷 연결 | 최초 빌드 시 TensorFlow (~600MB), YAMNet (~17MB) 다운로드 |

Docker 설치 여부 확인:

```powershell
docker --version
```

---

## 3. 빠른 시작 (TL;DR)

**Windows PowerShell:**

```powershell
# 1. 이미지 빌드 (최초 1회, 10~15분 소요)
docker build -t yamnet-danger:latest .

# 2. 출력 폴더 준비
mkdir output -ErrorAction SilentlyContinue

# 3. WAV 파일 분석
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  -v "${PWD}/output:/app/output" `
  yamnet-danger:latest `
  --input data/sample/test.wav --threshold 0.5 --verbose
```

**macOS/Linux bash:**

```bash
# 1. 이미지 빌드
docker build -t yamnet-danger:latest .

# 2. 출력 폴더 준비
mkdir -p output

# 3. WAV 파일 분석
docker run --rm \
  -v "$PWD/data/sample:/app/data/sample:ro" \
  -v "$PWD/output:/app/output" \
  yamnet-danger:latest \
  --input data/sample/test.wav --threshold 0.5 --verbose
```

---

## 4. 이미지 빌드

```powershell
docker build -t yamnet-danger:latest .
```

**빌드 단계별 요약:**

1. `python:3.11-slim` 베이스 이미지 준비
2. `apt-get install`: libsndfile1, ffmpeg, libportaudio2 설치
3. `pip install -r requirements.txt`: TensorFlow, librosa 등 설치 (가장 오래 걸림)
4. `python scripts/verify_inference.py`: TF-Hub에서 YAMNet 모델 다운로드 후 이미지 내부에 캐시
5. `COPY src/ config/ tests/`: 소스 코드 복사

**빌드 시간:**

- 최초 빌드: 10~15분 (TF ~600MB + YAMNet ~17MB 다운로드 포함)
- 이후 재빌드: 소스 코드만 바뀐 경우 수초~수십초 (레이어 캐시 활용)
- `requirements.txt`가 바뀐 경우: pip 레이어부터 재실행

**빌드 컨텍스트 크기 확인:**

`.dockerignore` 적용 결과 빌드 컨텍스트는 약 8.1MB 수준입니다. 빌드 시작 시 `Sending build context to Docker daemon` 메시지에서 크기를 확인할 수 있습니다. 이 값이 수십 MB 이상이면 `.dockerignore` 설정을 점검하세요.

---

## 5. WAV 파일 분석

**사전 준비:** 분석할 WAV 파일을 `data/sample/` 폴더에 넣고, `output/` 폴더를 호스트에 미리 생성합니다.

**Windows PowerShell:**

```powershell
mkdir output -ErrorAction SilentlyContinue
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  -v "${PWD}/output:/app/output" `
  yamnet-danger:latest `
  --input data/sample/test.wav --threshold 0.5 --verbose
```

**macOS/Linux bash:**

```bash
mkdir -p output
docker run --rm \
  -v "$PWD/data/sample:/app/data/sample:ro" \
  -v "$PWD/output:/app/output" \
  yamnet-danger:latest \
  --input data/sample/test.wav --threshold 0.5 --verbose
```

**각 옵션 의미:**

| 옵션 | 의미 |
|---|---|
| `--rm` | 컨테이너 종료 후 자동 삭제 |
| `-v .../data/sample:ro` | 입력 WAV 파일 읽기 전용 마운트 |
| `-v .../output` | 분석 결과(JSONL 등)를 호스트에 영구 저장 |
| `--threshold 0.5` | 위험 소리 트리거 임계값 |
| `--verbose` | 매 윈도우의 12종 클래스 점수 출력 |

---

## 6. 동작 확인 — 강제 트리거

파이프라인이 정상 동작하는지 확인하려면 `--threshold 0.0` 으로 임계값을 낮춰 모든 입력에 대해 트리거를 강제로 발생시킵니다.

```powershell
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  yamnet-danger:latest `
  --input data/sample/test.wav --threshold 0.0
```

---

## 7. 결과 로그 저장 (`--log`)

`--log output/run.jsonl` 옵션을 추가하면 클래스별 점수와 debounce votes를 포함한 이벤트가 JSONL 형식으로 누적됩니다.

```powershell
docker run --rm `
  -v "${PWD}/data/sample:/app/data/sample:ro" `
  -v "${PWD}/output:/app/output" `
  yamnet-danger:latest `
  --input data/sample/test.wav --threshold 0.4 --log output/run.jsonl --verbose
```

컨테이너 종료 후 호스트의 `output/run.jsonl` 파일에서 바로 확인할 수 있습니다.

---

## 8. 테스트 실행

YAMNet 로딩이 필요 없는 테스트만 (빠름):

```powershell
docker run --rm --entrypoint pytest yamnet-danger:latest tests/ -v -k "not yamnet"
```

전체 테스트 (YAMNet 로딩 포함, 이미지 내부 캐시를 사용하므로 네트워크 없이 동작):

```powershell
docker run --rm --entrypoint pytest yamnet-danger:latest tests/ -v
```

---

## 9. 자주 쓰는 옵션 요약

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input <file\|mic>` | 필수 | WAV 파일 경로 또는 `mic` (Docker에서 mic 불가) |
| `--threshold <float>` | `0.5` | 위험 소리 트리거 임계값 (0.0~1.0) |
| `--debounce-window <int>` | `3` | debounce 슬라이딩 윈도우 크기 N |
| `--debounce-k <int>` | `2` | 트리거 다수결 임계 K (K/N) |
| `--no-debounce` | 꺼짐 | debounce 비활성화 플래그 |
| `--log <path>` | 없음 | JSONL 로그 저장 경로 |
| `--verbose` | 꺼짐 | 매 윈도우의 클래스별 점수 및 votes 출력 |

전체 옵션 확인:

```powershell
docker run --rm yamnet-danger:latest --help
```

---

## 10. 마이크 모드는 왜 지원하지 않나

macOS와 Windows의 Docker Desktop은 호스트 오디오 디바이스에 직접 접근하는 경로를 공식 지원하지 않습니다. Linux에서는 `/dev/snd` 바인드 마운트로 실험적 접근이 가능하지만, 호스트 오디오 서브시스템(ALSA/PulseAudio/PipeWire)과의 충돌 위험이 있어 권장하지 않습니다.

**권장:** 마이크 모드는 호스트 venv 환경에서 실행하세요. `README.md`의 venv 설치 절차를 참고하십시오.

Linux 사용자의 실험적 시도 (동작 보장 없음):

```bash
docker run --rm -it --device /dev/snd yamnet-danger:latest --input mic --threshold 0.4
```

---

## 11. 트러블슈팅

| 증상 | 해결 |
|---|---|
| 첫 빌드가 TF 다운로드에서 오래 멈춘 듯 보임 | 정상입니다. TF ~600MB 다운로드 중이므로 기다리세요. `docker build` 진행 표시를 확인하세요. |
| `Error: cannot find -lsndfile` 또는 `OSError: cannot load library` | 이미지를 다시 빌드하세요: `docker build --no-cache -t yamnet-danger:latest .` |
| `permission denied` (`output/` 관련, Linux) | `chmod 777 output` 또는 `docker run` 에 `--user $(id -u):$(id -g)` 옵션 추가 |
| Windows에서 볼륨 마운트 실패 | 경로에 한글/공백이 있으면 전체를 따옴표로 감싸세요. Docker Desktop > Settings > Resources > File Sharing에서 `C:` 드라이브 공유 여부를 확인하고, WSL2 백엔드 활성화를 권장합니다. |
| `output/` 디렉터리 관련 오류 (Windows) | Docker가 없는 디렉터리를 root 소유로 생성할 수 있습니다. `mkdir output -ErrorAction SilentlyContinue` 로 미리 생성하세요. |
| sounddevice / portaudio ImportError | libportaudio2가 이미지에 포함되어 있어 정상적으로 임포트되어야 합니다. 발생 시 GitHub 이슈로 제보해 주세요. |
| 이미지 크기가 너무 큼 | TensorFlow 1GB+ 포함으로 정상입니다 (~3.77GB). 팀 공유 시 `docker save` 또는 컨테이너 레지스트리(ghcr.io) 활용을 검토하세요. |
| `data/sample/` 비어있음 오류 | 분석할 WAV 파일을 `data/sample/` 폴더에 직접 넣은 뒤 다시 실행하세요. |

---

## 12. 이미지 정리

현재 이미지 목록 확인:

```powershell
docker images yamnet-danger
```

이미지 삭제:

```powershell
docker rmi yamnet-danger:latest
```

---

## 13. 다음 단계 (참고용)

아래 항목은 현재 범위에 포함되지 않으며, 향후 별도 작업으로 검토됩니다.

- **GitHub Container Registry 배포**: 빌드된 이미지를 `ghcr.io`에 push하여 팀원이 `docker pull`만으로 사용할 수 있도록 구성 (별도 CI 작업)
- **docker-compose 도입**: 현재는 단일 컨테이너로 충분하지만, MQTT 브로커 등 사이드카 서비스가 추가될 경우 재검토
