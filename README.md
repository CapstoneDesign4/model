# CapstoneDesign Model 설치 튜토리얼

YAMNet 기반 위험 소리 감지 모델을 처음 설치하고 실행하는 절차입니다. 현재 구현은 M1 베이스라인으로, WAV 파일 또는 마이크 입력을 받아 12종 위험 소리 점수를 출력하고 임계값을 넘으면 이벤트를 기록합니다.

## 1. 프로젝트 구조

```text
model/
├── requirements.txt              # Python 의존성
├── config/
│   └── whitelist.yaml            # 위험 소리 클래스, threshold, cooldown 설정
├── src/
│   ├── cli.py                    # 실행 진입점: python -m src.cli
│   ├── audio_io/                 # 파일/마이크 입력
│   ├── model/                    # YAMNet 로드 및 위험 클래스 필터링
│   └── postprocess/              # threshold + cooldown 트리거
├── scripts/
│   └── verify_inference.py       # 설치 및 YAMNet 추론 검증
├── tests/
│   └── test_yamnet_load.py       # 단위 테스트
├── data/
│   └── sample/                   # 사용자가 직접 샘플 오디오를 넣는 위치
└── docs/                         # 설계/실행 참고 문서
```

## 2. 사전 준비

필수:

- Python 3.11
- pip, venv
- 인터넷 연결: 최초 실행 시 TensorFlow Hub에서 YAMNet 모델을 다운로드합니다.
- 마이크 실행 시 OS에서 입력 장치가 정상 인식되어야 합니다.

주의:

- 현재 `requirements.txt`는 Python 3.11 기준으로 `tensorflow>=2.13,<2.16`을 사용합니다.
- Python 3.12로 만든 가상환경에서는 TensorFlow 2.13~2.15가 설치되지 않습니다.
- 이미 `.venv`를 Python 3.12로 만들었다면 삭제하고 Python 3.11로 다시 생성해야 합니다.
- 저장소에는 샘플 WAV와 YAMNet 모델 파일이 포함되어 있지 않습니다.

Linux 또는 Raspberry Pi에서 마이크를 사용할 경우 PortAudio가 필요할 수 있습니다.

```bash
sudo apt update
sudo apt install -y libportaudio2 libsndfile1
```

WAV 외 MP3/FLAC/OGG 같은 포맷을 자주 다룰 경우 `ffmpeg` 설치를 권장합니다.

```bash
sudo apt install -y ffmpeg
```

macOS에서는 다음을 사용할 수 있습니다.

```bash
brew install portaudio libsndfile ffmpeg
```

## 3. 저장소 받기

Git으로 받을 경우:

```powershell
git clone <저장소 URL>
cd model
```

ZIP으로 받은 경우 압축을 푼 뒤 프로젝트 루트로 이동합니다.

```powershell
cd C:\CapstoneDesign\model
```

## 4. Python 3.11 가상환경 생성

먼저 Python 3.11이 설치되어 있는지 확인합니다.

```powershell
py -3.11 --version
```

`Python 3.11.x`가 출력되어야 합니다. `py` 명령이 없거나 3.11이 없다면 Python 3.11을 설치한 뒤 새 터미널을 엽니다.

기존 `.venv`가 Python 3.12로 만들어져 있다면 먼저 제거합니다.

```powershell
deactivate
Remove-Item -Recurse -Force .\.venv
```

그 다음 Python 3.11로 가상환경을 새로 생성합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

마지막 명령의 출력이 `Python 3.11.x`인지 확인합니다.

`Activate.ps1` 실행이 막히면 PowerShell에서 한 번만 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

`py -3.11`이 동작하지 않지만 Python 3.11 실행 파일 경로를 알고 있다면 직접 지정할 수 있습니다.

```powershell
& "C:\Users\<사용자명>\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

가상환경이 활성화되면 프롬프트 앞에 `(.venv)`가 표시됩니다.

## 5. 의존성 설치

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치되는 주요 패키지는 다음과 같습니다.

```text
tensorflow>=2.13,<2.16
tensorflow-hub>=0.15
numpy>=1.24,<2.0
librosa>=0.10
soundfile>=0.12
scipy>=1.11
sounddevice>=0.4.6
pyyaml>=6.0
pytest>=7.4
```

TensorFlow 설치는 시간이 오래 걸릴 수 있습니다. 설치 후 Python과 pip가 같은 가상환경을 보고 있는지 확인합니다.

```powershell
python --version
python -m pip --version
```

## 6. YAMNet 환경 검증

다음 명령은 더미 오디오로 YAMNet 로드와 추론을 확인합니다.

```powershell
python scripts\verify_inference.py
```

Linux/macOS:

```bash
python scripts/verify_inference.py
```

최초 실행 시 `https://tfhub.dev/google/yamnet/1` 모델을 다운로드합니다. 성공하면 마지막에 다음 문구가 출력됩니다.

```text
PASS: YAMNet inference OK
```

샘플 WAV 파일이 있다면 파일 입력도 검증할 수 있습니다.

```powershell
python scripts\verify_inference.py --file data\sample\test.wav
```

오프라인 또는 사내망 환경에서는 YAMNet 캐시 위치를 지정할 수 있습니다.

```powershell
$env:TFHUB_CACHE_DIR="C:\CapstoneDesign\tfhub_modules"
python scripts\verify_inference.py
```

Linux/macOS:

```bash
export TFHUB_CACHE_DIR="$HOME/tfhub_modules"
python scripts/verify_inference.py
```

## 7. 실행 방법

CLI 진입점은 다음입니다.

```powershell
python -m src.cli --help
```

### WAV 파일 분석

`data/sample/` 폴더에 사용자가 직접 WAV 파일을 넣습니다. 저장소에는 샘플 오디오가 포함되어 있지 않습니다.

```powershell
python -m src.cli --input data\sample\test.wav --threshold 0.5 --verbose
```

위험 소리를 강제로 잘 감지하는지 흐름만 보고 싶다면 threshold를 낮출 수 있습니다.

```powershell
python -m src.cli --input data\sample\test.wav --threshold 0.0
```

### 마이크 실시간 분석

먼저 입력 장치 목록을 확인합니다.

```powershell
python -c "import sounddevice; print(sounddevice.query_devices())"
```

기본 마이크로 실행:

```powershell
python -m src.cli --input mic --threshold 0.4 --verbose
```

특정 마이크 장치를 지정:

```powershell
python -m src.cli --input mic --device 1 --threshold 0.4
```

트리거 이벤트를 JSONL 파일로 저장:

```powershell
python -m src.cli --input mic --threshold 0.4 --log output\run.jsonl
```

종료는 `Ctrl+C`입니다.

### 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--input` | 필수 | `mic` 또는 WAV 파일 경로 |
| `--threshold` | `0.5` | 전체 위험 클래스 공통 임계값 |
| `--config` | `config/whitelist.yaml` | 위험 클래스 설정 파일 |
| `--hop` | `0.48` | 윈도우 hop 길이, 초 단위 |
| `--verbose` | 꺼짐 | 매 윈도우의 12종 score 출력 |
| `--log` | 없음 | JSONL 로그 저장 경로 |
| `--device` | 기본 장치 | 마이크 장치 인덱스 |

## 8. 설정 파일

위험 소리 클래스는 `config/whitelist.yaml`에서 관리합니다.

현재 포함된 클래스:

```text
screaming
baby_cry
glass_shatter
breaking
gunshot
explosion
fire_alarm
smoke_alarm
siren
civil_defense_siren
car_alarm
vehicle_horn
```

각 항목은 YAMNet 클래스 인덱스, threshold, cooldown을 가집니다. 별도 설정 파일을 만들어 실행할 수도 있습니다.

```powershell
python -m src.cli --input mic --config config\whitelist_home.yaml
```

CLI의 `--threshold`를 사용하면 설정 파일의 threshold가 전체 클래스에 대해 일괄 오버라이드됩니다.

## 9. 테스트

설치 후 단위 테스트를 실행합니다.

```powershell
python -m pytest tests -v
```

주의:

- `DangerFilter`, `Trigger` 테스트는 네트워크 없이 실행됩니다.
- YAMNet 로드 테스트는 TensorFlow, TensorFlow Hub, 모델 캐시 또는 네트워크가 필요합니다.
- 의존성이 설치되어 있지 않으면 일부 테스트는 skip되거나 실패할 수 있습니다.

## 10. 문제 해결

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: tensorflow` | 가상환경 활성화 후 `python -m pip install -r requirements.txt`를 다시 실행합니다. |
| TensorFlow 설치 실패 | `python --version`이 `Python 3.11.x`인지 확인합니다. Python 3.12 가상환경이면 `.venv`를 삭제하고 `py -3.11 -m venv .venv`로 다시 만듭니다. |
| `Activate.ps1` 실행 차단 | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`를 한 번 실행합니다. |
| YAMNet 다운로드 실패 | 인터넷 연결을 확인하거나 `TFHUB_CACHE_DIR`를 지정해 사전 캐시된 모델을 사용합니다. |
| `OSError: PortAudio library not found` | Linux/Raspberry Pi에서는 `sudo apt install libportaudio2`를 설치합니다. |
| 마이크 입력이 무음 | OS 사운드 설정에서 입력 장치와 입력 볼륨을 확인하고 `sounddevice.query_devices()`로 장치 인덱스를 확인합니다. |
| `sounddevice status: input overflow` 경고 | 다른 무거운 프로그램을 종료하고 다시 실행합니다. 필요하면 `--hop` 값을 늘려 부하를 줄입니다. |
| WAV 파일을 찾을 수 없음 | `data/sample/`에 실제 오디오 파일을 넣고 경로를 다시 확인합니다. |

## 11. 현재 구현 범위

현재 M1에서 구현된 것:

- TF-Hub YAMNet 로드
- WAV 파일 입력
- 마이크 실시간 입력
- 위험 클래스 12종 score 추출
- threshold + cooldown 기반 트리거
- 콘솔 출력 및 JSONL 로그 저장

아직 구현되지 않았거나 이후 단계인 것:

- 실제 노이즈 캔슬링 전처리
- debounce 기반 다수결 후처리
- 경량 분류 헤드 학습
- TFLite 변환
- UART/Serial 임베디드 연동

추가 참고 문서:

- `docs/m1-initial-model-spec.md`
- `docs/mic-quickstart.md`
- `docs/development-plan.md`
