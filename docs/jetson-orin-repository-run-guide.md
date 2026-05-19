# Jetson Orin에서 이 저장소 실행하기

> 작성일: 2026-05-19  
> 대상: Jetson Orin 화면이 켜진 상태에서 이 저장소의 YAMNet 위험음 분류 코드와 ESP32 네트워크 입력 준비 코드를 실행하려는 팀원

---

## 1. 전체 순서

Jetson Orin에서 할 일은 아래 순서다.

1. Jetson 화면에서 터미널 열기
2. 시스템 패키지 설치
3. GitHub 저장소 받기
4. 실행할 브랜치 선택
5. Python 가상환경 만들기
6. Python 의존성 설치
7. YAMNet 모델 로딩 검증
8. 샘플 WAV 파일 추론
9. ESP32용 network receiver 실행
10. UDP sender로 network receiver 테스트

---

## 2. 터미널 열기

Jetson 화면에서 키보드로 아래를 누른다.

```text
Ctrl + Alt + T
```

터미널이 열리면 현재 위치를 확인한다.

```bash
pwd
```

원하는 작업 폴더로 이동한다. 예시는 홈 디렉터리 아래 `capstone` 폴더를 사용한다.

```bash
mkdir -p ~/capstone
cd ~/capstone
```

---

## 3. 시스템 패키지 설치

먼저 apt 패키지를 설치한다.

```bash
sudo apt update
sudo apt install -y \
  git \
  python3 \
  python3-pip \
  python3-venv \
  ffmpeg \
  libsndfile1 \
  libportaudio2
```

설치 확인:

```bash
git --version
python3 --version
python3 -m pip --version
```

---

## 4. 저장소 받기

아직 Jetson에 저장소가 없다면 clone한다. 저장소가 public이면 아래 명령은 username/password 없이 받아져야 한다.

```bash
cd ~/capstone
git clone https://github.com/CapstoneDesign4/model.git
cd model
```

### username/password를 물어보는 경우

GitHub에서 username/password를 물어본다면 보통 둘 중 하나다.

- 저장소가 private이라 인증이 필요함
- HTTPS 인증 캐시나 URL 문제로 GitHub가 로그인 시도를 요구함

GitHub는 일반 계정 비밀번호로 `git clone`하는 방식을 지원하지 않는다. private 저장소라면 비밀번호 대신 SSH key, GitHub CLI 로그인, 또는 Personal Access Token이 필요하다.

### 방법 A: public 저장소 ZIP으로 받기

저장소가 public이고 Git 로그인 없이 파일만 받으면 된다면 ZIP으로 받을 수 있다.

```bash
cd ~/capstone
wget -O model.zip https://github.com/CapstoneDesign4/model/archive/refs/heads/main.zip
unzip model.zip
mv model-main model
cd model
```

P0 브랜치가 아직 `main`에 merge되지 않았다면 브랜치 ZIP을 받는다.

```bash
cd ~/capstone
wget -O model-p0.zip https://github.com/CapstoneDesign4/model/archive/refs/heads/feat/esp32-jetson-p0-network.zip
unzip model-p0.zip
mv model-feat-esp32-jetson-p0-network model
cd model
```

이 방법은 Git 인증이 필요 없지만, 저장소가 private이면 ZIP 다운로드도 로그인 없이는 안 된다.

### 방법 B: SSH key로 받기

private 저장소라면 Jetson에서 SSH key를 만들고 GitHub에 한 번 등록하는 방법이 가장 깔끔하다. 이후부터는 username/password를 묻지 않는다.

Jetson에서 key 생성:

```bash
ssh-keygen -t ed25519 -C "jetson-orin"
```

계속 Enter를 눌러 기본 경로에 저장한다. 공개키를 출력한다.

```bash
cat ~/.ssh/id_ed25519.pub
```

출력된 한 줄을 GitHub에 등록한다.

- 개인 계정에 등록: GitHub → Settings → SSH and GPG keys → New SSH key
- 저장소 deploy key로 등록: repository → Settings → Deploy keys → Add deploy key

등록 후 연결 확인:

```bash
ssh -T git@github.com
```

그 다음 SSH 주소로 clone한다.

```bash
cd ~/capstone
git clone git@github.com:CapstoneDesign4/model.git
cd model
```

P0 브랜치를 쓰려면:

```bash
git checkout feat/esp32-jetson-p0-network
git pull
```

### 방법 C: GitHub CLI로 로그인해서 받기

키 등록이 번거롭다면 GitHub CLI의 device login을 사용할 수 있다.

```bash
sudo apt update
sudo apt install -y gh
gh auth login
```

화면에 나오는 device code를 브라우저에서 입력해 로그인한다. 이후:

```bash
cd ~/capstone
gh repo clone CapstoneDesign4/model
cd model
```

이미 받아둔 저장소가 있다면 그 폴더로 이동 후 최신 내용을 가져온다.

```bash
cd ~/capstone/model
git fetch --all --prune
```

---

## 5. 브랜치 선택

PR이 아직 merge되지 않았다면 ESP32/Jetson P0 작업 브랜치를 사용한다.

```bash
git checkout feat/esp32-jetson-p0-network
git pull
```

PR이 merge된 뒤라면 `main`을 사용한다.

```bash
git checkout main
git pull
```

현재 브랜치 확인:

```bash
git branch --show-current
git status
```

---

## 6. Python 가상환경 만들기

저장소 루트에서 가상환경을 만든다.

```bash
cd ~/capstone/model
python3 -m venv .venv
source .venv/bin/activate
```

가상환경이 켜졌는지 확인한다.

```bash
which python
python --version
python -m pip --version
```

터미널 앞에 `(.venv)`가 보이면 정상이다.

다음부터 새 터미널을 열 때는 저장소 폴더에서 다시 활성화한다.

```bash
cd ~/capstone/model
source .venv/bin/activate
```

---

## 7. TensorFlow Hub 캐시 위치 고정

YAMNet 모델은 처음 실행할 때 TensorFlow Hub에서 다운로드된다. 캐시 위치를 고정해두면 불완전 다운로드 문제를 추적하기 쉽다.

```bash
mkdir -p "$HOME/tfhub_modules"
export TFHUB_CACHE_DIR="$HOME/tfhub_modules"
```

매번 자동 적용하려면 `~/.bashrc`에 추가한다.

```bash
echo 'export TFHUB_CACHE_DIR="$HOME/tfhub_modules"' >> ~/.bashrc
source ~/.bashrc
```

확인:

```bash
python -c "import os; print(os.environ.get('TFHUB_CACHE_DIR'))"
```

---

## 8. Python 의존성 설치

pip을 먼저 업데이트한다.

```bash
python -m pip install --upgrade pip setuptools wheel
```

그 다음 저장소 의존성을 설치한다.

```bash
python -m pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import numpy; print('numpy ok')"
python -c "import librosa; print('librosa ok')"
python -c "import tensorflow as tf; print('tensorflow', tf.__version__)"
python -c "import tensorflow_hub as hub; print('tensorflow_hub ok')"
```

### TensorFlow 설치가 실패하는 경우

Jetson Orin은 JetPack/Python 버전에 따라 일반 `pip install tensorflow`가 실패할 수 있다. 이 경우 무리하게 계속 설치하지 말고 아래 순서로 확인한다.

```bash
python --version
uname -m
cat /etc/nv_tegra_release
```

확인할 것:

- `uname -m`이 `aarch64`인지
- JetPack/L4T 버전이 무엇인지
- 현재 Python 버전이 TensorFlow wheel과 맞는지

Jetson용 TensorFlow를 별도로 설치한 뒤, 나머지 패키지를 설치한다.

```bash
python -m pip install tensorflow-hub numpy librosa soundfile scipy sounddevice pyyaml pytest
```

그 다음 다시 확인한다.

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow_hub as hub; print('hub ok')"
```

---

## 9. YAMNet 로딩 검증

더미 오디오로 YAMNet 로딩과 추론을 확인한다.

```bash
python scripts/verify_inference.py
```

성공하면 마지막에 아래가 출력된다.

```text
PASS: YAMNet inference OK
```

### 캐시 오류가 나는 경우

아래와 비슷한 오류가 나면 YAMNet 캐시가 불완전하게 받아진 것이다.

```text
contains neither saved_model.pb nor saved_model.pbtxt
```

해결:

```bash
mv "$TFHUB_CACHE_DIR" "${TFHUB_CACHE_DIR}_broken_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TFHUB_CACHE_DIR"
python scripts/verify_inference.py
```

---

## 10. 샘플 WAV 추론 실행

샘플 파일이 있는지 확인한다.

```bash
find data/sample -name "*.wav" | head
```

사이렌 샘플을 실행한다.

```bash
python -m src.cli --input data/sample/siren/1-31482-A-42.wav --verbose
```

다른 샘플:

```bash
python -m src.cli --input data/sample/glass_shatter/1-20133-A-39.wav --verbose
python -m src.cli --input data/sample/baby_cry/1-187207-A-20.wav --verbose
python -m src.cli --input data/sample/vehicle_horn/1-17124-A-43.wav --verbose
```

로그 파일로 저장하려면:

```bash
mkdir -p output
python -m src.cli \
  --input data/sample/siren/1-31482-A-42.wav \
  --verbose \
  --log output/jetson_file_test.jsonl
```

---

## 11. 로컬 마이크 실행

Jetson에 USB 마이크나 오디오 입력 장치가 연결되어 있다면 장치 목록을 확인한다.

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

기본 마이크로 실행:

```bash
python -m src.cli --input mic --threshold 0.4 --verbose
```

장치 번호를 지정하려면:

```bash
python -m src.cli --input mic --device 1 --threshold 0.4 --verbose
```

종료는 `Ctrl+C`다.

주의: 현재 최종 목표 구조에서는 Jetson에 마이크를 직접 붙이는 것이 아니라 ESP32가 마이크 음성을 보내는 구조다. 마이크 모드는 Jetson 단독 동작 확인용이다.

---

## 12. ESP32 네트워크 수신 모드 실행

ESP32가 아직 없어도 network receiver를 먼저 실행할 수 있다.

터미널 1에서 Jetson receiver 실행:

```bash
cd ~/capstone/model
source .venv/bin/activate

python -m src.cli \
  --input network \
  --listen-host 0.0.0.0 \
  --listen-port 5005 \
  --device-count 2 \
  --verbose \
  --log output/network_run.jsonl
```

이 상태에서 터미널은 계속 켜둔다.

Jetson IP 확인:

```bash
hostname -I
```

출력 예:

```text
192.168.0.23
```

이 IP를 ESP32 또는 테스트 sender의 `--host` 값으로 사용한다.

---

## 13. UDP sender로 network receiver 테스트

터미널 2를 새로 열고 같은 저장소/가상환경으로 들어간다.

```bash
cd ~/capstone/model
source .venv/bin/activate
```

같은 Jetson 안에서 receiver를 테스트하려면 `127.0.0.1`로 보낸다.

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/siren/1-31482-A-42.wav \
  --device-id 1 \
  --host 127.0.0.1 \
  --port 5005
```

장치 2도 테스트한다.

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/glass_shatter/1-20133-A-39.wav \
  --device-id 2 \
  --host 127.0.0.1 \
  --port 5005
```

다른 PC에서 Jetson으로 보낼 때는 `127.0.0.1` 대신 Jetson IP를 넣는다.

```bash
python scripts/send_pcm_udp.py \
  --file data/sample/siren/1-31482-A-42.wav \
  --device-id 1 \
  --host <jetson-ip> \
  --port 5005
```

성공하면 receiver 터미널에 다음과 비슷한 정보가 보인다.

```text
device=1 seq=...
WINDOW scores:
...
DANGER: siren
```

---

## 14. 테스트 실행

TensorFlow/YAMNet 캐시가 필요 없는 빠른 테스트:

```bash
python -m pytest \
  tests/test_network_stream.py \
  tests/test_debounce_trigger.py \
  tests/test_synthetic_signals.py \
  tests/test_live_display.py \
  -q
```

전체 테스트:

```bash
python -m pytest tests -q
```

전체 테스트는 YAMNet 모델 다운로드/캐시 상태에 따라 실패할 수 있다. `saved_model.pb` 관련 오류가 나면 9장의 캐시 복구 절차를 따른다.

---

## 15. 자주 생기는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `git: command not found` | Git 미설치 | `sudo apt install -y git` |
| `python -m venv` 실패 | venv 패키지 없음 | `sudo apt install -y python3-venv` |
| `ModuleNotFoundError: tensorflow` | TensorFlow 설치 실패 또는 가상환경 미활성화 | `source .venv/bin/activate` 후 import 확인 |
| TensorFlow 설치 실패 | JetPack/Python/wheel 호환 문제 | 8장의 TensorFlow 실패 절차 확인 |
| `saved_model.pb` 없음 | TF-Hub 캐시 불완전 | 9장의 캐시 삭제/재다운로드 절차 실행 |
| WAV 파일 없음 | 브랜치/저장소 상태 또는 경로 문제 | `find data/sample -name "*.wav"` |
| network receiver가 패킷을 못 받음 | IP/포트/방화벽/다른 네트워크 문제 | `hostname -I`, `--listen-port`, 같은 Wi-Fi 확인 |
| `device_id`가 안 보임 | sender 실행 안 됨 또는 잘못된 포트 | receiver 켠 뒤 sender 실행, port 5005 확인 |
| 마이크 입력이 무음 | 장치 인식/권한/볼륨 문제 | `sounddevice.query_devices()`와 OS 사운드 설정 확인 |

---

## 16. 다음 단계

Jetson에서 위 절차가 통과되면 다음 개발로 넘어간다.

1. ESP32 한 대에서 I2S 마이크 캡처 구현
2. ESP32가 이 문서의 UDP 패킷 규격으로 Jetson에 전송
3. Jetson `--input network`로 실제 ESP32 오디오 수신
4. Jetson에서 위험음 분류 결과를 ESP32로 JSON 반환
5. ESP32에서 진동 모터 제어
