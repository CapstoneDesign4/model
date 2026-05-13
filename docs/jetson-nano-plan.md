# Jetson Nano 배포 계획서: YAMNet 위험 소리 감지 시스템

> 버전: v0.1 (2026-05-14)
> 참조: `CLAUDE.md`, `docs/development-plan.md`, `docs/docker-plan.md`

---

## 1. 목적과 범위

### 1.1 사용자 요구사항

> "나중에 엔비디아 젯슨 나노에서 가능하면 학습을 시키고, 젯슨에서 모델을 실행시켜서 마이크에서 받은 소리 데이터로 위험 소리를 분석할건데 어떻게 해야할지"

이를 세 가지 하위 목표로 정리한다.

1. (가능하면) Jetson Nano에서 M3 헤드 파인튜닝 수행
2. Jetson Nano에서 추론 실행 (YAMNet backbone + 학습된 헤드)
3. USB 또는 I2S 마이크 입력으로 실시간 위험 소리 분석

### 1.2 "추론 위주 / 학습은 호스트 PC"를 권장하는 이유

Jetson Nano는 Maxwell GPU 기반 엣지 디바이스로 설계 목적 자체가 추론(inference)이다. 학습(training)은 반복적인 그래디언트 연산과 대용량 배치 처리를 필요로 하며, 4GB 공유 메모리(CPU+GPU 합산) 환경에서는 YAMNet backbone 로딩만으로도 약 1GB를 소비하여 학습 배치를 위한 여유 메모리가 매우 부족하다. 또한 Jetson Nano는 EOL 기기로 공식 지원 Python이 3.6.9에 고정되어 있어 우리 프로젝트의 Python 3.11 + TF 2.13 환경과 정합성 문제가 있다. 따라서 **학습은 호스트 PC 또는 Colab에서 수행하고, 산출된 가중치 파일만 Jetson에 복사하여 추론 전용으로 운용하는 것을 본 문서의 권장 전략으로 정한다.**

이 결정은 docker-plan.md의 "학습(M3)은 Colab/GPU 머신 사용" 방침과 동일한 방향이다.

---

## 2. 현실 점검: 하드웨어와 우리 코드의 정합성

### 2.1 Jetson Nano 하드웨어 제약 요약

| 항목 | 사양 | 의미 |
|---|---|---|
| GPU | Maxwell, 128 CUDA cores | Tensor Core 없음. TensorRT INT8은 Xavier 이후만 효과적 |
| 메모리 | 4GB (CPU+GPU 공유) / 2GB 모델도 존재 | 추론 런타임만 해도 1~1.5GB 예상. 2GB 모델은 비추천 |
| CPU | ARM Cortex-A57 quad-core (aarch64) | x86_64 바이너리 휠 사용 불가 |
| 스토리지 | microSD 기본 (NVMe 미지원) | USB 3.0 SSD 연결이 차선책 |
| JetPack | 4.6.x (마지막 공식 지원) — EOL | Ubuntu 18.04 기반. JetPack 5/6 미지원 |
| 공식 지원 Python | 3.6.9 | NVIDIA 제공 TF 휠은 TF 2.7이 최신 |
| TF 최대 버전 | `tensorflow==2.7.0+nv22.x` | NVIDIA 공식 aarch64 휠 기준 |

### 2.2 우리 프로젝트 현재 스택

| 항목 | 현재 값 |
|---|---|
| Python | 3.11 |
| TensorFlow | 2.13~2.15 |
| YAMNet | TF-Hub (`tfhub.dev/google/yamnet/1`) |
| 헤드 | Dense 2층 sigmoid, 12 클래스 |
| 마이크 입력 | `sounddevice` (PortAudio) |

### 2.3 미스매치 분석

우리 코드는 **Jetson Nano 위에서 그대로 실행되지 않는다.** 핵심 충돌은 다음과 같다.

| 충돌 항목 | 호스트(PC/Colab) | Jetson Nano | 해결 방향 |
|---|---|---|---|
| Python 버전 | 3.11 | 3.6.9 (시스템) | Jetson 전용 별도 환경 구성. pyenv 또는 deadsnakes PPA로 Python 3.8 확보 가능 (3.11은 불가) |
| TensorFlow | 2.13~2.15 | 2.7.0+nv22.x 최대 | Jetson 측은 TF 2.7 또는 TFLite/ONNX 런타임으로 대체 |
| 모델 포맷 | `.h5` / SavedModel (TF 2.13+) | TF 2.7로 직접 로드 시 실패 가능 | 학습 후 TFLite(.tflite) 또는 ONNX(.onnx)로 변환하여 전달 |
| aarch64 휠 | 미필요 | x86 휠 사용 불가 | NVIDIA 공식 휠 또는 pip aarch64 지원 휠만 설치 |

### 2.4 Jetson Nano의 능력 한계선

- **YAMNet 추론 (CPU/GPU)**: 가능. 0.48s hop 안에 1프레임 처리 목표 달성 가능성 높음.
- **M3 경량 헤드 학습 (소규모)**: 조건부 가능. backbone freeze + Dense 2층(파라미터 수만 개)이므로 배치 크기 16 이하, epoch 수를 줄이면 4GB 내에서 가능할 수 있음. 단 학습 속도가 매우 느림 (수 시간).
- **YAMNet 전체 파인튜닝**: 불가. backbone 파라미터 수백만 개의 그래디언트 연산이 4GB를 초과함.

---

## 3. 권장 전략: "호스트 학습 → Jetson 추론"

이 흐름을 본 프로젝트의 **메인 배포 경로**로 정한다.

```
[호스트 PC / Colab]
  M3: YAMNet backbone freeze + Dense 헤드 학습
    → 산출물: models/head.h5  또는  models/head.tflite
    → (선택) ONNX 변환: models/head.onnx

[파일 전송]
  SCP 또는 USB 복사
    models/head.tflite  →  Jetson:/home/user/danger-audio/models/

[Jetson Nano]
  M4: TFLite 런타임으로 추론
    → python3 -m src.cli --input mic --threshold 0.4
```

### 3.1 장점

- 학습 반복(hyperparameter 탐색, 데이터 증강 실험)을 빠른 GPU 환경에서 수행 가능
- Python 버전 충돌 없이 호스트에서 최신 TF/librosa 사용
- Jetson은 OOM 위험 없이 안정적으로 추론만 수행
- 모델 가중치 파일(수십~수백 KB)만 전송하면 되므로 배포 경량

### 3.2 단점

- 호스트에서 TFLite 변환 단계가 1단계 추가됨
- YAMNet backbone을 TFLite로 변환할 때 일부 op 호환성 이슈가 알려져 있음 (§5 참조)
- Jetson 환경 구성 1회 투자 필요 (§6 체크리스트 참조)

---

## 4. 대안 전략: "Jetson에서 헤드만 학습 (실험적)"

이 절은 "할 수 있는가"를 문서화하는 목적으로 작성한다. **권장하지 않는다.**

### 4.1 가능성 분석

YAMNet backbone이 완전히 freeze된 상태에서 경량 헤드(Dense(256, ReLU) → Dropout → Dense(12, Sigmoid))만 학습하면 학습 대상 파라미터가 약 260,000개 수준이다. 이는 배치 크기 16, float32 기준 약 4MB의 그래디언트 버퍼에 불과하므로, YAMNet 로딩 후 남은 메모리(약 2~3GB)에 이론적으로 들어간다.

### 4.2 환경 구성 난이도

| 작업 | 난이도 | 비고 |
|---|---|---|
| JetPack 4.6 + NVIDIA TF 2.7 휠 설치 | 중간 | 별도 pip 인덱스 필요 |
| Python 3.8 확보 (deadsnakes PPA) | 낮음 | TF 2.7은 Python 3.6~3.8 지원 |
| numpy, scipy, librosa aarch64 호환 | 중간 | 버전 핀 주의 (numpy 1.19 계열) |
| 학습 데이터셋 microSD 저장 | 중간 | FSD50K 전체 약 30GB. 일부만 사용 권장 |

### 4.3 예상 학습 시간 및 제약

- A57 CPU + Maxwell GPU 학습 배치 처리: epoch당 수십 분 예상
- 1,000 스텝 기준 5~10시간 이상 소요 가능
- microSD I/O 병목으로 데이터 로딩이 훈련 속도를 제한함
- swap 4GB 없이 OOM 발생 가능성 높음

### 4.4 결론

이 전략의 실용적 가치는 "Jetson 단독 독립 운용 시나리오(인터넷 없는 현장 재학습)"에 한정된다. 일반 개발 흐름에서는 호스트 학습이 압도적으로 효율적이므로, 이 절의 내용은 M3 완료 후 "실험적 검증" 단계에서만 시도한다.

---

## 5. 모델 포맷·런타임 선택

| 옵션 | 변환 난이도 | Nano 추론 성능 | 파이프라인 통합 | 비고 |
|---|---|---|---|---|
| TF SavedModel + TF 2.7+nv | 변환 불필요 | 보통 | 쉬움 | RAM 부담 큼. TF 2.7↔2.13 포맷 불일치 위험 |
| **TFLite (CPU)** | 중간 (Select TF Ops 필요 가능성) | 보통 | 쉬움 | GPU 미사용. 가장 안정적. **1차 권장** |
| TFLite + GPU delegate | 중간~높음 | 빠름 | 보통 | Maxwell GPU 호환성 불확실. 검증 필요 |
| ONNX + onnxruntime-gpu | 높음 | 빠름 | 보통 | aarch64 휠 빌드 필요할 수 있음. **2차 권장** |
| TensorRT (.engine) | 높음 | 가장 빠름 | 어려움 | YAMNet op 일부 미지원 가능. INT8 가속은 Maxwell에서 효과 제한적 |

### 5.1 권장 순서

1. **1차: TFLite (CPU)** — 변환 경로가 가장 명확하고 Jetson Nano에서 안정성이 검증된 경로다. YAMNet의 일부 TF op이 TFLite 기본 커널에 없을 수 있으므로 변환 시 `--enable_select_tf_ops` 플래그를 사용한다.
2. **2차: ONNX Runtime** — TFLite CPU로 latency 목표(1프레임 < 400ms)를 달성하지 못할 경우 ONNX + onnxruntime-gpu 경로를 시도한다. TF → ONNX 변환은 `tf2onnx` 툴을 사용한다.
3. **3차: TensorRT** — 본격 양산 또는 Orin Nano 마이그레이션 단계에서 검토한다. Maxwell에서는 INT8 Tensor Core 가속이 없어 FP16 대비 효과가 제한적이다.

### 5.2 TFLite Select TF Ops 필요 이유

YAMNet은 내부적으로 STFT 및 mel filterbank 연산을 포함하는데, 이 중 일부는 TFLite 기본 내장 커널에 없는 TF op을 사용한다. `select_tf_ops`를 활성화하면 해당 op을 TFLite 런타임에 TF op 구현체로 fallback하여 실행하므로 바이너리 크기가 약간 증가하지만 변환 실패를 막을 수 있다.

---

## 6. Jetson Nano 환경 구성 체크리스트

이 절은 Jetson Nano를 처음 셋업할 때 순서대로 따를 수 있는 실행 가능한 목록이다.

### 6.1 SD 카드 준비

- [ ] NVIDIA 공식 JetPack 4.6.x 이미지를 다운로드
  ```
  https://developer.nvidia.com/embedded/jetpack-sdk-46
  ```
- [ ] Balena Etcher 또는 `dd`로 64GB 이상 microSD에 플래싱
- [ ] 최초 부팅 후 시스템 업데이트: `sudo apt update && sudo apt upgrade -y`

### 6.2 swap 설정 (OOM 방지 필수)

YAMNet 로딩(약 100MB) + 런타임 총 1~1.5GB 예상. swap 없이 GUI 환경에서 실행하면 OOM 가능성이 있다.

```bash
# GUI 비활성화 (runlevel 3 상당)
sudo systemctl set-default multi-user.target

# zram 비활성화
sudo systemctl disable nvzramconfig

# 4GB swap 파일 생성
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 6.3 시스템 라이브러리 설치

```bash
sudo apt install -y \
  libsndfile1 libsndfile1-dev \
  ffmpeg \
  libportaudio2 portaudio19-dev \
  python3-pip python3-dev \
  git
```

### 6.4 Python 버전 확보

JetPack 4.6 기본 Python은 3.6.9다. TF 2.7은 Python 3.6~3.8을 지원하므로 Python 3.8을 추가 설치한다.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv ~/danger-audio-env
source ~/danger-audio-env/bin/activate
```

### 6.5 NVIDIA 공식 TensorFlow 2.7 휠 설치

```bash
# NVIDIA 공식 aarch64 TF 2.7 휠
pip install --upgrade pip
pip install \
  "https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.7.0+nv22.1-cp38-cp38-linux_aarch64.whl"
```

> 정확한 URL은 NVIDIA 공식 페이지에서 JetPack 버전에 맞게 확인한다.
> 참고: `https://developer.nvidia.com/embedded/downloads#?search=tensorflow`

### 6.6 Jetson 전용 requirements.txt 설치

메인 `requirements.txt`는 Python 3.11 + TF 2.13 기준이므로 Jetson에서는 사용 불가다. 별도 `requirements-jetson.txt`를 사용한다 (§6.7 참조).

```bash
pip install -r requirements-jetson.txt
```

### 6.7 requirements-jetson.txt (별도 관리)

이 파일은 `requirements.txt`와 **분리하여 관리**한다. 메인 파일을 수정하지 않는다.

```
# requirements-jetson.txt
# Python 3.8, JetPack 4.6 (TF 2.7) 환경 전용
# 메인 requirements.txt(Python 3.11, TF 2.13)와 별도 관리

tensorflow==2.7.0+nv22.1   # NVIDIA 공식 aarch64 휠로 설치 (위 §6.5 참조)
tensorflow-hub>=0.12,<0.13  # TF 2.7 호환 버전
numpy>=1.19,<1.22           # TF 2.7 호환 numpy
librosa>=0.9,<0.10          # Python 3.8 호환 확인된 버전
soundfile>=0.10
scipy>=1.7,<1.9
sounddevice>=0.4.4
pyyaml>=5.4
```

### 6.8 프로젝트 코드 배포

```bash
git clone https://github.com/CapstoneDesign4/model.git ~/danger-audio
cd ~/danger-audio

# 또는 SCP로 직접 복사
# scp -r user@호스트PC:/path/to/model/ ~/danger-audio/
```

### 6.9 학습된 모델 가중치 배치

```bash
# 호스트에서 전송
scp models/head.tflite jetson@<JETSON_IP>:~/danger-audio/models/jetson/head.tflite
```

모델 저장 경로는 `models/jetson/`으로 구분하여 호스트용 가중치와 분리한다.

### 6.10 smoke test

```bash
cd ~/danger-audio
source ~/danger-audio-env/bin/activate
python3 -m src.cli --input data/sample/test.wav --threshold 0.0 --verbose
```

크래시 없이 12종 score가 출력되면 환경 구성 완료다.

---

## 7. 마이크 입력 (Jetson에서 실시간 분석)

### 7.1 USB 마이크 (권장)

USB 마이크는 ALSA/PortAudio 호환성이 좋고 추가 배선 없이 플러그앤플레이로 동작한다. **1차 권장 방식이다.**

| 모델 | 특징 | 비고 |
|---|---|---|
| Samson Go Mic | USB, 컴팩트, 16kHz 지원 | 저렴하고 검증된 선택 |
| ReSpeaker USB Mic Array | 마이크 어레이, 빔포밍 내장 | 향후 노이즈 캔슬링 업그레이드 시 유리 |
| Blue Snowball / Yeti Nano | USB, 고품질 | 데스크탑 환경에 적합 |

어느 USB 마이크를 표준으로 정할지는 팀 합의가 필요하다 (§12 Open Questions 참조).

### 7.2 디바이스 인덱스 확인

```bash
arecord -l          # ALSA 디바이스 목록
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

출력된 인덱스를 CLI `--device` 옵션에 지정한다.

```bash
python3 -m src.cli --input mic --device 1 --threshold 0.4
```

### 7.3 마이크 게인 캘리브레이션

```bash
alsamixer   # F6으로 USB 마이크 카드 선택 → Capture 게인 60~80% 설정
```

게인이 너무 낮으면 YAMNet score가 전반적으로 낮게 나오고, 너무 높으면 클리핑으로 왜곡이 발생한다. 조용한 환경에서 게인을 조정하며 `--threshold 0.0 --verbose`로 실시간 score를 확인하여 정상 환경음의 score가 0.05 이하가 되도록 맞춘다.

### 7.4 I2S 마이크 (M5 이후 검토)

MEMS I2S 마이크(예: ICS-43434, SPH0645)는 Jetson Nano GPIO 핀에 직접 연결하며 배선 설정과 디바이스 트리 수정이 필요하다. 소비 전력이 낮고 소형화에 유리하지만 드라이버 설정 복잡도가 높아 M5 UART 연동 이후 단계에서 검토한다.

---

## 8. 성능 목표 및 예상 수치

| 항목 | 예상값 | 근거 및 비고 |
|---|---|---|
| YAMNet 추론 latency (CPU) | 150~300ms / 프레임 | x86 CPU 기준 ~40ms. A57은 4~8x 느린 것으로 알려짐 |
| 실시간 처리 목표 | 1프레임 < 400ms | 0.48s hop 기준, 마진 포함 |
| GPU 사용 시 latency | 50~150ms 예상 | TF 2.7 GPU 경로 또는 TFLite GPU delegate 성공 시 |
| 메모리 (YAMNet 로드) | 약 100MB | TF-Hub 모델 기준 |
| 메모리 (런타임 총계) | 1~1.5GB 예상 | TF 런타임 + numpy 버퍼 + 파이썬 힙 |
| 사용 가능한 메모리 여유 | ~2.5GB (4GB 모델 기준) | GUI 비활성화 + swap 4GB 구성 후 |
| 모델 크기 (.tflite) | ≤ 5MB 목표 | `development-plan.md` KPI 기준 |
| CPU 사용률 | 1코어 기준 50~80% 예상 | 실시간 처리 중 측정 필요 |

**결론**: CPU 단독(TFLite)으로 0.48s hop 이내 처리가 가능할 가능성이 높다. 첫 배포는 TFLite CPU로 진행하고, latency 목표(< 400ms) 미달 시 GPU 경로를 검토한다.

모니터링 명령:

```bash
# Jetson 자원 실시간 모니터링
sudo tegrastats --interval 1000

# 추론 latency 측정 (CLI verbose 모드 타임스탬프 활용)
python3 -m src.cli --input data/sample/test.wav --verbose 2>&1 | grep "\[" | head -20
```

---

## 9. 데이터 흐름 다이어그램

```
[USB 마이크 16kHz mono]
  │
  ▼
sounddevice (PortAudio, device 인덱스 지정)
  │
  ▼
링 버퍼 (0.96s 윈도우 / 0.48s hop)
  │  src/audio_io/mic_stream.py
  ▼
노이즈 캔슬링 (M2 WebRTC NS / M4 추가 검토)
  │  src/preprocess/noise_suppress.py
  ▼
YAMNet backbone (TFLite 또는 TF 2.7+nv)
  │  Jetson GPU 또는 CPU 사용
  │  출력: 521-d scores 또는 1024-d embedding
  ▼
경량 헤드 (.tflite 또는 .h5)
  │  12종 danger class sigmoid score
  ▼
trigger (클래스별 임계값 + cooldown + debounce K/N=2/3)
  │  src/postprocess/trigger.py
  ▼
UART JSON 알림 (임베디드 보드로) ← M5 이후
  │  src/embedded/uart_sender.py
  ▼
[MCU / 알림 장치]
```

**M4에서의 변경점**: YAMNet backbone이 TFLite로 대체되고, 헤드 가중치는 `models/jetson/head.tflite`에서 로딩된다. 마이크 파이프라인 코드(`src/audio_io/`, `src/postprocess/`)는 변경 없이 재사용한다.

---

## 10. 마일스톤 매핑

`development-plan.md` §8의 마일스톤 정의를 Jetson 배포 관점에서 재매핑한다.

| 마일스톤 | 주요 작업 | 산출물 | Jetson 관련 여부 |
|---|---|---|---|
| M1 (완료) | YAMNet + threshold 베이스라인 | 추론 스크립트, CLI | 호스트 PC에서만 동작 |
| M2 (완료) | Debounce K/N 구현 | trigger.py debounce, whitelist.yaml 갱신 | 호스트 PC에서만 동작 |
| M3 (진행 예정) | 헤드 파인튜닝 (호스트 PC/Colab) | `models/head.h5`, 평가 리포트 | 학습은 호스트에서. Jetson 무관 |
| **M4 (Jetson 1차 배포)** | TFLite 변환 + Jetson 추론 검증 | `models/jetson/head.tflite`, Jetson 환경 구성 | **Jetson 진입점** |
| M5 (UART 연동) | trigger → UART JSON 송신 | `uart_sender.py`, MCU 수신 코드 | Jetson ↔ MCU 직렬 연결 |
| M6 (성능 최적화, 선택) | TensorRT / GPU delegate / INT8 양자화 | `.engine` 또는 최적화 TFLite | Jetson 성능 개선 |

### M3 → M4 전환 조건

M4 착수 전 M3 Exit Criteria(`development-plan.md` §8)가 충족되어야 한다.

- 위험 클래스 평균 F1 ≥ 0.80
- Recall ≥ 0.85
- `models/head.h5` 호스트 저장 완료

M4에서 해야 할 첫 번째 기술 작업은 `models/head.h5`를 TFLite로 변환하는 것이다. YAMNet backbone + 헤드를 단일 TFLite 그래프로 묶을지, backbone과 헤드를 별도 TFLite 파일로 분리할지는 M4 착수 시점에 결정한다 (§12 Open Questions 참조).

---

## 11. 리스크와 대응

| 리스크 | 영향도 | 발생 가능성 | 대응 |
|---|---|---|---|
| Jetson Nano EOL — 부품 단종/지원 중단 | 높음 | 높음 | Orin Nano 마이그레이션 경로를 미리 검토한다. Orin은 JetPack 6 / Ubuntu 22.04 / Python 3.10 지원으로 우리 스택과 훨씬 정합성이 좋다 |
| TF 2.7 ↔ TF 2.13 모델 포맷 불호환 | 높음 | 높음 | 학습 시 `.h5` 직접 사용 대신 TFLite 또는 ONNX로 export. `tf.saved_model.save` 후 `tf.lite.TFLiteConverter`로 변환 |
| 4GB RAM OOM | 높음 | 중간 | swap 4GB 추가 필수. GUI 비활성화. `tegrastats`로 상시 모니터링 |
| TFLite 변환 시 일부 op 미지원 | 중간 | 중간 | `--enable_select_tf_ops` 적용. 해결 안 되면 ONNX 경유 전환 |
| USB 마이크 클럭 드리프트 | 중간 | 낮음 | `sounddevice` 콜백 timestamp 사용, 주기적 리샘플링 보정 |
| aarch64 pip 휠 부재 | 중간 | 중간 | numpy/scipy/librosa는 공식 aarch64 휠 존재. 없는 패키지는 소스 빌드 (`pip install --no-binary`) |
| 첫 환경 구성에 1~3시간 소요 | 낮음 | 높음 | 환경 구성 완료 후 SD 카드 이미지를 백업하여 팀 공유. 재셋업 시간 단축 |
| Maxwell GPU TFLite delegate 비호환 | 중간 | 중간 | GPU delegate 실패 시 CPU fallback. TFLite CPU 단독으로 성능 목표 달성 가능 여부 먼저 측정 |

---

## 12. 다음 단계 / Open Questions

다음 항목은 팀 합의가 필요한 미결 사항이다.

1. **표준 USB 마이크 모델 결정**: 어떤 마이크를 공식 하드웨어로 정할지. ReSpeaker USB는 어레이 + 빔포밍으로 향후 노이즈 캔슬링 강화에 유리하지만 가격이 높다.

2. **M3 학습 산출 포맷 결정**: 처음부터 TFLite 포맷으로 학습 후 저장할지(변환 단계 생략), 아니면 `.h5`로 받고 M4에서 변환할지. `.h5` → TFLite 변환 경로가 잘 동작한다면 후자가 호스트 학습 환경에서 더 편리하다.

3. **Jetson Nano 4GB vs 2GB 타겟**: 2GB는 런타임 1~1.5GB 예상치 대비 여유가 거의 없어 비추천한다. 4GB 모델을 타겟으로 정하고 2GB는 지원 범위 밖으로 명시한다.

4. **backbone과 헤드의 TFLite 분리 여부**: 단일 그래프 변환이 실패할 경우 YAMNet backbone을 별도 TFLite로 두고 헤드만 별도 모델로 로딩하는 2-model 구성이 대안이다.

5. **Orin Nano 마이그레이션 시점**: 장기적으로 Jetson Nano EOL 대응이 필요하다면 Orin Nano(JetPack 6, Python 3.10, CUDA 11.4)로의 이전을 별도 안건으로 수립한다.

---

## 참고 자료

- NVIDIA JetPack SDK 4.6: `https://developer.nvidia.com/embedded/jetpack-sdk-46`
- NVIDIA TensorFlow for Jetson (aarch64 휠): `https://developer.nvidia.com/embedded/downloads#?search=tensorflow`
- TFLite Select TF Ops 문서: `https://www.tensorflow.org/lite/guide/ops_select`
- tf2onnx (TF → ONNX 변환): `https://github.com/onnx/tensorflow-onnx`
- ONNX Runtime aarch64: `https://onnxruntime.ai/docs/install/`
- tegrastats 사용법: `https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/AT/JetsonLinuxToolchain/tegrastats.html`
- deadsnakes PPA (Ubuntu 18.04 Python 3.8): `https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa`

---

*문서 버전: v0.1 (2026-05-14). M4 착수 시 §6 체크리스트 및 §5 포맷 선택 결과를 반영하여 갱신할 것.*
