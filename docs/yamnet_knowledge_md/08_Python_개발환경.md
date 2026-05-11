# Python / 개발 환경

### 8.1 가상환경(venv)과 의존성 관리

**가상환경**: 프로젝트별로 Python 패키지를 격리하는 방법이다. 시스템 Python에 영향 없이 프로젝트별 버전을 독립 관리한다.

```
프로젝트 루트/
├── .venv/          ← 가상환경 (git에 포함 안 함)
├── requirements.txt ← 의존성 목록
└── src/
```

- `requirements.txt`: 필요한 패키지와 버전을 기록. `pip install -r requirements.txt`로 일괄 설치.
- `.venv`: 실제 패키지가 설치되는 디렉터리. `.gitignore`에 추가해 저장소에 포함하지 않는다.

> 이 프로젝트는 Python 3.11을 요구한다. `py -3.11 -m venv .venv`로 생성.

---

### 8.2 주요 라이브러리 역할

| 라이브러리 | 역할 | 본 프로젝트에서 쓰임 |
|---|---|---|
| `tensorflow` | 딥러닝 프레임워크 | YAMNet 모델 로드 및 추론 실행 |
| `tensorflow_hub` | TF-Hub 모델 다운로드/로드 | YAMNet을 URL로 로드 |
| `librosa` | 오디오 파일 읽기, 리샘플링 | WAV 파일 전처리, 16kHz 변환 |
| `sounddevice` | 마이크 실시간 캡처 | `--input mic` 모드에서 오디오 입력 |
| `numpy` | 수치 배열 연산 | 파형, scores, 임베딩 배열 처리 |
| `pytest` | 단위 테스트 프레임워크 | `tests/` 하위 테스트 실행 |
| `PyYAML` | YAML 파일 파싱 | `config/whitelist.yaml` 로드 |

---

### 8.3 pytest 단위 테스트 패턴 (mock, fixture)

**fixture**: 테스트 함수에 공통으로 필요한 준비 데이터나 객체를 제공하는 함수다. `@pytest.fixture` 데코레이터로 정의하고, 테스트 함수의 인자로 자동 주입된다.

**mock**: 실제 구현 대신 가짜 객체를 주입해 외부 의존성(네트워크, YAMNet 등) 없이 테스트하는 방법이다.

```
테스트 파일 예: tests/test_debounce_trigger.py

- YAMNet을 로드하지 않고 mock scores를 직접 주입
- Trigger 클래스의 debounce/cooldown 로직만 단독 테스트
- pytest 실행 시 네트워크 불필요
```

> 프로젝트 매핑: `pytest tests/ -v`로 네트워크 없이 실행 가능한 테스트만 돈다. YAMNet 로딩이 필요한 테스트는 `-k "yamnet"` 옵션으로 분리.

---

### 8.4 numpy 배열 shape 다루기 기본

오디오 처리에서 자주 등장하는 shape 패턴:

| 의미 | shape 예시 |
|---|---|
| 0.96초 파형 | `(15360,)` — 1D 배열 |
| YAMNet scores (2패치) | `(2, 521)` — 2D 배열 |
| 평균 scores | `(521,)` — 1D 배열 |
| 배치 추론 | `(B, 15360)` — B개 윈도우 묶음 |
| 임베딩 | `(2, 1024)` — 2패치 × 1024차원 |

shape 확인: `arr.shape`
특정 인덱스 추출: `scores[..., [302, 304, 390]]`  — 위험 클래스 인덱스 슬라이싱
패치 평균: `scores.mean(axis=0)` — (num_patches, 521) → (521,)

---

## 추가 설명: 개발 환경에서 가장 먼저 확인할 것

Python 프로젝트는 환경 문제가 자주 발생한다. 문제가 생기면 코드부터 고치기보다 먼저 환경을 확인해야 한다.

```powershell
python --version
pip --version
where python
```

가상환경이 활성화되어 있는지도 확인한다.

```powershell
(.venv) PS C:\project>
```

### numpy shape 디버깅 습관

오디오 AI 디버깅에서는 shape 출력이 매우 중요하다.

```python
print("waveform", waveform.dtype, waveform.shape)
print("scores", scores.shape)
print("embeddings", embeddings.shape)
```
