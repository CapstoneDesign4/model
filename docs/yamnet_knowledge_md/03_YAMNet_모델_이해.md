# YAMNet 모델

### 3.1 YAMNet이란

- **정식 명칭**: Yet Another Multiclass Network
- **기반 아키텍처**: MobileNetV1 (경량 depthwise separable convolution)
- **학습 데이터**: AudioSet (YouTube 클립 200만+, 521개 사운드 클래스)
- **목적**: 범용 환경음 분류
- **공개**: Google, Apache-2.0 라이선스

참조: [Google AI Blog - AudioSet](https://ai.googleblog.com/2017/03/announcing-audioset-dataset-for-audio.html)

---

### 3.2 입력/출력 사양

```
입력: float32 waveform, shape=(N,)
      샘플링 레이트 = 16,000 Hz
      최소 길이 = 0.96초 (15,360 샘플)

출력:
  scores:     (num_patches, 521)  각 패치별 521클래스 확률
  embeddings: (num_patches, 1024) 각 패치별 음향 임베딩
  spectrogram: (num_frames, 64)   로그 멜 스펙트로그램
```

> 프로젝트 매핑: `model/yamnet_wrapper.py`에서 scores를 받아 danger_filter로 넘긴다.

---

### 3.3 num_patches가 여러 개인 이유

YAMNet은 내부적으로 입력을 **0.48초 단위의 패치(patch)**로 쪼개서 처리한다.

```
0.96초 입력 →  패치 1 (0.00~0.48초)  → scores[0] (521,)
               패치 2 (0.48~0.96초)  → scores[1] (521,)

num_patches = 2  (0.96초 입력 기준)
```

여러 패치의 scores를 평균(mean)하면 0.96초 전체를 대표하는 단일 (521,) 벡터가 된다. 본 프로젝트에서는 이 평균 scores를 사용한다.

---

### 3.4 YAMNet class_map과 화이트리스트 12종

YAMNet은 521개 클래스 전체를 분류하지만, 본 프로젝트는 그 중 위험 소리 12종만 관심 대상으로 선별한다.

| 역할 | 설명 |
|---|---|
| `yamnet_class_map.csv` | 인덱스 0~520과 AudioSet 클래스명의 매핑 테이블 |
| `config/whitelist.yaml` | 프로젝트에서 사용할 위험 클래스 인덱스 13개 (12종 의미, glass+shatter 통합) |

**화이트리스트 인덱스 목록** (CLAUDE.md 기준):

| 인덱스 | 클래스 | 비고 |
|---|---|---|
| 11 | Screaming | 비명 |
| 20 | Baby cry, infant cry | 영아 울음 |
| 302 | Gunshot, gunfire | 총소리 |
| 304 | Explosion | 폭발음 |
| 390 | Siren | 사이렌 (상위) |
| 391 | Civil defense siren | 민방위 사이렌 |
| 393 | Ambulance (siren) | 구급차 사이렌 |
| 394 | Fire engine, fire truck (siren) | 소방차 사이렌 |
| 420 | Fire alarm | 화재 경보 |
| 421 | Smoke detector, smoke alarm | 연기 감지기 |
| 435 | Glass | 유리음 |
| 437 | Shatter | 파손음 |
| 464 | Vehicle horn, car horn, honking | 차량 경적 |

glass(435) + shatter(437)는 `max()`로 통합해 `glass_shatter` 단일 이벤트로 처리한다.

> 주의: 위 인덱스는 `yamnet_class_map.csv` 직접 조회로 최종 확인이 필요하다. (CLAUDE.md M2 전 P1 항목)

---

### 3.5 TF-Hub에서 모델을 받아 쓰는 방식

TF-Hub(TensorFlow Hub)는 사전학습 모델을 URL로 배포하는 플랫폼이다. 코드 첫 실행 시 모델 파일을 자동 다운로드해 로컬 캐시에 저장한다.

```
URL: https://tfhub.dev/google/yamnet/1
캐시 위치: 환경변수 TFHUB_CACHE_DIR (미설정 시 임시 디렉터리)
최초 실행: 모델 다운로드 (~26MB)
이후 실행: 캐시에서 즉시 로드
```

> 프로젝트 매핑: `scripts/verify_inference.py`를 처음 실행하면 다운로드가 발생한다. 인터넷 연결 필요.

---

## 추가 설명: YAMNet을 프로젝트에 연결하는 방식

YAMNet은 위험 소리 전용 모델이 아니라 범용 환경음 분류 모델이다. 따라서 YAMNet이 주는 521개 score 중 프로젝트에 필요한 일부만 사용한다.

### 기본 구현 흐름

```python
scores, embeddings, spectrogram = yamnet(waveform)
mean_scores = scores.numpy().mean(axis=0)

gunshot = mean_scores[302]
explosion = mean_scores[304]
siren = mean_scores[390]
```

### 주의할 점

`mean(axis=0)`은 전체 patch 평균을 낸다. 사이렌처럼 지속되는 소리에는 자연스럽지만, 총소리처럼 짧은 소리는 평균 과정에서 약해질 수 있다. 이런 경우 `max(axis=0)`도 비교해볼 필요가 있다.

```python
mean_scores = scores.numpy().mean(axis=0)
max_scores = scores.numpy().max(axis=0)
```

실험에서는 mean과 max를 둘 다 로그로 남겨 어떤 방식이 더 적합한지 확인하는 것이 좋다.
