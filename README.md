# Habitus: 시선·화면 콘텐츠 융합 기반 실시간 인지적 몰입도 추론 시스템

> Real-Time Cognitive State Inference and Adaptive Learning Coaching via the Integration of Visual Attention and Screen Content

**개발 기간** : 2026.03 ~ 2026.06 (4개월) <br />
**팀 구성** : 3인 (이지수, 김남우, 홍윤조) <br />
**담당 역할** : SAES(Scenario-Adaptive Engagement Score) 설계 및 구현, Early Fusion BiLSTM 모델링, Ablation 실험 설계 및 분석, 논문 작성 <br />

---

## 1. 프로젝트 소개

온라인 학습 환경에서는 학습자가 화면을 응시하고 있더라도 실제로 콘텐츠에 집중하고 있는지 판단하기 어렵습니다. 기존의 몰입도 측정 시스템은 대부분 시선이 화면 안에 있는지 없는지(binary)만 판단하거나, 표정·자세 등 표면적 행동 신호에 의존해 **인지적으로는 이탈했지만 신체적으로는 화면을 보고 있는(mind wandering) 상태**를 포착하지 못한다는 한계가 있었습니다.

**Habitus**는 이 문제를 해결하기 위해 (1) 시선 추적 데이터와 (2) 웹페이지의 실제 화면 구조 정보(MAUI)를 융합하여, 사람이 직접 라벨링하지 않고도 객관적인 몰입도 지표를 자동으로 산출하는 시스템입니다.

### 핵심 아이디어
- 시선이 "어디에 머물렀는가"뿐 아니라 "그곳이 실제로 어떤 학습 콘텐츠였는가"까지 함께 해석
- 수작업 라벨링 없이, 시선의 객관적 동역학(분산, 회귀 비율, 시나리오별 행동 패턴)만으로 의사(pseudo) 정답 라벨을 생성
- 하나의 모델이 논문 읽기 / 코드 디버깅 / 강의 시청이라는 이질적인 세 가지 과제 상황에 모두 일반화

---

## 2. 프로젝트 목적

1. 수동 주석(annotation) 없이도 확장 가능한 몰입도 ground-truth 지표(SAES) 설계
2. 시선 데이터만으로는 놓치는 "콘텐츠 맥락"을 화면 구조 정보(MAUI)와 결합해 보완
3. 참가자 간 개인차가 큰 시선 데이터에서도 일반화되는 경량 시계열 모델 구축
4. Leave-One-Out Cross-Validation을 통해 소규모 데이터에서도 신뢰 가능한 성능 검증 체계 마련

---

## 3. 파이프라인 구조

```
원본 gaze CSV (MediaPipe + MAUI Chrome Extension 수집)
            │
            ▼
┌─────────────────────────────┐
│       auto_label.py          │
│  - 시선 피처 추출 (9개)        │
│  - MAUI 섹션 매핑             │
│  - SAES pseudo-GT 산출       │
│  - 시나리오별 sigmoid 정규화   │
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│        train_loo.py          │
│  - 시나리오 내 GT minmax 정규화 │
│  - 슬라이딩 윈도우 시퀀스 생성  │
│  - Early Fusion BiLSTM 학습  │
│  - LOO-CV 평가 및 시각화      │
└─────────────────────────────┘
```

---

## 4. 기술 스택

| 영역 | 기술 |
|---|---|
| 데이터 수집 | MediaPipe Tasks API (Iris Landmark), MAUI Chrome Extension (Manifest V3) |
| 모델링 | PyTorch, Early Fusion BiLSTM |
| 데이터 처리 | Pandas, NumPy, SciPy, StandardScaler |
| 검증 | Leave-One-Out Cross-Validation (참가자 단위) |
| 실험 환경 | Google Colab (GPU), Python 3.10 |

---

## 5. 핵심 기여

### 5.1 SAES (Scenario-Adaptive Engagement Score)
사람이 직접 라벨링하지 않고, 시선의 객관적 신호만으로 몰입도를 산출하는 의사 정답 지표를 설계했습니다.

```
SAES = σ( α·(1 − D_norm) + β·S_scenario + γ·(1 − B_norm) )
```

- `D_norm` : 정규화된 시선 분산 (높을수록 주의 산만)
- `B_norm` : 정규화된 역행(backward-gaze) 비율
- `S_scenario` : 과제별 특화 신호 (논문 읽기 → 좌→우 일관성 / 코드 디버깅 → 마우스 속도 / 강의 시청 → 영상 영역 응시 비율)
- 모델 입력 피처와의 순환 정의(circularity)를 피하기 위해, 콘텐츠 맥락 신호(section score, scroll regularity 등)는 SAES 산출에서 의도적으로 제외

### 5.2 Early Fusion BiLSTM 모델
- 시선 스트림(9) + MAUI 스트림(5) + 시나리오 원-핫(3) = 총 17차원 입력
- BiLSTM(hidden=128, layers=2) → Linear → Sigmoid 구조로 연속형 몰입도 점수 출력
- 7명 참가자 대상 LOO-CV에서 **평균 R² = 0.518** 달성

### 5.3 Ablation 실험
- MAUI 화면 콘텐츠 피처 추가 시 강의 시청 시나리오에서 가장 큰 성능 향상 (R² +0.47 이상)
- Within-scenario normalization 적용 시 평균 R²가 0.116 → 0.518로 향상, 실패 fold 수도 4개 → 1개로 감소

---

## 6. 결과 요약

| 구성 | Mean R² | 실패 Fold (R² < 0) |
|---|---|---|
| Gaze-Only | 0.403 | 1 / 7 |
| Gaze + MAUI (제안 모델) | **0.518** | 1 / 7 |
| Normalization 미적용 | 0.116 | 4 / 7 |
| Normalization 적용 (제안) | **0.518** | 1 / 7 |

- SAES와 모델 입력 피처 간 Pearson 상관계수 r = 0.58 ~ 0.88로, 두 지표가 서로 독립적이면서도 관련된 측면을 측정함을 확인

---

## 7. 한계 및 향후 계획

- 참가자 수가 적어(N=7) LOO-CV의 통계적 검정력에 제약이 있음 → 시나리오별 N≥5 확보 목표
- 현재 평가는 SAES 자체를 정답으로 삼는 self-consistent 방식 → NASA-TLX 등 외부 주관적 척도와의 비교 검증 필요
- 향후 blink rate, pupil dilation 등 생리 신호를 추가하고, 실시간 적응형 피드백 제공을 위한 추론 지연 평가 진행 예정

---

## 8. 참고 자료

- 논문 PDF: `` (Overleaf)
- 관련 레포: [MAUI - Visual Webpage Section Detection](https://github.com/jeho-lee/visual-webpage-section-detection)
