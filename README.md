# 🏘️ 떡락마을 방범대 (BITA-RADAR)
> **소상공인 위기 징후 예측 및 RAG 기반 정책 추천 시스템**

[cite_start]많은 소상공인이 데이터 기반의 위기 징후를 사전에 인지하지 못해 폐업에 이르는 문제를 해결하기 위해, 머신러닝 기반의 리스크 예측과 RAG 기술을 결합하여 실질적인 경영 개선 행동을 유도하는 솔루션입니다[cite: 43, 49].

---

## 🚀 핵심 기술 및 분석 포인트

### 1. 반응 변수의 재정의: `is_crash_final`
* [cite_start]단순 폐업 여부(`is_closed`)만으로는 포착하기 어려운 미세한 위기 신호를 감지하기 위해 '매출 급락' 지표를 고안했습니다[cite: 84, 86, 287].
* [cite_start]**기준**: 전월 대비 매출 변화율 -30% 이상, 이동평균 대비 -20% 이하 하락, 표준화 Z-score -2 이하 등을 종합하여 리스크를 정의했습니다[cite: 88, 295].

### 2. 고차원 피처 엔지니어링 (Customer Diversity)
* [cite_start]고객층의 편향성이 리스크에 미치는 영향을 수치화하기 위해 **샤넌 엔트로피(Shannon Entropy)**와 **지니-심슨 지수**를 활용한 다양성 지표를 생성했습니다[cite: 167, 168, 170].
* [cite_start]특정 고객군 의존도가 극단적으로 높은(엔트로피 0.3 이하) 점포를 고위험군으로 식별하여 모델의 예측력을 높였습니다[cite: 188].

### 3. RAG 기반 맞춤형 정책 컨설팅
* [cite_start]**System**: 분석된 리스크 점수와 가맹점의 상권 특성을 결합하여 **Chroma DB**에 구축된 정책 데이터를 검색합니다[cite: 526, 530, 534].
* [cite_start]**Output**: AI 컨설턴트가 성동구 지역 정책을 우선순위로 하여 '추천 이유, 주요 혜택, 자격 요건'을 포함한 전문 리포트를 생성합니다[cite: 548, 549].

---

## 📊 모델링 및 성과

| 구분 | 모델 (Model) | AUC | F1 Score |
| :--- | :--- | :---: | :---: |
| **프랜차이즈** | **XGBoost(0.65) + CatBoost(0.35) 앙상블** | **0.8645** | **0.6320** |
| **개인영업** | **XGBoost** | **0.8135** | **0.6788** |

* [cite_start]사업장 형태에 따라 모델을 분리하고, 롤링 교차 검증을 통해 최적의 가중치를 산출하여 성능을 5~10%p 개선했습니다[cite: 320, 321, 385, 388].

---

## 🛠️ Tech Stack
* [cite_start]**Data Analysis**: `Pandas`, `NumPy`, `EDA` [cite: 11, 20]
* [cite_start]**Machine Learning**: `XGBoost`, `CatBoost`, `LightGBM`, `Optuna` [cite: 307, 310, 311, 337]
* [cite_start]**AI/LLM**: `RAG (Retrieval-Augmented Generation)`, `Chroma DB`, `Sentence Transformer` [cite: 31, 529, 530]
* [cite_start]**Deployment**: `Streamlit` (대시보드 구현) [cite: 426]

---

👥 Authors
* 박소현: 데이터 전처리 및 모델링, RAG 시스템 설계, 대시보드 구현
* 정지민: 대시보드 구현 및 서비스 인터페이스 최적화
* 박현서: 데이터 분석 및 머신러닝 모델링
* 박현지: 데이터 분석 및 머신러닝 모델링

---
[cite_start]*본 프로젝트는 **BITA-RADAR** 팀에 의해 수행되었습니다[cite: 6].*
