# PRD — 뇌종양 3-Class 진단 웹 애플리케이션

**최종 수정일**: 2026-04-17

---

## 1. 프로젝트 개요

### 배경
MRI 기반 뇌종양 진단은 전문 방사선과 의사의 수작업에 의존하며, 판독 시간이 길고 접근성이 제한적이다. 딥러닝 세그멘테이션과 LLM 기반 소견문 자동화를 통해 1차 진단 보조 도구를 제공한다.

### 목표
- 4채널 NIfTI MRI 업로드 → 뇌종양 3종류 자동 분류 (GLI / MEN / MET)
- 3D 시각화로 세그멘테이션 결과 직관적 확인
- RAG 기반 LLM 소견문 자동 생성
- 예측 이력 관리

---

## 2. 사용자 시나리오

1. 사용자가 4채널 .nii 파일을 업로드한다.
2. SwinUNETR이 세그멘테이션을 수행하고, Random Forest가 종양 유형을 분류한다.
3. NiiVue 3D 뷰어에서 MRI 볼륨과 세그멘테이션 오버레이를 확인한다.
4. 예측 레이블과 클래스별 확률 바 차트를 확인한다.
5. Groq LLM이 RAG를 통해 의료 소견문을 생성한다.
6. 이전 예측 이력을 조회한다.

---

## 3. 기능 요구사항

### Frontend

| 컴포넌트 | 기능 |
|----------|------|
| UploadPanel | .nii 4채널 파일 업로드 |
| NiiVueViewer | 3D 볼륨 렌더링 + 세그멘테이션 오버레이 (WebGL) |
| ResultPanel | 예측 레이블 + 클래스별 확률 바 차트 |
| ReportPanel | LLM 생성 소견문 표시 |
| HistoryList | 이전 예측 이력 목록 및 재조회 |

### Backend API

| Endpoint | Method | 기능 |
|----------|--------|------|
| `/api/predict` | POST | .nii → 세그멘테이션 → 특징 추출 → RF 예측 |
| `/api/files/{id}/*` | GET | NiiVue용 .nii 파일 서빙 |
| `/api/report` | POST | 예측 결과 → RAG → Groq 소견문 생성 |
| `/api/history` | GET/POST | PostgreSQL 이력 조회/저장 |

---

## 4. ML 파이프라인

```
입력: 4채널 .nii (T1, T1ce, T2, FLAIR)
  ↓
SwinUNETR 세그멘테이션
  - 학습 데이터: BraTS 2023 GLI
  - Best Dice: 0.8828 (epoch 60)
  ↓
세그멘테이션 마스크 기반 특징 추출
  (src/classifier/feature_extractor)
  ↓
Random Forest 3-class 분류
  - 클래스: GLI (신경교종) / MEN (수막종) / MET (전이성)
  ↓
출력: 예측 레이블 + 클래스별 확률
```

---

## 5. 기술 스택

### Frontend
- React + Vite + Tailwind CSS
- NiiVue (WebGL 3D 렌더링)
- 배포: Vercel

### Backend
- FastAPI + Uvicorn
- PostgreSQL (이력 저장)
- ChromaDB + LangChain (RAG)
- Groq API (LLM, 배포) / Ollama (로컬 개발)

### ML
- SwinUNETR (세그멘테이션)
- scikit-learn Random Forest (분류)
- nibabel, numpy, scipy (NIfTI 처리)

### 인프라
- docker-compose: backend + PostgreSQL 컨테이너
- backend 컨테이너에 `/ml_pipeline` 볼륨 마운트로 ML 파이프라인 접근
- Vercel: frontend 배포 (예정)

---

## 6. 학습 데이터

| 데이터셋 | 종양 유형 | 상태 |
|----------|-----------|------|
| BraTS 2023 GLI | GLI (신경교종) | ✅ 학습 완료 |
| UCSF-BMSR v1.3 | MET (전이성, 대체 데이터) | 🔄 학습 예정 |
| BraTS 2023 MEN | MEN (수막종) | ⬜ 데이터 미확보 |

---

## 7. 진행 현황

| 항목 | 상태 |
|------|------|
| SwinUNETR GLI 학습 | ✅ 완료 (Dice 0.8828) |
| Backend 구현 | ✅ 완료 |
| Frontend 구현 | ✅ 완료 |
| Docker 구성 | ✅ 완료 |
| UCSF-BMSR MET 학습 | 🔄 예정 (Colab Pro) |
| BraTS MEN 데이터 확보 | ⬜ 미착수 |
| Vercel 배포 | ⬜ 미착수 |
| RF 모델 통합 테스트 | ⬜ 미착수 |

---

## 8. 비기능 요구사항

- .nii 파일 업로드 후 예측 결과 반환: 30초 이내
- 세그멘테이션 모델 Dice 목표: 0.85 이상
- Docker 환경에서 단일 명령(`docker-compose up`)으로 실행 가능
