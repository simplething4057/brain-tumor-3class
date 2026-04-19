# Brain Tumor 3-Class Diagnosis System

뇌 MRI 세그멘테이션 파일(`.nii`)을 업로드하면 **GLI(신경교종) · MEN(수막종) · MET(전이성 종양)** 3종을 자동 분류하고, RAG+LLM 기반 방사선 소견 보고서를 생성하는 웹 애플리케이션입니다.

---

## 데모

### 애플리케이션 화면

![Brain Tumor Diagnosis 애플리케이션](docs/screenshot.png)

> 좌: MRI 파일 업로드 패널 · 중앙: NiiVue 3D 뷰어 (Multi / 3D 뷰 + 종양 히트맵) · 우: 분류 결과 + 방사선 소견 보고서

### 자동 생성 보고서 샘플 (PDF)

📄 [sample-report.pdf](docs/sample-report.pdf)

GLI(신경교종) 케이스의 `[촬영 정보] → [임상 증상] → [MRI 소견] → [결론]` 형식 보고서 및 3D 영상 포함 2페이지 PDF 출력물입니다.

---

## 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [기술 스택](#기술-스택)
3. [구현 과정](#구현-과정)
4. [트러블슈팅](#트러블슈팅)
5. [설치 및 실행](#설치-및-실행)
6. [디렉토리 구조](#디렉토리-구조)

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend (React)                    │
│  FileUploader → NiiVueViewer(3D) → ResultPanel(PDF)     │
└────────────────────────┬────────────────────────────────┘
                         │ REST API (axios)
┌────────────────────────▼────────────────────────────────┐
│                    Backend (FastAPI)                     │
│                                                          │
│  POST /api/predict                                       │
│    └─ pipeline/predict.py                               │
│         ├─ nibabel  → 세그 마스크 파싱                    │
│         ├─ feature_extractor → 21개 정량적 특징 추출      │
│         ├─ Random Forest → GLI/MEN/MET 분류              │
│         └─ estimate_brain_region() → 해부학적 위치 추정  │
│                                                          │
│  POST /api/report/{id}                                   │
│    └─ rag/report_generator.py                           │
│         ├─ ChromaDB (RAG) → 관련 문헌 컨텍스트 검색      │
│         └─ Groq API (LLaMA-3.3-70B) → 보고서 생성        │
│                                                          │
│  PostgreSQL ←→ SQLAlchemy ORM                           │
└─────────────────────────────────────────────────────────┘
```

### ML 파이프라인

```
.nii 세그 파일
    │
    ▼
nibabel 로드 → label별 복셀 추출
    │
    ▼
feature_extractor.py
  ├─ total_voxels / total_volume_mm3
  ├─ et_ratio (조영증강 종양 비율)
  ├─ edema_ratio (부종 비율)
  ├─ core_ratio (괴사/핵심부 비율)
  ├─ lesion_count (병변 수)
  └─ has_tumor
    │
    ▼ (3 classes × 7 features = 21차원)
Random Forest Classifier (meta_classifier.pkl)
    │
    ▼
예측 결과 + 확률 (GLI / MEN / MET)
```

---

## 기술 스택

### 딥러닝 / ML

| 기술 | 역할 | 설명 |
|------|------|------|
| **SwinUNETR** | 뇌 병변 세그멘테이션 | BraTS 2023 GLI 데이터셋으로 학습 (Dice=0.8828). Swin Transformer 기반 U-Net 구조로 3D MRI에서 NCR/ED/ET 레이블 분리 |
| **Random Forest** | 3종 분류기 | SwinUNETR 출력에서 추출한 21개 정량적 특징으로 GLI/MEN/MET 분류 |
| **nibabel** | NIfTI 처리 | `.nii` 형식 MRI/세그멘테이션 파일 로드, voxel→mm³ 변환 |

### 백엔드

| 기술 | 역할 | 설명 |
|------|------|------|
| **FastAPI** | REST API | 비동기 엔드포인트, 멀티파트 파일 업로드 |
| **Uvicorn** | ASGI 서버 | FastAPI 실행 런타임 |
| **SQLAlchemy** | ORM | asyncpg(비동기) + psycopg2(동기) 혼용 |
| **PostgreSQL** | 관계형 DB | 예측 결과, 보고서, 위치 정보 영구 저장 |
| **ChromaDB** | 벡터 DB | 방사선과 문헌 임베딩, RAG 컨텍스트 검색 |
| **LangChain + Groq** | LLM 파이프라인 | LLaMA-3.3-70b-versatile로 임상 보고서 생성 |

### 프론트엔드

| 기술 | 역할 | 설명 |
|------|------|------|
| **React 18 + Vite** | UI 프레임워크 | 컴포넌트 기반 SPA |
| **Tailwind CSS** | 스타일링 | 다크 테마 유틸리티 클래스 |
| **NiiVue** | 3D MRI 뷰어 | WebGL 기반 NIfTI 렌더링, 종양 유형별 레이블 컬러맵 히트맵 |
| **jsPDF + html2canvas** | PDF 내보내기 | DOM 캡처 → 2페이지 PDF (한글 지원) |
| **axios** | HTTP 클라이언트 | 백엔드 API 통신 |

### 인프라

| 기술 | 역할 | 설명 |
|------|------|------|
| **Docker + Compose** | 컨테이너화 | 백엔드 + PostgreSQL 단일 명령 실행 |
| **Kaggle (T4×2)** | GPU 학습 | SwinUNETR 학습 (주 30시간 무료 쿼터 활용) |

---

## 구현 과정

### Phase 1 — SwinUNETR 세그멘테이션 학습

BraTS 2023 GLI 데이터셋, Kaggle T4×2 GPU 환경에서 SwinUNETR 학습. 최종 Dice **0.8828** 달성. Kaggle GPU 쿼터(주 30시간) 제한으로 추가 학습 대신 현재 모델을 기반으로 서비스 구축 방향으로 전환.

### Phase 2 — 3종 분류기 구현

세그멘테이션 마스크에서 nibabel로 복셀 통계를 추출하는 `feature_extractor.py` 작성. 21차원 특징 벡터로 Random Forest 분류기 학습. GLI/MEN/MET 레이블별 특징 분포 차이를 이용해 분류.

### Phase 3 — 웹 앱 설계 및 Docker 통합

FastAPI + React + PostgreSQL + ChromaDB 스택을 설계. `Brain-tumor-diagnosis` 폴더에서 개발 후 `LLM/brain-tumor-3class`로 전체 이전. Docker Compose로 백엔드 + DB를 컨테이너화.

### Phase 4 — NiiVue 3D 뷰어 + 종양 히트맵

NiiVue(WebGL) 통합 후, 종양 유형별(GLI/MEN/MET) 커스텀 레이블 컬러맵 구현. NiiVue의 인덱스 매핑 방식(`value/cal_max × 255`)을 분석해 256-entry LUT로 정확한 색상 적용.

### Phase 5 — RAG + LLM 보고서 자동 생성

ChromaDB에 방사선과 문헌을 임베딩 저장. Groq API (LLaMA-3.3-70b)로 실측값(종양 부피, ET 비율, 부종, 위치)이 포함된 임상 형식 보고서 자동 생성. 보고서 형식: `[촬영 정보] → [임상 증상] → [MRI 소견] → [결론]`.

### Phase 6 — PDF 내보내기

jsPDF + html2canvas 조합으로 2페이지 PDF 구현. 1페이지(예측 결과 + 3D 스크린샷), 2페이지(방사선 소견 보고서).

---

## 트러블슈팅

### T1. `No module named 'src'` (Docker)

**문제** Docker 컨테이너에서 루트 패키지 `src/`를 찾지 못해 ImportError.

**탐색** 컨테이너 내부에서 `sys.path` 확인 → `/ml_pipeline`이 PYTHONPATH에 없음.

**해결** `backend/Dockerfile`에 환경변수 추가.
```dockerfile
ENV PYTHONPATH="/ml_pipeline:${PYTHONPATH}"
```

---

### T2. `WindowsPath`가 Linux에서 오류

**문제** Windows에서 pickle 저장한 `meta_classifier.pkl`을 Linux 컨테이너에서 로드 시 `WindowsPath` 클래스 없음 오류.

**탐색** pickle은 경로 객체를 OS별 클래스로 직렬화. `pathlib.WindowsPath`는 Linux에 정의되지 않음.

**해결** 로드 직전에 패치.
```python
import pathlib
pathlib.WindowsPath = pathlib.PosixPath
```

---

### T3. Docker 볼륨 읽기 전용 오류

**문제** 업로드 파일 저장 중 `Read-only file system`.

**탐색** `docker-compose.yml` 볼륨에 `:ro` 플래그 존재.

**해결** `:ro` 제거.
```yaml
volumes:
  - ./:/ml_pipeline   # :ro 삭제
```

---

### T4. Random Forest 특징 컬럼명 불일치

**문제** `feature_extractor` 반환값(`gli_total_volume_mm3`)과 `FEAT_COLS`(`GLI_volume_mm3`) 불일치로 KeyError.

**탐색** 두 파일의 키 명칭 비교.

**해결** FEAT_COLS를 소문자 prefix + `total_volume_mm3` 기준으로 통일.
```python
FEAT_COLS = [f"{c.lower()}_{f}" for c in CLASSES for f in [...]]
```

---

### T5. NiiVue 컬러맵 색상 미적용

**문제** 레이블(0~3)에 색상 지정 후에도 단색만 표시.

**탐색** NiiVue LUT 인덱스 계산 방식: `(value / cal_max) × 255`. 단순 레이블 번호 ≠ LUT 인덱스.

**해결** 256-entry LUT + 레이블별 정확한 인덱스 계산.
```javascript
const idx = Math.round((label / maxLabel) * 255);
// label 0→0, 1→85, 2→170, 3→255
```

---

### T6. NiiVue `toUpperCase` 오류

**문제** Blob URL로 볼륨 전달 시 확장자 인식 실패.

**탐색** NiiVue가 URL에서 확장자를 파싱해 파일 타입 결정.

**해결** 볼륨 객체에 `name` 필드 추가.
```javascript
{ url: blobUrl, name: `${modality}.nii` }
```

---

### T7. Groq API 모델 지원 종료

**문제** `llama3-70b-8192` → decommissioned 오류.

**탐색** Groq API 에러 응답 확인.

**해결** 최신 모델로 교체.
```python
model_name = "llama-3.3-70b-versatile"
```

---

### T8. `.env` 파일 우선순위로 API 키 미인식

**문제** Docker Compose 환경변수로 전달했으나 pydantic-settings가 `backend/.env`를 우선 로드해 빈 값 사용.

**탐색** `docker exec`로 컨테이너 환경변수 확인 → `GROQ_API_KEY=` 빈 값.

**해결** 루트 `.env`와 `backend/.env` 양쪽에 동일한 값 설정.

---

### T9. PDF 한글 깨짐 + 3D 이미지 빈 화면

**문제** jsPDF로 생성한 PDF에서 한글 깨짐, NiiVue 3D 영상이 빈 이미지.

**탐색(한글)** jsPDF 내장 폰트에 한글 없음.

**탐색(이미지)** WebGL `preserveDrawingBuffer: false` 기본값으로 렌더 프레임 사이 `toDataURL()` 호출 시 빈 버퍼 반환.

**해결(한글)** `html2canvas`로 브라우저 DOM 캡처 → PDF에 이미지로 삽입.

**해결(이미지)** `toDataURL()` 직전 `drawScene()` 명시적 호출.
```javascript
nvRef.current.drawScene();
return canvasRef.current.toDataURL("image/jpeg", 0.92);
```

---

### T10. `brain_region` 컬럼 DB 미존재

**문제** ORM 모델에 추가한 컬럼이 실제 테이블에 없어 INSERT 오류.

**탐색** SQLAlchemy `create_all()`은 기존 테이블 컬럼을 수정하지 않음.

**해결** ALTER TABLE로 수동 추가.
```sql
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS brain_region VARCHAR(100);
```

---

## 설치 및 실행

### 사전 요구사항

- Docker Desktop
- Node.js 18+
- Groq API 키 ([console.groq.com](https://console.groq.com) 무료 발급)

### 1. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에서 GROQ_API_KEY 입력
```

### 2. 백엔드 + DB 실행

```powershell
docker-compose up -d --build
```

### 3. DB 컬럼 마이그레이션 (최초 1회)

```powershell
docker exec brain_tumor_db psql -U btuser -d brain_tumor -c "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS brain_region VARCHAR(100);"
```

### 4. 프론트엔드 실행

```powershell
cd frontend
npm install
npm run dev
# http://localhost:5173 접속
```

---

## 디렉토리 구조

```
brain-tumor-3class/
├── backend/
│   ├── api/routes.py              # FastAPI 엔드포인트
│   ├── core/config.py             # 환경변수 설정
│   ├── db/
│   │   ├── models.py              # SQLAlchemy ORM
│   │   └── crud.py                # DB CRUD
│   ├── pipeline/predict.py        # 예측 파이프라인
│   ├── rag/report_generator.py    # RAG + LLM 보고서 생성
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/src/
│   ├── App.jsx
│   └── components/
│       ├── NiiVueViewer.jsx       # 3D MRI + 히트맵
│       ├── ResultPanel.jsx        # 결과 + 보고서 + PDF
│       └── HistoryPanel.jsx
├── src/
│   ├── classifier/feature_extractor.py
│   └── inference/brats_infer.py  # SwinUNETR 추론
├── models/weights/
│   └── meta_classifier.pkl        # 학습된 RF 분류기
├── docker-compose.yml
├── .env.example                   # 환경변수 템플릿
└── .gitignore
```

---

> **면책 고지** 본 시스템은 AI 보조 진단 도구입니다. 최종 의학적 판단은 반드시 전문 방사선과 의사의 판독이 필요합니다. 연구·학습 목적으로만 사용하세요.
