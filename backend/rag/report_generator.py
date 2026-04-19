"""
RAG + LLM 기반 방사선 소견 보고서 생성
LLM_BACKEND=ollama → Ollama (로컬)
LLM_BACKEND=groq   → Groq API (클라우드)
"""
import httpx
from loguru import logger

from core.config import settings
from rag.vector_store import vector_store

PREDICTION_FULL = {
    "GLI": "신경교종 (Glioma)",
    "MEN": "수막종 (Meningioma)",
    "MET": "전이성 뇌종양 (Brain Metastasis)",
}

MODALITY_LABEL = {
    "t1c": "T1 조영증강 (T1C)",
    "t1n": "T1 비조영증강 (T1N)",
    "t2f": "T2 FLAIR (T2F)",
    "t2w": "T2 가중 (T2W)",
    "seg": "3D 병변 세그멘테이션",
}

REPORT_PROMPT_TEMPLATE = """\
당신은 신경방사선과 전문의입니다. 아래 데이터와 참고 지식을 바탕으로 \
실제 임상에서 사용하는 뇌 MRI 판독 보고서를 한국어로 작성하라.

## 참고 지식 (RAG)
{context}

## 입력 데이터
- AI 예측 종양 유형: {prediction} ({prediction_full})
- 진단 신뢰도: {confidence:.1%}  (GLI {gli_prob:.1%} / MEN {men_prob:.1%} / MET {met_prob:.1%})
- 종양 추정 위치: {brain_region}
- 사용 MRI 시퀀스: {modalities}

## 정량적 측정값
{feature_summary}

---

아래 4개 섹션 제목([촬영 정보], [임상 증상], [MRI 소견], [결론])을 \
반드시 포함하여 보고서를 완성하라. 한자 사용 금지. 자리표시자([X], [크기] 등) 사용 금지.

[촬영 정보]
- 사용된 MRI 시퀀스를 나열하라: {modalities}
- 3D 병변 세그멘테이션 마스크 포함 여부를 명시하라.

[임상 증상]
- {prediction} ({brain_region})에서 흔히 나타나는 예상 임상 증상을 2~4가지 서술하라.
- 종양 부피 {tumor_volume_cm3:.2f} cm³ 및 부종 {edema_volume_cm3:.2f} cm³를 \
  고려한 증상 심각도를 기술하라 (경증/중등도/중증).
- 조영증강 비율 {et_ratio_pct:.1f}%에 근거한 악성도 추정을 한 문장으로 기술하라.

[MRI 소견]
아래 항목을 번호 목록으로 구체적으로 서술하라.
1. 위치 및 형태: {brain_region}, 경계 선명도, 신호 강도 특성 (T1/T2/FLAIR 기준)
2. 크기: 종양 부피 {tumor_volume_cm3:.2f} cm³, \
   조영증강 종양(ET) 비율 {et_ratio_pct:.1f}%
3. 주변 부종: 부종 부피 {edema_volume_cm3:.2f} cm³ — 혈관성 부종 여부 및 범위
4. 괴사/핵심부: 괴사 비율 {core_ratio_pct:.1f}% — 고등급 여부 판단 근거 포함
5. 병변 수 및 침범: {lesion_count}개 병변 — 다발성 여부, 뇌막/혈관 침범 여부

[결론]
아래 항목을 번호 목록으로 기술하라.
1. 주진단: 종양 유형과 등급(저등급/고등급/양성/전이) 추정
2. 부수 소견: 부종, 괴사, 다발성 여부 등 임상적으로 중요한 소견
3. 권고사항: 조직검사, 추가 영상 검사(MRS, PWI 등), 추적 촬영 주기, 종양내과 협진 등

한국어로만 작성. 각 섹션 제목은 대괄호 형식([촬영 정보] 등)을 유지할 것.
환자 이름, 날짜, MR 번호 포함 금지.
"""


def _format_features(feats: dict, prediction: str) -> str:
    """예측 클래스의 주요 특징 — 실측값 포함하여 정리"""
    if not feats:
        return "특징 데이터 없음"

    prefix = prediction.lower()
    lines = []

    vol = feats.get(f"{prefix}_total_volume_mm3")
    if vol is not None:
        lines.append(f"- 종양 부피: {float(vol):.1f} mm³  ({float(vol)/1000:.2f} cm³)")

    et = feats.get(f"{prefix}_et_ratio")
    if et is not None:
        lines.append(f"- 조영증강 종양(ET) 비율: {float(et)*100:.1f}%")

    edema = feats.get(f"{prefix}_edema_ratio")
    if edema is not None and vol is not None:
        edema_vol = float(vol) * float(edema)
        lines.append(
            f"- 부종 비율: {float(edema)*100:.1f}%  (추정 부피: {edema_vol/1000:.2f} cm³)"
        )

    core = feats.get(f"{prefix}_core_ratio")
    if core is not None:
        lines.append(f"- 괴사/핵심부 비율: {float(core)*100:.1f}%")

    cnt = feats.get(f"{prefix}_lesion_count")
    if cnt is not None:
        lines.append(f"- 병변 수: {int(cnt)}개")

    return "\n".join(lines) if lines else "특징 데이터 없음"


def _extract_numeric(feats: dict, prediction: str) -> dict:
    """프롬프트 템플릿에 직접 삽입할 실측 수치"""
    prefix = prediction.lower()
    vol   = float(feats.get(f"{prefix}_total_volume_mm3", 0) or 0)
    et    = float(feats.get(f"{prefix}_et_ratio",   0) or 0)
    edema = float(feats.get(f"{prefix}_edema_ratio", 0) or 0)
    core  = float(feats.get(f"{prefix}_core_ratio",  0) or 0)
    cnt   = int(feats.get(f"{prefix}_lesion_count",  1) or 1)
    return {
        "tumor_volume_cm3":  vol / 1000,
        "et_ratio_pct":      et * 100,
        "edema_volume_cm3":  (vol * edema) / 1000,
        "core_ratio_pct":    core * 100,
        "lesion_count":      cnt,
    }


async def _call_ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            },
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


async def _call_groq(prompt: str) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )
    msg = await llm.ainvoke([HumanMessage(content=prompt)])
    return msg.content.strip()


async def generate_report(
    prediction: str,
    confidence: float,
    gli_prob: float,
    men_prob: float,
    met_prob: float,
    features: dict | None,
    brain_region: str | None = None,
    modalities: list[str] | None = None,
) -> str:
    """방사선 소견 보고서 생성 (RAG + LLM)"""
    query = f"{prediction} brain tumor MRI radiology report findings"
    context_docs = vector_store.query(query, n_results=4, label_filter=prediction)
    context = "\n\n---\n\n".join(context_docs) if context_docs else "참고 문서 없음."

    feats   = features or {}
    region  = brain_region or "불명확"
    nums    = _extract_numeric(feats, prediction)
    feature_summary = _format_features(feats, prediction)

    # 사용된 모달리티 문자열
    if modalities:
        mod_labels = [MODALITY_LABEL.get(m.lower(), m.upper()) for m in modalities]
        modalities_str = ", ".join(mod_labels)
    else:
        modalities_str = "T1C, T2F, 세그멘테이션 (업로드 파일 기준)"

    prompt = REPORT_PROMPT_TEMPLATE.format(
        context=context,
        prediction=prediction,
        prediction_full=PREDICTION_FULL.get(prediction, prediction),
        confidence=confidence,
        gli_prob=gli_prob,
        men_prob=men_prob,
        met_prob=met_prob,
        brain_region=region,
        modalities=modalities_str,
        feature_summary=feature_summary,
        **nums,
    )

    try:
        if settings.llm_backend == "groq" and settings.groq_api_key:
            logger.info("Groq API 사용")
            report = await _call_groq(prompt)
        else:
            logger.info("Ollama 사용")
            report = await _call_ollama(prompt)
        return report
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        return (
            f"[촬영 정보]\n{modalities_str}\n\n"
            f"[임상 증상]\n{region} {prediction} 종양 소견. "
            f"종양 부피 {nums['tumor_volume_cm3']:.2f} cm³, "
            f"부종 {nums['edema_volume_cm3']:.2f} cm³.\n\n"
            f"[MRI 소견]\n"
            f"1. {region}에 종양 관찰, ET 비율 {nums['et_ratio_pct']:.1f}%\n"
            f"2. 종양 부피 {nums['tumor_volume_cm3']:.2f} cm³\n"
            f"3. 부종 {nums['edema_volume_cm3']:.2f} cm³\n"
            f"4. 괴사 비율 {nums['core_ratio_pct']:.1f}%\n"
            f"5. 병변 수 {nums['lesion_count']}개\n\n"
            f"[결론]\n1. {prediction} 의증\n"
            f"2. LLM 오류로 상세 보고서 미생성. 오류: {e}"
        )
