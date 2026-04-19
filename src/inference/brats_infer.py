"""
BrainLesion/BraTS 패키지 기반 3종 추론 모듈
- Docker 필수: docker가 실행 중이어야 함
- GLI, MEN, MET 각 모델로 동일 입력 추론 후 segmentation 저장
"""
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from brats import (
    AdultGliomaPreTreatmentSegmenter,
    MeningiomaSegmenter,
    MetastasesSegmenter,
)
from brats.constants import (
    AdultGliomaPreTreatmentAlgorithms,
    MeningiomaAlgorithms,
    MetastasesAlgorithms,
)


ALGORITHM_MAP = {
    "GLI": {
        "BraTS23_1": AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
        "BraTS23_2": AdultGliomaPreTreatmentAlgorithms.BraTS23_2,
        "BraTS23_3": AdultGliomaPreTreatmentAlgorithms.BraTS23_3,
    },
    "MEN": {
        "BraTS23_1": MeningiomaAlgorithms.BraTS23_1,
        "BraTS23_2": MeningiomaAlgorithms.BraTS23_2,
        "BraTS23_3": MeningiomaAlgorithms.BraTS23_3,
        "BraTS25_1": MeningiomaAlgorithms.BraTS25_1,
        "BraTS25_2": MeningiomaAlgorithms.BraTS25_2,
    },
    "MET": {
        "BraTS23_1": MetastasesAlgorithms.BraTS23_1,
        "BraTS23_2": MetastasesAlgorithms.BraTS23_2,
        "BraTS23_3": MetastasesAlgorithms.BraTS23_3,
        "BraTS25_1": MetastasesAlgorithms.BraTS25_1,
        "BraTS25_2": MetastasesAlgorithms.BraTS25_2,
    },
}

SEGMENTER_CLS = {
    "GLI": AdultGliomaPreTreatmentSegmenter,
    "MEN": MeningiomaSegmenter,
    "MET": MetastasesSegmenter,
}


@dataclass
class InferenceInput:
    t1c: Path
    t1n: Path
    t2f: Path
    t2w: Path


def run_single_inference(
    tumor_type: str,
    algorithm_name: str,
    inputs: InferenceInput,
    output_file: Path,
    cuda_devices: str = "0",
) -> Path:
    """
    단일 피험자에 대해 특정 종양 타입 모델로 추론.

    Returns:
        output_file: 저장된 segmentation NIfTI 경로
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    algo_enum = ALGORITHM_MAP[tumor_type][algorithm_name]
    segmenter_cls = SEGMENTER_CLS[tumor_type]

    logger.info(f"[{tumor_type}/{algorithm_name}] 추론 시작 → {output_file.name}")

    segmenter = segmenter_cls(
        algorithm=algo_enum,
        cuda_devices=cuda_devices,
    )
    segmenter.infer_single(
        t1c=str(inputs.t1c),
        t1n=str(inputs.t1n),
        t2f=str(inputs.t2f),
        t2w=str(inputs.t2w),
        output_file=str(output_file),
    )

    logger.info(f"[{tumor_type}/{algorithm_name}] 완료")
    return output_file


def run_all_types(
    subject_id: str,
    inputs: InferenceInput,
    config: dict,
    output_base: Path,
) -> dict[str, Path]:
    """
    동일 입력에 GLI / MEN / MET 세 모델 모두 추론.

    Returns:
        {"GLI": Path, "MEN": Path, "MET": Path}
    """
    results = {}
    infer_cfg = config["inference"]
    cuda = infer_cfg.get("cuda_devices", "0")

    for tumor_type in ["GLI", "MEN", "MET"]:
        algorithm_name = infer_cfg[tumor_type]["algorithm"]
        out_file = output_base / tumor_type / f"{subject_id}.nii.gz"

        try:
            path = run_single_inference(
                tumor_type=tumor_type,
                algorithm_name=algorithm_name,
                inputs=inputs,
                output_file=out_file,
                cuda_devices=cuda,
            )
            results[tumor_type] = path
        except Exception as e:
            logger.error(f"[{tumor_type}] 추론 실패: {e}")
            results[tumor_type] = None

    return results
