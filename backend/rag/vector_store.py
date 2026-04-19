"""
ChromaDB 벡터스토어 초기화 및 WHO CNS 문서 로드
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from loguru import logger

from core.config import settings

WHO_CNS_DOCS = [
    {
        "id": "gli_def",
        "text": (
            "Glioma (GLI): Diffuse gliomas include IDH-mutant astrocytoma (Grade 2-4), "
            "IDH-mutant oligodendroglioma (Grade 2-3), and IDH-wildtype glioblastoma (Grade 4). "
            "WHO 2021 classification emphasizes molecular markers: IDH mutation, 1p/19q codeletion, "
            "TERT promoter mutation, EGFR amplification. Key MRI features: T2/FLAIR hyperintensity, "
            "ring enhancement in GBM, infiltrative margins. Peritumoral edema common in high-grade."
        ),
        "metadata": {"category": "tumor_type", "label": "GLI"},
    },
    {
        "id": "men_def",
        "text": (
            "Meningioma (MEN): Arise from meningothelial cells of the arachnoid layer. "
            "WHO Grade 1 (benign, 80%), Grade 2 (atypical, 15-20%), Grade 3 (anaplastic, rare). "
            "MRI: Extra-axial mass with dural tail sign, homogeneous enhancement, T1 iso/hypointense, "
            "T2 variable. Peritumoral edema may be present. Location: convexity, falx, sphenoid wing, "
            "posterior fossa. Calcification seen in 25%. No peritumoral enhancement beyond dural attachment."
        ),
        "metadata": {"category": "tumor_type", "label": "MEN"},
    },
    {
        "id": "met_def",
        "text": (
            "Brain Metastasis (MET): Secondary tumors from systemic malignancies. "
            "Common primaries: lung (40-50%), breast (15-25%), melanoma (5-20%), colorectal, renal. "
            "MRI: Well-circumscribed lesions at gray-white junction, ring or solid enhancement, "
            "significant surrounding vasogenic edema disproportionate to tumor size. "
            "Multiple lesions in 70% of cases. T1 hypointense core with rim enhancement. "
            "Leptomeningeal spread possible. DWI restriction in dense cellular tumors."
        ),
        "metadata": {"category": "tumor_type", "label": "MET"},
    },
    {
        "id": "mri_features",
        "text": (
            "MRI Radiological Terminology: "
            "T1WI (T1-weighted imaging): anatomical detail, blood products appear hyperintense. "
            "T2/FLAIR: edema and infiltration appear hyperintense. "
            "T1C (post-contrast T1): BBB breakdown shown as enhancement. "
            "ADC map: restricted diffusion (low ADC) indicates high cellularity. "
            "Enhancing Tumor (ET): active tumor with BBB disruption. "
            "Necrotic Core (NCR): central necrosis in aggressive tumors. "
            "Peritumoral Edema (ED/SNFH): surrounding vasogenic edema. "
            "Tumor volume (cm³), lesion count, ET/edema/necrosis ratios are key quantitative metrics."
        ),
        "metadata": {"category": "radiology", "label": "general"},
    },
    {
        "id": "gli_report_template",
        "text": (
            "Glioma radiology report template: "
            "Findings consistent with diffuse glioma. Intra-axial mass with infiltrative margins. "
            "T2/FLAIR hyperintense signal involving [region]. Post-contrast enhancement pattern: "
            "[ring/nodular/heterogeneous]. Enhancing tumor volume: [X] cm³. "
            "Surrounding T2-FLAIR signal abnormality (edema/infiltration): [X] cm³. "
            "Mass effect with [midline shift/sulcal effacement]. "
            "Impression: imaging features most consistent with high-grade glioma (GBM vs. Grade 3). "
            "Molecular correlation recommended (IDH, MGMT, 1p/19q)."
        ),
        "metadata": {"category": "report_template", "label": "GLI"},
    },
    {
        "id": "men_report_template",
        "text": (
            "Meningioma radiology report template: "
            "Extra-axial mass arising from the [location] with broad dural base. "
            "T1 isointense, T2 iso-to-hyperintense. Homogeneous avid post-contrast enhancement. "
            "Dural tail sign [present/absent]. Adjacent bone hyperostosis [present/absent]. "
            "Internal calcification [present/absent]. Peritumoral edema [minimal/moderate/significant]. "
            "No restricted diffusion. Mass effect on adjacent cortex: [description]. "
            "Impression: Findings consistent with WHO Grade 1 meningioma. "
            "Surgical planning with venous sinus involvement assessment recommended."
        ),
        "metadata": {"category": "report_template", "label": "MEN"},
    },
    {
        "id": "met_report_template",
        "text": (
            "Brain metastasis radiology report template: "
            "Multiple/single [N] enhancing lesion(s) at the gray-white matter junction. "
            "Ring/nodular enhancement pattern. Significant surrounding vasogenic edema. "
            "Largest lesion: [size] cm in [location]. "
            "Total enhancing tumor burden: [X] cm³. "
            "Leptomeningeal involvement: [present/absent]. "
            "Impression: MRI appearance consistent with brain metastatic disease. "
            "Correlation with systemic malignancy history recommended. "
            "Stereotactic radiosurgery (SRS) vs. whole-brain radiation therapy (WBRT) discussion warranted."
        ),
        "metadata": {"category": "report_template", "label": "MET"},
    },
]


class VectorStore:
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name="brain_tumor_rag",
                metadata={"hnsw:space": "cosine"},
            )
            # 초기 문서 로드
            if self._collection.count() == 0:
                self._load_initial_docs()
        return self._collection

    def _load_initial_docs(self):
        embedder = self._get_embedder()
        collection = self._collection
        texts = [d["text"] for d in WHO_CNS_DOCS]
        ids = [d["id"] for d in WHO_CNS_DOCS]
        metadatas = [d["metadata"] for d in WHO_CNS_DOCS]
        embeddings = embedder.encode(texts).tolist()
        collection.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)
        logger.info(f"RAG 초기 문서 {len(ids)}개 로드 완료")

    def query(self, query_text: str, n_results: int = 4, label_filter: str = None) -> list[str]:
        collection = self.get_collection()
        embedder = self._get_embedder()
        query_emb = embedder.encode([query_text]).tolist()

        where = None
        if label_filter:
            where = {"$or": [{"label": label_filter}, {"label": "general"}]}

        results = collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            where=where,
        )
        return results["documents"][0] if results["documents"] else []


# 싱글턴
vector_store = VectorStore()
