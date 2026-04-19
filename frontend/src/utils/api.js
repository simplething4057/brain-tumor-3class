import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 120000,
});

/**
 * seg .nii + 선택적 모달리티 파일로 예측 요청
 * @param {Object} files - { seg, t1c?, t1n?, t2f?, t2w? }
 */
export async function predict(files) {
  const formData = new FormData();
  formData.append("seg", files.seg);
  if (files.t1c) formData.append("t1c", files.t1c);
  if (files.t1n) formData.append("t1n", files.t1n);
  if (files.t2f) formData.append("t2f", files.t2f);
  if (files.t2w) formData.append("t2w", files.t2w);

  const { data } = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * NiiVue에서 사용할 파일 URL 반환
 */
export function getFileUrl(subjectId, modality) {
  return `/api/files/${subjectId}/${modality}`;
}

/**
 * RAG 보고서 생성 요청
 */
export async function generateReport(recordId) {
  const { data } = await api.post(`/report/${recordId}`);
  return data.report;
}

/**
 * 예측 이력 조회
 */
export async function fetchHistory(skip = 0, limit = 50) {
  const { data } = await api.get("/history", { params: { skip, limit } });
  return data;
}

/**
 * 단건 이력 조회
 */
export async function fetchHistoryItem(recordId) {
  const { data } = await api.get(`/history/${recordId}`);
  return data;
}
