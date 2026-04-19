import { useState, useCallback, useRef } from "react";
import FileUploader from "./components/FileUploader";
import NiiVueViewer from "./components/NiiVueViewer";
import ResultPanel from "./components/ResultPanel";
import HistoryPanel from "./components/HistoryPanel";
import { predict } from "./utils/api";

function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
        BT
      </div>
      <div>
        <h1 className="text-sm font-bold text-white leading-none">
          Brain Tumor Diagnosis
        </h1>
        <p className="text-xs text-gray-500">GLI · MEN · MET 3종 분류</p>
      </div>
    </div>
  );
}

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [recordId, setRecordId] = useState(null);
  const [subjectId, setSubjectId] = useState(null);
  const [availableModalities, setAvailableModalities] = useState([]);
  const [activeTab, setActiveTab] = useState("upload"); // "upload" | "history"
  const [error, setError] = useState(null);
  const nvViewerRef = useRef(null);

  const getScreenshot = useCallback(() => {
    return nvViewerRef.current?.takeScreenshot() ?? null;
  }, []);

  const handlePredict = useCallback(async (files) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predict(files);
      setResult({
        prediction: data.prediction,
        confidence: data.confidence,
        probabilities: data.probabilities,
      });
      setRecordId(data.record_id);
      setSubjectId(data.subject_id);

      // 업로드된 모달리티 목록
      const mods = ["seg"];
      if (files.t1c) mods.push("t1c");
      if (files.t1n) mods.push("t1n");
      if (files.t2f) mods.push("t2f");
      if (files.t2w) mods.push("t2w");
      setAvailableModalities(mods);
    } catch (e) {
      const msg =
        e.response?.data?.detail || e.message || "예측 요청 실패";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleHistorySelect = useCallback((rec) => {
    setResult({
      prediction: rec.prediction,
      confidence: rec.confidence,
      probabilities: {
        GLI: rec.gli_prob,
        MEN: rec.men_prob,
        MET: rec.met_prob,
      },
    });
    setRecordId(rec.id);
    setSubjectId(rec.subject_id);
    // 히스토리 케이스는 seg만 가정 (업로드 파일이 없으면 뷰어 미표시)
    setAvailableModalities(["seg"]);
    setActiveTab("upload");
  }, []);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* 헤더 */}
      <header className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800 flex-shrink-0">
        <Logo />
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          RF Classifier · NiiVue · RAG+LLM
        </div>
      </header>

      {/* 메인 레이아웃: 사이드바 + 뷰어 + 결과 */}
      <div className="flex flex-1 overflow-hidden">
        {/* 사이드바 (좌) */}
        <aside className="w-72 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col overflow-hidden">
          {/* 탭 */}
          <div className="flex border-b border-gray-800">
            {["upload", "history"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-xs font-medium uppercase transition ${
                  activeTab === tab
                    ? "text-white border-b-2 border-brand-500"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {tab === "upload" ? "업로드" : "이력"}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === "upload" ? (
              <div className="space-y-4">
                <FileUploader onPredict={handlePredict} isLoading={isLoading} />
                {error && (
                  <div className="rounded-lg bg-red-900 bg-opacity-30 border border-red-700 p-3">
                    <p className="text-xs text-red-400">{error}</p>
                  </div>
                )}
              </div>
            ) : (
              <HistoryPanel
                onSelect={handleHistorySelect}
                currentId={recordId}
              />
            )}
          </div>
        </aside>

        {/* 3D 뷰어 (중앙) */}
        <main className="flex-1 p-3 bg-gray-950 overflow-hidden">
          {subjectId && availableModalities.length > 0 ? (
            <NiiVueViewer
              ref={nvViewerRef}
              subjectId={subjectId}
              availableModalities={availableModalities}
              tumorType={result?.prediction ?? "GLI"}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mb-4">
                <svg
                  className="w-8 h-8 text-gray-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <p className="text-gray-600 text-sm">
                seg.nii 업로드 후 3D 뷰어가 표시됩니다
              </p>
              <p className="text-gray-700 text-xs mt-1">
                T1C/T2F 파일을 함께 올리면 멀티채널 뷰가 활성화됩니다
              </p>
            </div>
          )}
        </main>

        {/* 결과 패널 (우) */}
        {result && (
          <aside className="w-80 flex-shrink-0 bg-gray-900 border-l border-gray-800 overflow-y-auto p-4">
            <ResultPanel result={result} recordId={recordId} getScreenshot={getScreenshot} />
          </aside>
        )}
      </div>
    </div>
  );
}
