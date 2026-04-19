import { useEffect, useState } from "react";
import { fetchHistory } from "../utils/api";

const LABEL_BADGE = {
  GLI: "bg-yellow-900 text-yellow-300",
  MEN: "bg-green-900 text-green-300",
  MET: "bg-red-900 text-red-300",
};

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleString("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HistoryPanel({ onSelect, currentId }) {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchHistory(0, 50);
      setRecords(data);
    } catch (e) {
      console.error("이력 로드 실패:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          분석 이력
        </h2>
        <button
          onClick={load}
          className="text-xs text-gray-500 hover:text-gray-300 transition"
        >
          새로고침
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center py-4">
          <div className="animate-spin w-5 h-5 border-2 border-brand-500 border-t-transparent rounded-full" />
        </div>
      ) : records.length === 0 ? (
        <p className="text-xs text-gray-500 text-center py-4">이력 없음</p>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
          {records.map((rec) => (
            <button
              key={rec.id}
              onClick={() => onSelect(rec)}
              className={`w-full text-left rounded-lg p-3 border transition ${
                currentId === rec.id
                  ? "border-brand-500 bg-brand-500 bg-opacity-10"
                  : "border-gray-700 bg-gray-800 hover:border-gray-500"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                    LABEL_BADGE[rec.prediction]
                  }`}
                >
                  {rec.prediction}
                </span>
                <span className="text-xs text-gray-500">
                  {formatDate(rec.created_at)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400 font-mono truncate max-w-[140px]">
                  #{rec.id} {rec.subject_id.slice(0, 8)}...
                </span>
                <span className="text-xs text-gray-300">
                  {(rec.confidence * 100).toFixed(0)}%
                </span>
              </div>
              {rec.report && (
                <span className="text-xs text-indigo-400 mt-1 block">
                  📋 보고서 있음
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
