import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { generateReport } from "../utils/api";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";

// [섹션 제목] 패턴을 파싱해 시각적으로 구분 렌더링
const SECTION_COLORS = {
  "촬영 정보": "text-sky-400 border-sky-700",
  "임상 증상": "text-amber-400 border-amber-700",
  "MRI 소견":  "text-emerald-400 border-emerald-700",
  "결론":      "text-indigo-400 border-indigo-700",
};

function ReportBody({ text }) {
  if (!text) return null;

  // [섹션 제목] 기준으로 분할
  const sectionRegex = /\[([^\]]+)\]/g;
  const parts = [];
  let lastIndex = 0;
  let match;
  const headers = [];

  while ((match = sectionRegex.exec(text)) !== null) {
    if (lastIndex < match.index) {
      parts.push({ type: "text", content: text.slice(lastIndex, match.index) });
    }
    parts.push({ type: "header", content: match[1] });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ type: "text", content: text.slice(lastIndex) });
  }

  return (
    <div className="space-y-3">
      {parts.map((part, i) => {
        if (part.type === "header") {
          const colorClass = SECTION_COLORS[part.content.trim()] ?? "text-gray-400 border-gray-600";
          return (
            <div key={i} className={`flex items-center gap-2 border-b pb-1 mt-4 first:mt-0 ${colorClass}`}>
              <span className="text-xs font-bold uppercase tracking-widest">{part.content}</span>
            </div>
          );
        }
        // 일반 텍스트: 번호 목록·줄바꿈 처리
        const lines = part.content.trim().split("\n").filter(Boolean);
        return (
          <div key={i} className="space-y-1">
            {lines.map((line, j) => {
              const isNumbered = /^\d+\./.test(line.trim());
              const isBullet   = /^[-·•]/.test(line.trim());
              return (
                <p
                  key={j}
                  className={`text-xs leading-relaxed ${
                    isNumbered || isBullet ? "pl-3 text-gray-300" : "text-gray-300"
                  }`}
                >
                  {line.replace(/\*\*(.+?)\*\*/g, "$1")}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

const LABEL_INFO = {
  GLI: {
    name: "Glioma",
    name_ko: "신경교종",
    color: "text-yellow-400",
    bg: "bg-yellow-400",
    desc: "IDH 변이 여부 및 분자 마커 확인 권고",
  },
  MEN: {
    name: "Meningioma",
    name_ko: "수막종",
    color: "text-green-400",
    bg: "bg-green-400",
    desc: "경막 기반 외부 종양 — 수술적 접근 고려",
  },
  MET: {
    name: "Metastasis",
    name_ko: "전이성 종양",
    color: "text-red-400",
    bg: "bg-red-400",
    desc: "원발성 악성종양 병력 확인 및 전신 검사 권고",
  },
};

function ProbBar({ label, prob, isMax }) {
  const info = LABEL_INFO[label];
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className={isMax ? info.color + " font-bold" : "text-gray-400"}>
          {label} — {info.name_ko}
        </span>
        <span className={isMax ? info.color + " font-bold" : "text-gray-400"}>
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-gray-700">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${info.bg}`}
          style={{ width: `${prob * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function ResultPanel({ result, recordId, getScreenshot }) {
  const [report, setReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [reportError, setReportError] = useState(null);
  const [exportingPdf, setExportingPdf] = useState(false);
  const reportRef = useRef(null);

  // 보고서가 생성되면 해당 영역으로 자동 스크롤
  useEffect(() => {
    if (report && reportRef.current) {
      reportRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [report]);

  if (!result) return null;

  const { prediction, confidence, probabilities } = result;
  const info = LABEL_INFO[prediction];

  const handleGenerateReport = async () => {
    setLoadingReport(true);
    setReportError(null);
    try {
      const text = await generateReport(recordId);
      setReport(text);
    } catch (e) {
      setReportError("보고서 생성 실패: " + e.message);
    } finally {
      setLoadingReport(false);
    }
  };

  // HTML div를 html2canvas로 캡처하여 데이터URL 반환
  const captureDiv = async (htmlStr, id) => {
    const container = document.createElement("div");
    container.style.cssText = "position:fixed;left:-9999px;top:0;z-index:-1;";
    container.innerHTML = htmlStr;
    document.body.appendChild(container);
    const el = container.querySelector(`#${id}`);
    const canvas = await html2canvas(el, {
      scale: 2,
      useCORS: true,
      backgroundColor: "#111827",
      logging: false,
    });
    document.body.removeChild(container);
    return canvas;
  };

  // jsPDF 한 페이지에 캔버스 이미지를 꽉 채워 삽입
  const addCanvasPage = (pdf, canvas, addNew = false) => {
    if (addNew) pdf.addPage();
    const pdfW = pdf.internal.pageSize.getWidth();
    const pdfH = pdf.internal.pageSize.getHeight();
    const imgW = pdfW;
    const imgH = (canvas.height / canvas.width) * imgW;
    const imgData = canvas.toDataURL("image/jpeg", 0.95);
    // 이미지가 한 페이지를 초과하면 비율 맞춰 축소
    if (imgH <= pdfH) {
      pdf.addImage(imgData, "JPEG", 0, 0, imgW, imgH);
    } else {
      // 페이지 높이에 맞춰 축소
      const scale = pdfH / imgH;
      pdf.addImage(imgData, "JPEG", 0, 0, imgW * scale, pdfH);
    }
  };

  const handleExportPdf = async () => {
    setExportingPdf(true);
    try {
      const screenshot = getScreenshot?.();

      const predColorMap = {
        GLI: { border: "#FBBF24", text: "#FBBF24" },
        MEN: { border: "#34D399", text: "#34D399" },
        MET: { border: "#F87171", text: "#F87171" },
      };
      const col = predColorMap[prediction] ?? { border: "#6B7280", text: "#6B7280" };
      const dateStr = new Date().toLocaleString("ko-KR");
      const FONT = "'Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',Arial,sans-serif";

      const sortedProbs = Object.entries(probabilities).sort(([, a], [, b]) => b - a);
      const probBarsHtml = sortedProbs.map(([label, prob]) => {
        const c = predColorMap[label] ?? { text: "#9CA3AF", border: "#9CA3AF" };
        const isMax = label === prediction;
        return `
          <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
              <span style="font-size:14px;font-weight:${isMax ? "700" : "400"};color:${isMax ? c.text : "#9CA3AF"};">
                ${label} — ${LABEL_INFO[label].name_ko}
              </span>
              <span style="font-size:14px;font-weight:${isMax ? "700" : "400"};color:${isMax ? c.text : "#9CA3AF"};">
                ${(prob * 100).toFixed(1)}%
              </span>
            </div>
            <div style="background:#374151;border-radius:6px;height:10px;width:100%;">
              <div style="background:${c.border};width:${(prob * 100).toFixed(1)}%;height:10px;border-radius:6px;"></div>
            </div>
          </div>`;
      }).join("");

      // ── PAGE 1: 결과 요약 + 3D 영상 ──────────────────────────────
      const screenshotSection = screenshot
        ? `<div style="margin-top:28px;">
             <div style="font-size:13px;font-weight:700;color:#9CA3AF;margin-bottom:10px;letter-spacing:0.05em;">3D MRI 영상 (NiiVue)</div>
             <img src="${screenshot}" style="width:100%;border-radius:10px;border:1px solid #374151;display:block;" />
           </div>`
        : `<div style="margin-top:28px;padding:40px;text-align:center;color:#6B7280;border:1px dashed #374151;border-radius:10px;">
             3D 영상 캡처 불가 (뷰어에서 MRI 로드 후 내보내기)
           </div>`;

      const page1Html = `
        <div id="pdf-page1" style="width:794px;min-height:1123px;background:#111827;color:#F9FAFB;
          font-family:${FONT};box-sizing:border-box;display:flex;flex-direction:column;">
          <div style="background:#1F2937;padding:20px 36px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #374151;">
            <span style="font-size:22px;font-weight:700;color:#F9FAFB;">Brain Tumor Diagnosis Report</span>
            <span style="font-size:12px;color:#9CA3AF;">${dateStr}</span>
          </div>
          <div style="padding:30px 36px;flex:1;">
            <div style="border:2px solid ${col.border};border-radius:12px;padding:20px 24px;margin-bottom:28px;display:flex;justify-content:space-between;align-items:flex-start;">
              <div>
                <div style="font-size:32px;font-weight:900;color:${col.text};letter-spacing:-1px;">${prediction}</div>
                <div style="font-size:15px;color:#E5E7EB;margin-top:4px;">${LABEL_INFO[prediction].name} — ${LABEL_INFO[prediction].name_ko}</div>
                <div style="font-size:12px;color:#9CA3AF;margin-top:10px;">${LABEL_INFO[prediction].desc}</div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:30px;font-weight:900;color:${col.text};">${(confidence * 100).toFixed(1)}%</div>
                <div style="font-size:12px;color:#9CA3AF;margin-top:2px;">신뢰도</div>
              </div>
            </div>
            <div style="margin-bottom:8px;font-size:12px;font-weight:700;color:#9CA3AF;letter-spacing:0.05em;text-transform:uppercase;">분류 확률</div>
            ${probBarsHtml}
            ${screenshotSection}
          </div>
          <div style="background:#1F2937;padding:12px 36px;border-top:1px solid #374151;">
            <span style="font-size:10px;color:#6B7280;">1 / 2 · 본 보고서는 AI 보조 진단 결과이며, 최종 진단은 전문의의 판독이 필요합니다.</span>
          </div>
        </div>`;

      // ── PAGE 2: 방사선 소견 보고서 ──────────────────────────────
      const reportMd = report
        ? report
            .replace(/\*\*(.+?)\*\*/g, `<span style="font-weight:700;color:#C7D2FE;">$1</span>`)
            .replace(/\n\n/g, "</p><p style='margin:0 0 14px 0;'>")
            .replace(/\n/g, "<br/>")
        : "<span style='color:#6B7280;'>보고서가 생성되지 않았습니다. 보고서 생성 후 PDF를 내보내세요.</span>";

      const page2Html = `
        <div id="pdf-page2" style="width:794px;min-height:1123px;background:#111827;color:#F9FAFB;
          font-family:${FONT};box-sizing:border-box;display:flex;flex-direction:column;">
          <div style="background:#1F2937;padding:20px 36px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #374151;">
            <span style="font-size:22px;font-weight:700;color:#F9FAFB;">방사선 소견 보고서</span>
            <span style="font-size:12px;color:#9CA3AF;">${prediction} · ${(confidence * 100).toFixed(1)}% · ${dateStr}</span>
          </div>
          <div style="padding:36px;flex:1;">
            <div style="display:inline-block;background:#312E81;color:#A5B4FC;font-size:12px;font-weight:700;padding:4px 12px;border-radius:20px;margin-bottom:24px;letter-spacing:0.05em;">
              RAG + LLM (llama-3.3-70b-versatile)
            </div>
            <div style="font-size:15px;line-height:2;color:#E5E7EB;">
              <p style="margin:0 0 14px 0;">${reportMd}</p>
            </div>
          </div>
          <div style="background:#1F2937;padding:12px 36px;border-top:1px solid #374151;">
            <span style="font-size:10px;color:#6B7280;">2 / 2 · 본 보고서는 AI 보조 진단 결과이며, 최종 진단은 전문의의 판독이 필요합니다.</span>
          </div>
        </div>`;

      // 두 페이지 병렬 캡처
      const [canvas1, canvas2] = await Promise.all([
        captureDiv(page1Html, "pdf-page1"),
        captureDiv(page2Html, "pdf-page2"),
      ]);

      const pdf = new jsPDF({ unit: "mm", format: "a4", orientation: "portrait" });
      addCanvasPage(pdf, canvas1, false);
      addCanvasPage(pdf, canvas2, true);

      pdf.save(`brain-tumor-${prediction}-${recordId ?? "report"}.pdf`);
    } catch (err) {
      console.error("PDF 생성 오류:", err);
      alert("PDF 생성 오류: " + err.message);
    } finally {
      setExportingPdf(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* 예측 결과 헤더 */}
      <div className={`rounded-lg p-4 border border-opacity-30 bg-opacity-10 ${info.bg} border-current`}>
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">
              예측 결과
            </p>
            <h3 className={`text-2xl font-bold ${info.color}`}>
              {prediction}
            </h3>
            <p className="text-sm text-gray-300">{info.name} — {info.name_ko}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-400">신뢰도</p>
            <p className={`text-xl font-bold ${info.color}`}>
              {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-2 border-t border-gray-700 pt-2">
          {info.desc}
        </p>
      </div>

      {/* 확률 바 */}
      <div className="space-y-2">
        <p className="text-xs text-gray-400 uppercase tracking-wider">분류 확률</p>
        {Object.entries(probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([label, prob]) => (
            <ProbBar
              key={label}
              label={label}
              prob={prob}
              isMax={label === prediction}
            />
          ))}
      </div>

      {/* RAG 보고서 섹션 */}
      <div>
        {!report ? (
          <button
            onClick={handleGenerateReport}
            disabled={loadingReport}
            className={`w-full py-2 rounded-lg text-sm font-medium transition border ${
              loadingReport
                ? "border-gray-600 text-gray-500 cursor-not-allowed"
                : "border-indigo-500 text-indigo-400 hover:bg-indigo-500 hover:text-white"
            }`}
          >
            {loadingReport ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full" />
                보고서 생성 중 (LLM)...
              </span>
            ) : (
              "📋 방사선 소견 보고서 생성 (RAG + LLM)"
            )}
          </button>
        ) : (
          <div ref={reportRef} className="rounded-lg bg-gray-800 border border-gray-600 p-4 flex flex-col">
            <div className="flex items-center justify-between mb-4 flex-shrink-0">
              <h4 className="text-sm font-semibold text-indigo-300">
                📋 방사선 소견 보고서
              </h4>
              <button
                onClick={() => setReport(null)}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                닫기
              </button>
            </div>
            <ReportBody text={report} />
          </div>
        )}
        {reportError && (
          <p className="text-xs text-red-400 mt-2">{reportError}</p>
        )}
      </div>

      {/* PDF 내보내기 버튼 */}
      <button
        onClick={handleExportPdf}
        disabled={exportingPdf}
        className={`w-full py-2 rounded-lg text-sm font-medium transition border ${
          exportingPdf
            ? "border-gray-600 text-gray-500 cursor-not-allowed"
            : "border-purple-500 text-purple-400 hover:bg-purple-500 hover:text-white"
        }`}
      >
        {exportingPdf ? (
          <span className="flex items-center justify-center gap-2">
            <span className="animate-spin w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full" />
            PDF 생성 중...
          </span>
        ) : (
          "📄 PDF 내보내기 (3D 영상 + 보고서)"
        )}
      </button>
    </div>
  );
}
