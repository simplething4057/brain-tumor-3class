import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from "react";
import { Niivue } from "@niivue/niivue";
import { getFileUrl } from "../utils/api";

const COLORMAP_OPTIONS = ["gray", "hot", "cool", "green"];

// 종양 유형별 레이블 → 색상 매핑 (RGBA 0~255)
const SEG_COLORMAPS = {
  GLI: [
    { label: 0, name: "BG",  r: 0,   g: 0,   b: 0,   a: 0   }, // 배경 (투명)
    { label: 1, name: "NCR", r: 220, g: 50,  b: 50,  a: 200 }, // 괴사핵 → 빨강
    { label: 2, name: "ED",  r: 240, g: 200, b: 40,  a: 200 }, // 부종   → 노랑
    { label: 3, name: "ET",  r: 255, g: 255, b: 255, a: 220 }, // 조영증강 → 흰색
  ],
  MEN: [
    { label: 0, name: "BG",   r: 0,   g: 0,   b: 0,   a: 0   },
    { label: 1, name: "ET",   r: 255, g: 255, b: 255, a: 220 }, // 조영증강 → 흰색
    { label: 2, name: "NE_T", r: 220, g: 50,  b: 50,  a: 200 }, // 비조영증강 → 빨강
    { label: 3, name: "SNFH", r: 240, g: 200, b: 40,  a: 200 }, // SNFH → 노랑
  ],
  MET: [
    { label: 0, name: "BG",   r: 0,   g: 0,   b: 0,   a: 0   },
    { label: 1, name: "NETC", r: 220, g: 50,  b: 50,  a: 200 }, // 핵 → 빨강
    { label: 2, name: "SNFH", r: 240, g: 200, b: 40,  a: 200 }, // SNFH → 노랑
    { label: 3, name: "ET",   r: 255, g: 255, b: 255, a: 220 }, // 조영증강 → 흰색
  ],
};

function buildLabelColormap(tumorType) {
  const entries = SEG_COLORMAPS[tumorType] ?? SEG_COLORMAPS["GLI"];
  const maxLabel = 3; // cal_max
  const R = new Array(256).fill(0);
  const G = new Array(256).fill(0);
  const B = new Array(256).fill(0);
  const A = new Array(256).fill(0);
  const I = Array.from({ length: 256 }, (_, i) => i);

  // NiiVue는 label값을 (value/cal_max)*255 위치로 매핑
  // label 0→0, label 1→85, label 2→170, label 3→255
  entries.forEach(({ label, r, g, b, a }) => {
    const idx = Math.round((label / maxLabel) * 255);
    // 인접 인덱스까지 채워서 얇은 경계에서도 색상이 보이게
    for (let d = -2; d <= 2; d++) {
      const i = Math.max(0, Math.min(255, idx + d));
      if (label === 0 && d !== 0) continue; // BG는 정확히 0만
      R[i] = r; G[i] = g; B[i] = b; A[i] = a;
    }
  });

  return { R, G, B, A, I };
}

const NiiVueViewer = forwardRef(function NiiVueViewer({ subjectId, availableModalities, tumorType = "GLI" }, ref) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [activeModality, setActiveModality] = useState(
    availableModalities.includes("t1c") ? "t1c" : availableModalities[0]
  );
  const [showSeg, setShowSeg] = useState(true);
  const [sliceType, setSliceType] = useState(2); // 2=multiplanar, 3=3D render
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!canvasRef.current || !subjectId) return;

    const nv = new Niivue({
      show3Dcrosshair: true,
      backColor: [0.07, 0.07, 0.1, 1],
      crosshairColor: [1, 0, 0, 0.8],
      selectionBoxColor: [1, 1, 1, 0.5],
      isColorbar: false,
    });

    nv.attachToCanvas(canvasRef.current);
    nvRef.current = nv;

    loadVolumes(nv, subjectId, activeModality, showSeg);

    return () => {
      // cleanup
    };
  }, [subjectId]);

  const loadVolumes = async (nv, sid, modality, withSeg) => {
    setIsLoading(true);
    setError(null);
    try {
      const volumes = [
        {
          url: getFileUrl(sid, modality),
          name: `${modality}.nii`,
          colormap: "gray",
          opacity: 1.0,
          visible: true,
        },
      ];

      if (withSeg) {
        volumes.push({
          url: getFileUrl(sid, "seg"),
          name: "seg.nii",
          colormap: "gray",
          opacity: 0.55,
          visible: true,
        });
      }

      await nv.loadVolumes(volumes);

      // 세그 볼륨에 label colormap 적용
      if (withSeg && nv.volumes.length >= 2) {
        const segVol = nv.volumes[nv.volumes.length - 1];
        const labelCmap = buildLabelColormap(tumorType);
        if (typeof nv.setColormapLabel === "function") {
          nv.setColormapLabel(segVol.id, labelCmap);
        } else {
          // fallback: addColormap 방식
          const cmapName = `seg_${tumorType}_${Date.now()}`;
          nv.addColormap(cmapName, labelCmap);
          segVol.colormap = cmapName;
        }
        segVol.cal_min = 0;
        segVol.cal_max = 3;
        nv.updateGLVolume();
      }

      nv.setSliceType(sliceType);
    } catch (e) {
      setError(`NiiVue 로드 실패: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModalityChange = async (mod) => {
    setActiveModality(mod);
    if (nvRef.current && subjectId) {
      await loadVolumes(nvRef.current, subjectId, mod, showSeg);
    }
  };

  const handleToggleSeg = async () => {
    const next = !showSeg;
    setShowSeg(next);
    if (nvRef.current && subjectId) {
      await loadVolumes(nvRef.current, subjectId, activeModality, next);
    }
  };

  const handleSliceType = (type) => {
    setSliceType(type);
    if (nvRef.current) nvRef.current.setSliceType(type);
  };

  // 외부에서 스크린샷을 찍을 수 있도록 ref 노출
  useImperativeHandle(ref, () => ({
    takeScreenshot: () => {
      if (!canvasRef.current || !nvRef.current) return null;
      try {
        // WebGL은 preserveDrawingBuffer:false가 기본이므로
        // drawScene()으로 강제 렌더링 후 즉시 캡처해야 함
        if (typeof nvRef.current.drawScene === "function") {
          nvRef.current.drawScene();
        }
        return canvasRef.current.toDataURL("image/jpeg", 0.92);
      } catch {
        return null;
      }
    },
  }));

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* 뷰어 툴바 */}
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 border-b border-gray-700 flex-wrap">
        {/* 모달리티 선택 */}
        <span className="text-xs text-gray-400 mr-1">모달리티:</span>
        {availableModalities
          .filter((m) => m !== "seg")
          .map((mod) => (
            <button
              key={mod}
              onClick={() => handleModalityChange(mod)}
              className={`px-2 py-1 rounded text-xs font-mono uppercase transition ${
                activeModality === mod
                  ? "bg-brand-500 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
            >
              {mod}
            </button>
          ))}

        <div className="w-px h-4 bg-gray-600 mx-1" />

        {/* Seg 토글 */}
        <button
          onClick={handleToggleSeg}
          className={`px-2 py-1 rounded text-xs transition ${
            showSeg
              ? "bg-orange-600 text-white"
              : "bg-gray-700 text-gray-400 hover:bg-gray-600"
          }`}
        >
          {showSeg ? "SEG ON" : "SEG OFF"}
        </button>

        <div className="w-px h-4 bg-gray-600 mx-1" />

        {/* 슬라이스 뷰 선택 */}
        <span className="text-xs text-gray-400">뷰:</span>
        {[
          { label: "Axial", val: 0 },
          { label: "Multi", val: 2 },
          { label: "3D", val: 3 },
        ].map(({ label, val }) => (
          <button
            key={val}
            onClick={() => handleSliceType(val)}
            className={`px-2 py-1 rounded text-xs transition ${
              sliceType === val
                ? "bg-indigo-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 캔버스 */}
      <div className="relative flex-1 min-h-0">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-80 z-10">
            <div className="text-center">
              <div className="animate-spin w-8 h-8 border-4 border-brand-500 border-t-transparent rounded-full mx-auto mb-2" />
              <p className="text-sm text-gray-300">MRI 로딩 중...</p>
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ touchAction: "none" }}
        />
      </div>
    </div>
  );
});

export default NiiVueViewer;
