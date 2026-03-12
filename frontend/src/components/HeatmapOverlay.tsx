import { useEffect, useRef } from "react";
import type { ReportRegion } from "../types/api";

const TYPE_COLORS: Record<string, string> = {
  splicing: "rgba(239,68,68,0.65)",
  copy_move: "rgba(234,179,8,0.65)",
  ai_inpainting: "rgba(168,85,247,0.65)",
  removal: "rgba(249,115,22,0.65)",
};

const TYPE_LABELS: Record<string, string> = {
  splicing: "Splicing",
  copy_move: "Copy-move",
  ai_inpainting: "AI inpainting",
  removal: "Removal",
};

interface Props {
  imageSrc: string;
  regions: ReportRegion[];
  naturalWidth: number;
  naturalHeight: number;
}

export default function HeatmapOverlay({
  imageSrc,
  regions,
  naturalWidth,
  naturalHeight,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || regions.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const scaleX = canvas.width / naturalWidth;
    const scaleY = canvas.height / naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const region of regions) {
      const [x, y, w, h] = region.bbox;
      ctx.fillStyle = TYPE_COLORS[region.type] ?? "rgba(239,68,68,0.65)";
      ctx.fillRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);

      ctx.strokeStyle = ctx.fillStyle.replace("0.65", "1");
      ctx.lineWidth = 2;
      ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);

      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillRect(x * scaleX + 2, y * scaleY + 2, 90, 18);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(
        `${TYPE_LABELS[region.type] ?? region.type} ${(region.confidence * 100).toFixed(0)}%`,
        x * scaleX + 5,
        y * scaleY + 14,
      );
    }
  }, [regions, naturalWidth, naturalHeight]);

  return (
    <div className="relative inline-block">
      <img
        ref={imgRef}
        src={imageSrc}
        alt="Analyzed image"
        className="rounded-lg max-w-full"
        style={{ display: "block" }}
      />
      <canvas
        ref={canvasRef}
        width={naturalWidth}
        height={naturalHeight}
        className="absolute inset-0 rounded-lg pointer-events-none"
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
