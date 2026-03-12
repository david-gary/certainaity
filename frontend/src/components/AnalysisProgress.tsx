import { useEffect, useRef, useState } from "react";
import { fetchReport, pollJob } from "../lib/api";
import type { AnalysisReport } from "../types/api";

const STAGES = ["ingest", "features", "inference", "resilience", "report"] as const;
type Stage = (typeof STAGES)[number];

const STAGE_LABELS: Record<Stage, string> = {
  ingest: "Ingesting image",
  features: "Extracting forensic features",
  inference: "Running ensemble inference",
  resilience: "Running resilience test",
  report: "Generating report",
};

interface Props {
  jobId: string;
  token: string;
  onComplete: (report: AnalysisReport) => void;
  onError: (msg: string) => void;
}

export default function AnalysisProgress({ jobId, token, onComplete, onError }: Props) {
  const [currentStage, setCurrentStage] = useState<Stage | null>(null);
  const [dots, setDots] = useState("");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const dotsInterval = setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 400);
    return () => clearInterval(dotsInterval);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const status = await pollJob(jobId, token);

          if (status.stage && STAGES.includes(status.stage as Stage)) {
            setCurrentStage(status.stage as Stage);
          }

          if (status.state === "SUCCESS") {
            const report = await fetchReport(jobId, token);
            if (!cancelled) onComplete(report);
            return;
          }

          if (status.state === "FAILURE") {
            if (!cancelled) onError(status.error ?? "Analysis failed.");
            return;
          }
        } catch (err) {
          if (!cancelled) onError(err instanceof Error ? err.message : String(err));
          return;
        }

        await new Promise((r) => setTimeout(r, 2000));
      }
    }

    poll();
    return () => { cancelled = true; };
  }, [jobId, token, onComplete, onError]);

  const currentIdx = currentStage ? STAGES.indexOf(currentStage) : -1;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold mb-1">Analyzing image{dots}</h2>
        <p className="text-gray-400 text-sm font-mono">Job ID: {jobId}</p>
      </div>

      <div className="space-y-3">
        {STAGES.map((stage, idx) => {
          const done = idx < currentIdx;
          const active = idx === currentIdx;
          const pending = idx > currentIdx;

          return (
            <div key={stage} className="flex items-center gap-4">
              <div
                className={`
                  w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0
                  ${done ? "bg-green-600 text-white" : ""}
                  ${active ? "bg-brand-500 text-white animate-pulse" : ""}
                  ${pending ? "bg-gray-800 text-gray-500" : ""}
                `}
              >
                {done ? "✓" : idx + 1}
              </div>
              <span
                className={`text-sm ${active ? "text-white font-medium" : done ? "text-green-400" : "text-gray-600"}`}
              >
                {STAGE_LABELS[stage]}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
