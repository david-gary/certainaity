import { useState } from "react";
import UploadZone from "./components/UploadZone";
import AnalysisProgress from "./components/AnalysisProgress";
import ReportView from "./components/ReportView";
import TokenInput from "./components/TokenInput";
import type { AnalysisReport, SubmitResponse } from "./types/api";

type Phase = "idle" | "polling" | "done" | "error";

export default function App() {
  const [token, setToken] = useState("");
  const [submission, setSubmission] = useState<SubmitResponse | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [errorMsg, setErrorMsg] = useState("");

  function handleSubmitted(sub: SubmitResponse) {
    setSubmission(sub);
    setPhase("polling");
    setReport(null);
    setErrorMsg("");
  }

  function handleComplete(r: AnalysisReport) {
    setReport(r);
    setPhase("done");
  }

  function handleError(msg: string) {
    setErrorMsg(msg);
    setPhase("error");
  }

  function reset() {
    setSubmission(null);
    setReport(null);
    setPhase("idle");
    setErrorMsg("");
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold tracking-tight text-white">
            Certain<span className="text-brand-500">aity</span>
          </span>
          <span className="text-xs text-gray-500 mt-1">v1.1</span>
        </div>
        <TokenInput value={token} onChange={setToken} />
      </header>

      <main className="flex-1 p-6 max-w-4xl mx-auto w-full">
        {phase === "idle" && (
          <UploadZone
            token={token}
            onSubmitted={handleSubmitted}
            onError={handleError}
          />
        )}

        {phase === "polling" && submission && (
          <AnalysisProgress
            jobId={submission.job_id}
            token={token}
            onComplete={handleComplete}
            onError={handleError}
          />
        )}

        {phase === "done" && report && (
          <>
            <ReportView report={report} token={token} />
            <button
              onClick={reset}
              className="mt-6 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
            >
              Analyze another image
            </button>
          </>
        )}

        {phase === "error" && (
          <div className="rounded-lg border border-red-800 bg-red-950/40 p-6 text-center">
            <p className="text-red-400 font-medium mb-4">{errorMsg}</p>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
            >
              Try again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
