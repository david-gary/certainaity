import { pdfUrl } from "../lib/api";
import type { AnalysisReport, ReportRegion } from "../types/api";

const TYPE_BADGE: Record<string, string> = {
  splicing: "bg-red-900 text-red-300",
  copy_move: "bg-yellow-900 text-yellow-300",
  ai_inpainting: "bg-purple-900 text-purple-300",
  removal: "bg-orange-900 text-orange-300",
};

interface Props {
  report: AnalysisReport;
  token: string;
}

function Verdict({ report }: { report: AnalysisReport }) {
  const pct = (report.overall_confidence * 100).toFixed(1);
  if (report.manipulation_detected) {
    return (
      <div className="rounded-xl bg-red-950/50 border border-red-800 p-5">
        <p className="text-red-400 font-bold text-lg">MANIPULATION DETECTED</p>
        <p className="text-gray-300 text-sm mt-1">
          Overall confidence: <span className="font-semibold">{pct}%</span>
        </p>
        {report.anti_forensic_warning && (
          <p className="text-orange-400 text-sm mt-2 font-medium">
            Warning: anti-forensic processing suspected — confidence dropped under re-compression.
          </p>
        )}
      </div>
    );
  }
  return (
    <div className="rounded-xl bg-green-950/50 border border-green-800 p-5">
      <p className="text-green-400 font-bold text-lg">NO MANIPULATION DETECTED</p>
      <p className="text-gray-300 text-sm mt-1">
        Overall confidence: <span className="font-semibold">{pct}%</span>
      </p>
    </div>
  );
}

function RegionRow({ region, index }: { region: ReportRegion; index: number }) {
  const [x, y, w, h] = region.bbox;
  const badge = TYPE_BADGE[region.type] ?? "bg-gray-800 text-gray-300";
  return (
    <tr className="border-t border-gray-800">
      <td className="py-2 px-3 text-gray-500 text-sm">{index + 1}</td>
      <td className="py-2 px-3">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${badge}`}>
          {region.type.replace("_", " ")}
        </span>
      </td>
      <td className="py-2 px-3 text-sm font-mono text-gray-300">
        {x},{y} {w}×{h}
      </td>
      <td className="py-2 px-3 text-sm text-gray-300">
        {(region.confidence * 100).toFixed(1)}%
      </td>
      <td className="py-2 px-3 text-xs text-gray-500 max-w-xs truncate">
        {region.evidence}
      </td>
    </tr>
  );
}

export default function ReportView({ report, token: _token }: Props) {
  const ts = new Date(report.analysis_timestamp).toLocaleString();

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold">Analysis Report</h2>
          <p className="text-sm text-gray-400 mt-0.5">{ts}</p>
        </div>
        <a
          href={pdfUrl(report.job_id)}
          download={`forenscope-${report.job_id}.pdf`}
          className="px-4 py-2 bg-brand-700 hover:bg-brand-500 rounded-lg text-sm font-medium transition-colors"
        >
          Download PDF
        </a>
      </div>

      <Verdict report={report} />

      <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5 space-y-3">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
          Chain of Custody
        </h3>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
          <dt className="text-gray-500">File</dt>
          <dd className="text-gray-200">{report.file_name}</dd>
          <dt className="text-gray-500">Job ID</dt>
          <dd className="font-mono text-gray-200 text-xs">{report.job_id}</dd>
          <dt className="text-gray-500">SHA-256</dt>
          <dd className="font-mono text-gray-200 text-xs break-all">{report.sha256}</dd>
          <dt className="text-gray-500">Models</dt>
          <dd className="text-gray-200">{report.models_used.join(", ")}</dd>
          <dt className="text-gray-500">Execution</dt>
          <dd className="text-gray-200">{report.execution_time_ms} ms</dd>
        </dl>
      </div>

      {report.regions.length > 0 && (
        <div className="rounded-xl border border-gray-800 overflow-hidden">
          <div className="px-5 py-3 bg-gray-900/70 border-b border-gray-800">
            <h3 className="text-sm font-semibold text-gray-300">
              Detected Regions ({report.regions.length})
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-xs text-gray-500 text-left">
                  <th className="py-2 px-3">#</th>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Bounding Box</th>
                  <th className="py-2 px-3">Confidence</th>
                  <th className="py-2 px-3">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {report.regions.map((r, i) => (
                  <RegionRow key={i} region={r} index={i} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
