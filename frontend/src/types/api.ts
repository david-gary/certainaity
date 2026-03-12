export interface SubmitResponse {
  job_id: string;
  poll_url: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  state: "PENDING" | "STARTED" | "SUCCESS" | "FAILURE" | "RETRY";
  stage?: string;
  result?: AnalysisResult;
  error?: string;
}

export interface AnalysisResult {
  job_id: string;
  sha256: string;
  overall_confidence: number;
}

export interface ReportRegion {
  bbox: [number, number, number, number];
  type: "splicing" | "copy_move" | "ai_inpainting" | "removal";
  confidence: number;
  evidence: string;
}

export interface AnalysisReport {
  job_id: string;
  file_name: string;
  sha256: string;
  analysis_timestamp: string;
  manipulation_detected: boolean;
  overall_confidence: number;
  regions: ReportRegion[];
  anti_forensic_warning: boolean;
  models_used: string[];
  execution_time_ms: number;
}
