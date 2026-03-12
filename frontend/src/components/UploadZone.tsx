import { useCallback, useState } from "react";
import { submitImage } from "../lib/api";
import type { SubmitResponse } from "../types/api";

interface Props {
  token: string;
  onSubmitted: (sub: SubmitResponse) => void;
  onError: (msg: string) => void;
}

const ACCEPTED = ["image/jpeg", "image/png", "image/tiff", "image/webp"];
const MAX_MB = 50;

export default function UploadZone({ token, onSubmitted, onError }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!ACCEPTED.includes(file.type)) {
        onError(`Unsupported format: ${file.type}. Accepted: JPEG, PNG, TIFF, WebP.`);
        return;
      }
      if (file.size > MAX_MB * 1024 * 1024) {
        onError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_MB} MB.`);
        return;
      }
      if (!token) {
        onError("Enter a bearer token before uploading.");
        return;
      }

      setFileName(file.name);
      const url = URL.createObjectURL(file);
      setPreview(url);

      setUploading(true);
      try {
        const sub = await submitImage(file, token);
        onSubmitted(sub);
      } catch (err) {
        onError(err instanceof Error ? err.message : String(err));
      } finally {
        setUploading(false);
      }
    },
    [token, onSubmitted, onError],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold mb-1">Forensic Image Analysis</h1>
        <p className="text-gray-400 text-sm">
          Upload an image to detect manipulation: splicing, copy-move, AI inpainting, and removal.
        </p>
      </div>

      <label
        htmlFor="file-input"
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          flex flex-col items-center justify-center gap-4
          rounded-2xl border-2 border-dashed p-16 cursor-pointer
          transition-colors duration-150
          ${dragging
            ? "border-brand-500 bg-brand-900/20"
            : "border-gray-700 bg-gray-900/40 hover:border-gray-500 hover:bg-gray-900/60"
          }
          ${uploading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input
          id="file-input"
          type="file"
          accept={ACCEPTED.join(",")}
          className="sr-only"
          onChange={onInputChange}
          disabled={uploading}
        />

        {preview ? (
          <img
            src={preview}
            alt={fileName ?? "preview"}
            className="max-h-48 max-w-full rounded-lg object-contain"
          />
        ) : (
          <div className="text-5xl select-none">🔍</div>
        )}

        <div className="text-center">
          {uploading ? (
            <p className="text-brand-400 font-medium">Uploading…</p>
          ) : (
            <>
              <p className="font-medium">
                {fileName ?? "Drop an image here, or click to browse"}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                JPEG · PNG · TIFF · WebP — up to {MAX_MB} MB
              </p>
            </>
          )}
        </div>
      </label>
    </div>
  );
}
