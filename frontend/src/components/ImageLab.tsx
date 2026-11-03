import React, { useState } from 'react';
import {
  Image as ImageIcon,
  Zap,
  RotateCw,
  AlertCircle,
  FileCheck,
  Maximize2,
} from 'lucide-react';
import { UploadZone } from './UploadZone';
import { VerdictBanner } from './VerdictBanner';
import { ScanningOverlay } from './ScanningOverlay';
import { ImageAnalysisResult, HistoryItem, BackendStatus } from '../types/forensics';
import { analyzeImage } from '../services/api';

interface ImageLabProps {
  onScanCompleted: (item: HistoryItem) => void;
  backendStatus: BackendStatus;
}

export const ImageLab: React.FC<ImageLabProps> = ({ onScanCompleted, backendStatus }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState('');
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState<ImageAnalysisResult | null>(null);
  const [attributionImageError, setAttributionImageError] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const explanation = result?.explanation;
  const explanationHasImages = Boolean(
    explanation?.overlay_png_base64 && explanation?.crop_png_base64
  );

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setAttributionImageError(false);
    setErrorMessage(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const clearFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setAttributionImageError(false);
    setErrorMessage(null);
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    if (!backendStatus.connected) {
      setErrorMessage(
        "The Python detector backend is offline. From the backend folder, run .\\.venv\\Scripts\\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000."
      );
      return;
    }

    setIsScanning(true);
    setResult(null);
    setErrorMessage(null);
    setScanProgress(15);
    setScanPhase('Preprocessing raster image into 384x384 tensor...');

    const timer1 = setTimeout(() => {
      setScanProgress(45);
      setScanPhase('Preparing the 384 × 384 model input...');
    }, 400);

    const timer2 = setTimeout(() => {
      setScanProgress(80);
      setScanPhase('Scoring the image and computing input attribution...');
    }, 900);

    try {
      const analysis = await analyzeImage(selectedFile);
      clearTimeout(timer1);
      clearTimeout(timer2);
      setScanProgress(100);
      setIsScanning(false);
      setResult(analysis);
      setAttributionImageError(false);

      // Record exact completed audit in session history
      const historyItem: HistoryItem = {
        id: `img-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        fileName: selectedFile.name,
        mediaType: 'image',
        fileSizeFormatted: `${(selectedFile.size / 1024).toFixed(1)} KB`,
        verdictDisplay: analysis.verdict,
        scoreDisplay: `${(analysis.score * 100).toFixed(2)}%`,
        details: `Logit: ${analysis.logit !== undefined ? analysis.logit.toFixed(3) : '—'} • ${analysis.image_size[0]}×${analysis.image_size[1]}`,
        previewUrl: previewUrl || undefined,
        latencyMs: analysis.inferenceDurationMs,
        fullResult: { type: 'image', data: analysis },
      };
      onScanCompleted(historyItem);
    } catch (err: any) {
      clearTimeout(timer1);
      clearTimeout(timer2);
      setIsScanning(false);
      setErrorMessage(err.message || 'Image analysis encountered an error.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="bg-[#191B30]/80 border border-[#62E8FF]/30 rounded-2xl p-6 backdrop-blur-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-white/10 gap-3">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-[#62E8FF]/10 border border-[#62E8FF]/30 flex items-center justify-center text-[#62E8FF]">
                <ImageIcon className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-bold tracking-tight text-[#62E8FF]">
                IMAGE FORENSIC LAB
              </h3>
            </div>
            <p className="text-xs md:text-sm text-[#94A3B8] mt-1.5">
              Production input: RGB; if the long edge exceeds 768 px, it is proportionally downscaled with bilinear interpolation, then JPEG-encoded at quality 60 (4:2:0). Model input preparation: resize the shorter edge to 440 px, center-crop to 384&times;384, then ImageNet-normalize.
            </p>
          </div>

          <span className="font-mono text-xs font-bold px-3 py-1.5 rounded-lg border border-[#62E8FF]/30 bg-[#62E8FF]/10 text-[#62E8FF] self-start sm:self-auto">
            FORMATS: JPG, PNG, WEBP, BMP
          </span>
        </div>

        {/* Upload Zone */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="lg:col-span-2">
            <UploadZone
              onFileSelected={handleFileSelected}
              acceptedFormats={['.JPG', '.JPEG', '.PNG', '.WEBP', '.BMP']}
              acceptedMimeTypes="image/jpeg,image/png,image/webp,image/bmp"
              mediaLabel="image file"
              selectedFile={selectedFile}
              onClearFile={clearFile}
              accentColor="cyan"
            />
          </div>

          {/* Action Card */}
          <div className="flex flex-col justify-between bg-[#121424] border border-white/5 rounded-2xl p-5">
            <div>
              <div className="font-mono text-xs text-[#62E8FF] font-bold uppercase tracking-wider mb-2">
                Inference Protocol
              </div>
              <p className="text-xs text-[#94A3B8] leading-relaxed mb-4">
                Upload any authentic or synthetic image to evaluate against the Community Forensics ViT-Small checkpoint.
              </p>

              <div className="p-3 rounded-xl bg-[#10111F] border border-white/5 font-mono text-[11px] text-[#94A3B8] space-y-1">
                <div className="flex justify-between">
                  <span>PATCH SIZE:</span>
                  <span className="text-white">16 &times; 16</span>
                </div>
                <div className="flex justify-between">
                  <span>INPUT SHAPE:</span>
                  <span className="text-white">384 &times; 384</span>
                </div>
                <div className="flex justify-between">
                  <span>LOGIT CUTOFF:</span>
                  <span className="text-[#62E8FF]">&ge; 1.359375</span>
                </div>
              </div>
            </div>

            <button
              disabled={!selectedFile || isScanning}
              onClick={runAnalysis}
              className="mt-4 w-full py-3 px-4 rounded-xl font-bold text-xs tracking-wider uppercase text-white bg-gradient-to-r from-[#00D2FF] via-[#7928CA] to-[#FF0080] border border-white/20 hover:shadow-[0_0_25px_rgba(98,232,255,0.4)] transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center justify-center gap-2"
            >
              {isScanning ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>ANALYZING WITH ViT...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>ANALYSE IMAGE ⚡</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Backend Error Alert if any */}
        {errorMessage && (
          <div className="mt-5 p-4 rounded-xl bg-[#FF3366]/15 border border-[#FF3366]/40 flex items-start gap-3 text-xs text-[#FF75D8]">
            <AlertCircle className="w-5 h-5 text-[#FF3366] shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Execution Notice:</strong> {errorMessage}
            </div>
          </div>
        )}

        {/* Image Preview Viewport */}
        {previewUrl && (
          <div className="mt-6 pt-6 border-t border-white/10">
            <div className="flex items-center justify-between gap-3 mb-3 font-mono text-xs text-[#62E8FF]">
              <span>TARGET IMAGE PREVIEW</span>
              <span className="text-[#94A3B8]">{selectedFile?.name}</span>
            </div>

            <div className="relative rounded-2xl bg-[#0E0F1C] border border-[#62E8FF]/30 p-4 flex items-center justify-center min-h-[260px] overflow-hidden">
              {/* Corner brackets */}
              <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-[#62E8FF]" />
              <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-[#62E8FF]" />
              <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-[#62E8FF]" />
              <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-[#62E8FF]" />

              <img
                src={previewUrl}
                alt="Uploaded target"
                className="max-h-80 w-auto object-contain rounded-xl border border-white/10"
              />
            </div>
          </div>
        )}

        {/* Scanning Animation */}
        {isScanning && (
          <ScanningOverlay
            currentPhase={scanPhase}
            progressPercent={scanProgress}
            mediaType="image"
            modelLabel="ViT-S/16 · RGB → max 768 px long edge (bilinear) → JPEG q60 4:2:0 → 440 px short edge → 384×384 crop"
          />
        )}

        {/* RESULTS SECTION */}
        {result && (
          <div className="mt-8 space-y-5 animate-in fade-in duration-500">
            <VerdictBanner
              verdict={result.verdict}
              score={result.score}
              sourceLabel="ViT-Small/16 Classifier"
              latencyMs={result.inferenceDurationMs}
              subtitle={
                result.verdict === 'AI-generated'
                  ? 'Model logit output crossed the decision boundary (≥ 1.359375), flagging the sample as synthetic.'
                  : 'Model logit output remained below the decision boundary (< 1.359375), evaluating as likely real according to this model.'
              }
            />

            {/* Detailed Telemetry Chips */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">MODEL LOGIT</div>
                <div className="font-mono text-xl font-bold text-[#A78BFA] mt-0.5">
                  {result.logit !== undefined ? result.logit.toFixed(3) : '—'}
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  Cutoff threshold: 1.359
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">IMAGE DIMENSIONS</div>
                <div className="font-mono text-xl font-bold text-[#62E8FF] mt-0.5">
                  {result.image_size[0]} &times; {result.image_size[1]}
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  Raw input resolution
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">MODEL IDENTIFIER</div>
                <div className="font-mono text-base font-bold text-[#FF75D8] mt-0.5">
                  ViT-Small/16
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  RGB → max 768 px long edge (bilinear) → JPEG q60 4:2:0; model: 440 px short edge → 384&times;384 center crop + ImageNet normalization
                </div>
              </div>

              <div className="bg-[#121424] border border-white/5 rounded-xl p-3.5">
                <div className="font-mono text-[10px] text-[#64748B] uppercase">SPATIAL PATCHES</div>
                <div className="font-mono text-xl font-bold text-[#10B981] mt-0.5">
                  576 Patches
                </div>
                <div className="text-[10px] text-[#64748B] mt-0.5">
                  16&times;16 patch dimensions
                </div>
              </div>
            </div>
            {result.analysis_note && (
              <p className="text-xs text-[#94A3B8] leading-relaxed">
                {result.analysis_note}
              </p>
            )}
            <section className="rounded-2xl bg-[#121424] border border-[#A78BFA]/30 p-5 space-y-3">
              <div>
                <h4 className="font-bold text-sm text-[#A78BFA]">Image comparison and attribution</h4>
                {explanationHasImages && explanation ? (
                  <>
                    <p className="text-xs text-[#94A3B8] mt-1">{explanation.method} · {explanation.scope}</p>
                    <p className="text-xs text-[#94A3B8] mt-1">Approximate input-gradient attribution. It is not Grad-CAM, causal proof, or confirmation of manipulation.</p>
                  </>
                ) : (
                  <p className="text-xs text-[#94A3B8] mt-1">A usable attribution image was not included in this response. The detector verdict and score are still available.</p>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <figure className="space-y-2">
                  {previewUrl ? (
                    <img className="w-full max-h-80 object-contain rounded-xl border border-white/10 bg-black/20" src={previewUrl} alt="Original uploaded image analyzed by TruthLens" />
                  ) : (
                    <p className="text-xs text-[#94A3B8]">Original image preview is unavailable.</p>
                  )}
                  <figcaption className="text-xs text-[#94A3B8]">Original uploaded image</figcaption>
                </figure>
                {explanationHasImages && explanation && !attributionImageError ? (
                  <>
                    <figure className="space-y-2">
                      <img className="w-full max-h-80 object-contain rounded-xl border border-white/10 bg-black/20" src={`data:image/png;base64,${explanation.crop_png_base64}`} alt="Exact resized center crop analyzed by the image model" onError={() => setAttributionImageError(true)} />
                      <figcaption className="text-xs text-[#94A3B8]">Exact model input crop</figcaption>
                    </figure>
                    <figure className="space-y-2">
                      <img className="w-full max-h-80 object-contain rounded-xl border border-white/10 bg-black/20" src={`data:image/png;base64,${explanation.overlay_png_base64}`} alt="Positive input-gradient attribution overlay for the model output" onError={() => setAttributionImageError(true)} />
                      <figcaption className="text-xs text-[#94A3B8]">Attribution overlay (red/yellow indicates stronger positive attribution)</figcaption>
                    </figure>
                  </>
                ) : explanation ? (
                  <p className="md:col-span-2 text-xs text-[#94A3B8]" role="status">The attribution image could not be displayed. The verdict and score are unaffected.</p>
                ) : null}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
};
