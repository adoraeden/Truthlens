import React, { useState } from 'react';
import {
  Image as ImageIcon,
  Zap,
  RotateCw,
  AlertCircle,
} from 'lucide-react';
import { UploadZone } from './UploadZone';
import { VerdictBanner } from './VerdictBanner';
import { ScanningOverlay } from './ScanningOverlay';
import { ForensicVisualizationWorkspace } from './ForensicVisualizationWorkspace';
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
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const explanation = result?.explanation;

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setResult(null);
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
    setErrorMessage(null);
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    if (!backendStatus.connected) {
      setErrorMessage(
        "The Python detector backend is offline. From the backend folder, run .\\.venv\\Scripts\\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8002."
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

      // Record exact completed audit in session history
      const historyItem: HistoryItem = {
        id: `img-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        fileName: selectedFile.name,
        mediaType: 'image',
        fileSizeFormatted: `${(selectedFile.size / 1024).toFixed(1)} KB`,
        verdictDisplay: analysis.verdict,
        scoreDisplay: `Logit ${analysis.logit !== undefined ? (analysis.logit > 0 ? `+${analysis.logit.toFixed(3)}` : analysis.logit.toFixed(3)) : '—'}`,
        details: `Logit: ${analysis.logit !== undefined ? analysis.logit.toFixed(3) : '—'} (threshold ${analysis.decision_threshold.toFixed(3)}) • ${analysis.image_size[0]}×${analysis.image_size[1]}`,
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
              Neural artifact detection and multi-layer forensic diagnostics for synthetic and manipulated imagery.
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
                ViT-Small/16 classifier — 576 non-overlapping 16×16 patches on a 384×384 center crop.
              </p>

              <div className="p-3 rounded-xl bg-[#10111F] border border-white/5 font-mono text-[11px] text-[#94A3B8] space-y-1.5">
                <div className="flex justify-between">
                  <span>MODEL INPUT:</span>
                  <span className="text-white">384 &times; 384</span>
                </div>
                <div className="flex justify-between">
                  <span>PATCH GRID:</span>
                  <span className="text-white">576 (16&times;16)</span>
                </div>
                <div className="flex justify-between">
                  <span>LOGIT CUTOFF:</span>
                  <span className="text-[#62E8FF]">
                    {result ? `≥ ${result.decision_threshold.toFixed(3)}` : '≥ 0.000'}
                  </span>
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
            modelLabel="ViT-Small/16 · 384×384 Model Input"
          />
        )}

        {/* RESULTS SECTION */}
        {result && (
          <div className="mt-8 space-y-4 animate-in fade-in duration-500">
            {/* 1. Main Verdict Banner (Visual Focus) */}
            <VerdictBanner
              verdict={result.verdict}
              score={result.score}
              scoreCutoff={result.score_cutoff}
              logit={result.logit}
              decisionThreshold={result.decision_threshold}
              sourceLabel="ViT-Small/16 Classifier"
              latencyMs={result.inferenceDurationMs}
              thresholdExplanation={`Threshold Rule: Logit at or above ${result.decision_threshold.toFixed(3)} → AI-generated; below ${result.decision_threshold.toFixed(3)} → Likely Real.`}
              subtitle={
                result.verdict === 'AI-generated'
                  ? `Model raw logit crossed the decision threshold (≥ ${result.decision_threshold.toFixed(3)}), flagging synthetic generation patterns.`
                  : `Model raw logit remained below the decision threshold (< ${result.decision_threshold.toFixed(3)}), evaluating as likely authentic.`
              }
            />

            {/* 2. Compact Telemetry Strip */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Source Size
                </div>
                <div className="font-mono text-base font-bold text-white mt-0.5">
                  {result.image_size[0]} &times; {result.image_size[1]}
                </div>
              </div>
              <div className="bg-[#121424] border border-white/5 rounded-xl p-3">
                <div className="font-mono text-[10px] text-[#8290A8] uppercase tracking-wider">
                  Inference Time
                </div>
                <div className="font-mono text-base font-bold text-[#A78BFA] mt-0.5">
                  {result.inferenceDurationMs !== undefined
                    ? `${(result.inferenceDurationMs / 1000).toFixed(2)}s`
                    : '—'}
                </div>
              </div>
            </div>

            {/* 3. Probability Calibration (Kept separate, compact, single short notice) */}
            <section
              className="rounded-xl border border-white/10 bg-[#121424]/90 p-4"
              aria-label="Probability calibration"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs uppercase tracking-wider text-[#A78BFA] font-semibold">
                    Probability Calibration
                  </span>
                  <span
                    className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${
                      result.calibration_status === 'available'
                        ? 'border-[#34D399]/30 bg-[#34D399]/10 text-[#34D399]'
                        : 'border-white/15 bg-white/[0.04] text-[#A6B1C5]'
                    }`}
                  >
                    {result.calibration_status === 'available' ? 'Active' : 'Unavailable'}
                  </span>
                </div>
                <span className="text-[11px] text-[#8290A8]">
                  Independent from threshold verdict
                </span>
              </div>

              {result.calibration_status === 'available' &&
              result.ai_generated_probability !== undefined &&
              result.genuine_probability !== undefined ? (
                <div className="mt-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-[#FF75D8]/20 bg-[#FF75D8]/[0.05] p-3">
                      <div className="text-xs text-[#94A3B8]">AI-generated Probability</div>
                      <div className="mt-1 text-2xl font-bold text-[#FF75D8]">
                        {(result.ai_generated_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="rounded-xl border border-[#62E8FF]/20 bg-[#62E8FF]/[0.05] p-3">
                      <div className="text-xs text-[#94A3B8]">Genuine Probability</div>
                      <div className="mt-1 text-2xl font-bold text-[#62E8FF]">
                        {(result.genuine_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <p className="mt-2 text-[11px] text-[#8290A8]">
                    Calibration method: {result.calibration_method} scaling.
                  </p>
                </div>
              ) : (
                <p className="mt-2 text-xs text-[#94A3B8] leading-relaxed">
                  Calibrated probability is currently unavailable. TruthLens reports raw classification logits; calibrated probabilities are withheld until an independent calibrator is validated.
                </p>
              )}
            </section>

            {/* 4. Forensic Visualization Workspace */}
            <ForensicVisualizationWorkspace originalUrl={previewUrl} explanation={explanation} />
          </div>
        )}
      </div>
    </div>
  );
};
