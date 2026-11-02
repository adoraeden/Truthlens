import React from 'react';
import {
  Layers,
  Image as ImageIcon,
  Mic,
  Video,
  AlertTriangle,
  Info,
  Terminal,
  Cpu,
  CheckCircle2,
} from 'lucide-react';
import { BackendStatus } from '../types/forensics';

interface MethodologyViewProps {
  backendStatus: BackendStatus;
}

export const MethodologyView: React.FC<MethodologyViewProps> = ({ backendStatus }) => {
  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="bg-[#191B30]/80 border border-[#62E8FF]/30 rounded-2xl p-6 md:p-8 backdrop-blur-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-white/10 gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-[#62E8FF]/10 border border-[#62E8FF]/30 flex items-center justify-center text-[#62E8FF]">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="text-xl md:text-2xl font-bold tracking-tight text-white">
                SYSTEM ARCHITECTURE &amp; METHODOLOGY
              </h3>
            </div>
            <p className="text-xs md:text-sm text-[#94A3B8] mt-1.5 max-w-3xl leading-relaxed">
              Technical specification of the underlying machine learning models, preprocessing pipelines,
              sampling strides, and operational boundaries of TruthLens.
            </p>
          </div>

          <span
            className={`font-mono text-xs font-bold px-3 py-1.5 rounded-lg border self-start sm:self-auto ${
              backendStatus.connected
                ? 'bg-[#10B981]/10 text-[#10B981] border-[#10B981]/30'
                : 'bg-[#A78BFA]/10 text-[#A78BFA] border-[#A78BFA]/30'
            }`}
          >
            {backendStatus.connected ? 'BACKEND LINKED (:8000)' : 'STANDBY MODE'}
          </span>
        </div>

        {/* 3 Core Detection Pipelines */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* IMAGE METHODOLOGY */}
          <div className="rounded-2xl bg-[#121424] border border-[#62E8FF]/30 p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <ImageIcon className="w-5 h-5 text-[#62E8FF]" />
                <h4 className="font-bold text-sm text-[#62E8FF] uppercase tracking-wide">
                  Image Modality Pipeline
                </h4>
              </div>

              <div className="space-y-2 text-xs text-[#94A3B8] leading-relaxed">
                <div>
                  <strong className="text-white">Model Architecture:</strong> Fine-tuned ViT-Small/16 image classifier.
                </div>
                <div>
                  <strong className="text-white">Spatial Input:</strong> RGB conversion, shorter-edge resize to 440 pixels, then a 384 &times; 384 center crop with ImageNet normalization. Pixels outside the crop are not analyzed.
                </div>
                <div>
                  <strong className="text-white">Patch Tokenization:</strong> Non-overlapping 16 &times; 16 pixel patches producing exactly <span className="text-[#62E8FF] font-mono">576 spatial patches</span> passed to the Vision Transformer.
                </div>
                <div>
                  <strong className="text-white">Thresholding &amp; Scoring:</strong> Raw-logit cutoff <span className="text-[#62E8FF] font-mono">1.359375</span>. The displayed model score is centered on this cutoff, so it is 50% where the verdict changes. It is not a calibrated probability.
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-white/5 font-mono text-[10px] text-[#64748B]">
              SOURCE: Thermostatic/community-forensics-frontier-detector-2026-08
            </div>
          </div>

          {/* AUDIO METHODOLOGY */}
          <div className="rounded-2xl bg-[#121424] border border-[#A78BFA]/30 p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Mic className="w-5 h-5 text-[#A78BFA]" />
                <h4 className="font-bold text-sm text-[#A78BFA] uppercase tracking-wide">
                  Audio Modality Pipeline
                </h4>
              </div>

              <div className="space-y-2 text-xs text-[#94A3B8] leading-relaxed">
                <div>
                  <strong className="text-white">Model Architecture:</strong> PellaV2 audio deepfake detector built on a Wav2Vec-BERT feature backbone.
                </div>
                <div>
                  <strong className="text-white">Preprocessing Pipeline:</strong> FFmpeg converts incoming audio containers to 16,000 Hz, 1-channel mono PCM (<span className="text-[#A78BFA] font-mono">pcm_s16le</span>).
                </div>
                <div>
                  <strong className="text-white">Temporal Window:</strong> Analyzes up to four seconds. Longer recordings use a center crop; shorter recordings are analyzed in full and zero-padded to the model input length.
                </div>
                <div>
                  <strong className="text-white">Decision Rule:</strong> The model score cutoff is 0.50. Scores are not calibrated probabilities.
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-white/5 font-mono text-[10px] text-[#64748B]">
              SOURCE: PellaV2 Neural Audio Detector (models/pellav2)
            </div>
          </div>

          {/* VIDEO METHODOLOGY */}
          <div className="rounded-2xl bg-[#121424] border border-[#FF75D8]/30 p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Video className="w-5 h-5 text-[#FF75D8]" />
                <h4 className="font-bold text-sm text-[#FF75D8] uppercase tracking-wide">
                  Video Modality Pipeline
                </h4>
              </div>

              <div className="space-y-2 text-xs text-[#94A3B8] leading-relaxed">
                <div>
                  <strong className="text-white">Dual-Pipeline Structure:</strong> Visual video frames and the demuxed audio track are evaluated independently.
                </div>
                <div>
                  <strong className="text-white">Frame Sampling:</strong> Samples up to eight frames at midpoints of uniform time intervals. The sample timestamps are shown; other moments are not inspected.
                </div>
                <div>
                  <strong className="text-white">Visual Inference:</strong> Each extracted frame is evaluated using the ViT-Small/16 image detector.
                </div>
                <div>
                  <strong className="text-white">Audio Inference:</strong> Extracted audio is evaluated separately using the PellaV2 audio detector.
                </div>
                <div>
                  <strong className="text-white">Video Summary:</strong> Image and audio findings stay separate. The mean sampled-frame score is not a probability for the full video.
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-white/5 font-mono text-[10px] text-[#64748B]">
              INDEPENDENT DUAL-MODAL EVALUATION
            </div>
          </div>
        </div>
      </div>

      {/* Verified Limitations Section */}
      <div className="rounded-2xl bg-[#14162B] border border-[#FF75D8]/30 p-6 md:p-8 backdrop-blur-xl">
        <div className="flex items-center gap-2.5 mb-4 text-[#FF75D8]">
          <AlertTriangle className="w-5 h-5" />
          <h4 className="font-bold text-base tracking-tight">
            CRITICAL FORENSIC LIMITATIONS
          </h4>
        </div>

        <p className="text-xs md:text-sm text-[#94A3B8] leading-relaxed mb-6">
          Automated media screening provides model indicators, not definitive proof of authenticity.
          The following constraints are inherent to the underlying detector models:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-[#10111F] border border-white/5 space-y-1.5">
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#FF75D8]" />
              Scores Are Not Calibrated Probabilities
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Model outputs have not been calibrated against representative real-world data. A 90% score does not mean
              there is a 90% statistical chance the file is synthetic.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#10111F] border border-white/5 space-y-1.5">
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#FF75D8]" />
              Authentic Media Can Be Falsely Flagged
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Aggressive compression, lens blur, social media re-encoding, and denoising filters can perturb image/audio
              statistics, resulting in false positives.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#10111F] border border-white/5 space-y-1.5">
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#FF75D8]" />
              Re-Recorded Synthetic Speech May Evade Detection
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Synthetic audio played through an acoustic speaker and re-recorded via a microphone acquires real room
              reverberation and sensor noise that can mask generative characteristics.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#10111F] border border-white/5 space-y-1.5">
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#FF75D8]" />
              Unsampled Video Frames Are Not Inspected
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Video analysis samples up to eight uniform keyframes across runtime. Brief temporal deepfake inserts or
              face-swaps that occur between sampled timestamps will not be captured.
            </p>
          </div>
        </div>
      </div>

      {/* Backend Integration Instructions */}
      <div className="rounded-2xl bg-[#14162B] border border-white/10 p-6 md:p-8">
        <div className="flex items-center gap-2.5 mb-3 text-white">
          <Terminal className="w-5 h-5 text-[#62E8FF]" />
          <h4 className="font-bold text-base tracking-tight">
            Python Backend REST Integration
          </h4>
        </div>
        <p className="text-xs md:text-sm text-[#94A3B8] leading-relaxed mb-4">
          TruthLens invokes the Python image, audio, and video detectors through the FastAPI service.
        </p>

        <div className="bg-[#0B0C16] border border-white/10 rounded-xl p-4 font-mono text-xs text-[#94A3B8] space-y-2">
          <div className="text-[#62E8FF] font-bold"># Start the Python Detector REST API Server:</div>
          <div className="text-white bg-[#191B30] p-2.5 rounded-lg select-all">
            python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000
          </div>
          <div className="text-[11px] text-[#64748B] pt-1">
            The React frontend automatically communicates with endpoints: 
            <span className="text-[#A78BFA]"> /api/health</span>, 
            <span className="text-[#62E8FF]"> /api/detect/image</span>, 
            <span className="text-[#A78BFA]"> /api/detect/audio</span>, 
            <span className="text-[#FF75D8]"> /api/detect/video</span>.
          </div>
        </div>
      </div>
    </div>
  );
};
