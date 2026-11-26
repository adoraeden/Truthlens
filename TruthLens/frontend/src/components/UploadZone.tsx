import React, { useState } from 'react';
import { UploadCloud, FileCheck, X, HardDrive } from 'lucide-react';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  acceptedFormats: string[];
  acceptedMimeTypes: string;
  mediaLabel: string;
  selectedFile: File | null;
  onClearFile: () => void;
  accentColor?: 'cyan' | 'violet' | 'pink';
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onFileSelected,
  acceptedFormats,
  acceptedMimeTypes,
  mediaLabel,
  selectedFile,
  onClearFile,
  accentColor = 'cyan',
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  };

  const accentClasses = {
    cyan: {
      border: isDragOver ? 'border-[#62E8FF] shadow-[0_0_25px_rgba(98,232,255,0.4)]' : 'border-[#62E8FF]/30 hover:border-[#62E8FF]/80',
      iconText: 'text-[#62E8FF]',
      chipBg: 'bg-[#62E8FF]/10 text-[#62E8FF] border-[#62E8FF]/30',
    },
    violet: {
      border: isDragOver ? 'border-[#A78BFA] shadow-[0_0_25px_rgba(167,139,250,0.4)]' : 'border-[#A78BFA]/30 hover:border-[#A78BFA]/80',
      iconText: 'text-[#A78BFA]',
      chipBg: 'bg-[#A78BFA]/10 text-[#A78BFA] border-[#A78BFA]/30',
    },
    pink: {
      border: isDragOver ? 'border-[#FF75D8] shadow-[0_0_25px_rgba(255,117,216,0.4)]' : 'border-[#FF75D8]/30 hover:border-[#FF75D8]/80',
      iconText: 'text-[#FF75D8]',
      chipBg: 'bg-[#FF75D8]/10 text-[#FF75D8] border-[#FF75D8]/30',
    },
  }[accentColor];

  if (selectedFile) {
    return (
      <div className="relative rounded-2xl bg-[#191B30]/80 border border-[#62E8FF]/40 p-5 backdrop-blur-md shadow-lg shadow-[#62E8FF]/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-[#62E8FF]/10 border border-[#62E8FF]/30 flex items-center justify-center text-[#62E8FF]">
              <FileCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="text-sm font-bold text-white truncate max-w-sm md:max-w-md">
                {selectedFile.name}
              </div>
              <div className="font-mono text-xs text-[#94A3B8] flex items-center gap-2 mt-0.5">
                <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                <span>&bull;</span>
                <span className="uppercase">{selectedFile.type || 'RAW STREAM'}</span>
                <span>&bull;</span>
                <span className="text-[#10B981]">PAYLOAD BUFFER STAGED</span>
              </div>
            </div>
          </div>

          <button
            onClick={onClearFile}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-[#94A3B8] hover:text-[#FF75D8] border border-white/10 transition-colors"
            title="Remove selected file"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`relative rounded-2xl border-2 border-dashed ${accentClasses.border} bg-[#141527]/90 hover:bg-[#191B30]/90 p-8 text-center transition-all cursor-pointer group backdrop-blur-sm`}
    >
      <input
        type="file"
        onChange={handleChange}
        accept={acceptedMimeTypes}
        className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
      />

      <div className="flex flex-col items-center justify-center pointer-events-none">
        <div className="w-14 h-14 rounded-2xl bg-[#191B30] border border-white/10 flex items-center justify-center mb-3.5 group-hover:scale-110 transition-transform shadow-lg">
          <UploadCloud className={`w-7 h-7 ${accentClasses.iconText} transition-colors`} />
        </div>

        <div className="text-base font-bold text-white mb-1">
          Drop {mediaLabel} here, or <span className={accentClasses.iconText}>browse system files</span>
        </div>
        <div className="text-xs text-[#94A3B8] max-w-md mb-4">
          Local memory staging only. File bytes are loaded into an ephemeral RAM buffer and never stored permanently.
        </div>

        {/* Supported format badges */}
        <div className="flex flex-wrap items-center justify-center gap-1.5">
          {acceptedFormats.map((fmt) => (
            <span
              key={fmt}
              className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded border ${accentClasses.chipBg}`}
            >
              {fmt}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
