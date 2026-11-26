import React, { useState } from 'react';
import {
  History,
  Download,
  Trash2,
  FileText,
  Search,
} from 'lucide-react';
import { HistoryItem, MediaType, Verdict } from '../types/forensics';

interface HistoryViewProps {
  history: HistoryItem[];
  onClearHistory: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onClearHistory,
}) => {
  const [filterType, setFilterType] = useState<MediaType | 'all'>('all');
  const [filterVerdict, setFilterVerdict] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);

  const filteredItems = history.filter((item) => {
    const matchesType = filterType === 'all' || item.mediaType === filterType;
    const matchesVerdict =
      filterVerdict === 'all' ||
      (filterVerdict === 'AI-generated' &&
        (item.verdictDisplay === 'AI-generated' || item.verdictDisplay.includes('AI-FLAGGED'))) ||
      (filterVerdict === 'Likely real' &&
        (item.verdictDisplay === 'Likely real' || item.verdictDisplay.includes('Likely real')));
    const matchesSearch =
      searchQuery === '' || item.fileName.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesType && matchesVerdict && matchesSearch;
  });

  const exportReport = () => {
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(history, null, 2)
    )}`;
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', jsonString);
    downloadAnchor.setAttribute('download', `TruthLens_Actual_Session_Audit_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header & Controls */}
      <div className="bg-[#191B30]/80 border border-[#A78BFA]/30 rounded-2xl p-6 backdrop-blur-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-white/10 gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-[#A78BFA]/10 border border-[#A78BFA]/30 flex items-center justify-center text-[#A78BFA]">
                <History className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-bold tracking-tight text-white">
                FORENSIC AUDIT TRAIL
              </h3>
            </div>
            <p className="text-xs md:text-sm text-[#94A3B8] mt-1">
              Actual session audit records of completed detector analyses. No pre-populated or synthetic history entries.
            </p>
          </div>

          <div className="flex items-center gap-2.5 self-start sm:self-auto">
            <button
              onClick={exportReport}
              disabled={history.length === 0}
              className="px-3.5 py-2 rounded-xl bg-[#62E8FF]/10 hover:bg-[#62E8FF]/20 text-[#62E8FF] border border-[#62E8FF]/30 text-xs font-bold font-mono transition-all flex items-center gap-2 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>EXPORT JSON ({history.length})</span>
            </button>

            <button
              onClick={onClearHistory}
              disabled={history.length === 0}
              className="px-3.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-[#94A3B8] hover:text-[#FF75D8] border border-white/10 text-xs font-bold font-mono transition-all flex items-center gap-2 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>CLEAR SESSION</span>
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
          <div className="relative">
            <Search className="w-4 h-4 text-[#64748B] absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search filename..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-[#121424] border border-white/10 rounded-xl pl-10 pr-3 py-2 text-xs text-white placeholder-[#64748B] focus:border-[#62E8FF] focus:outline-none"
            />
          </div>

          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value as any)}
            className="bg-[#121424] border border-white/10 rounded-xl px-3 py-2 text-xs text-[#94A3B8] focus:border-[#A78BFA] focus:outline-none"
          >
            <option value="all">All Modalities (Image, Audio, Video)</option>
            <option value="image">Image Modality</option>
            <option value="audio">Audio Modality</option>
            <option value="video">Video Modality</option>
          </select>

          <select
            value={filterVerdict}
            onChange={(e) => setFilterVerdict(e.target.value)}
            className="bg-[#121424] border border-white/10 rounded-xl px-3 py-2 text-xs text-[#94A3B8] focus:border-[#FF75D8] focus:outline-none"
          >
            <option value="all">All Verdicts</option>
            <option value="AI-generated">Flagged as AI-Generated</option>
            <option value="Likely real">Likely Real Media</option>
          </select>
        </div>

        {/* History Table */}
        {filteredItems.length > 0 ? (
          <div className="space-y-2.5">
            {filteredItems.map((item) => {
              const isAi =
                item.verdictDisplay === 'AI-generated' || item.verdictDisplay.includes('AI-FLAGGED');
              return (
                <div
                  key={item.id}
                  onClick={() => setSelectedItem(item)}
                  className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl bg-[#121424] border border-white/5 hover:border-[#62E8FF]/30 transition-all cursor-pointer gap-3"
                >
                  <div className="flex items-center gap-3.5">
                    <div
                      className={`w-9 h-9 rounded-lg flex items-center justify-center font-mono text-xs font-bold shrink-0 ${
                        item.mediaType === 'image'
                          ? 'bg-[#62E8FF]/10 text-[#62E8FF] border border-[#62E8FF]/30'
                          : item.mediaType === 'audio'
                          ? 'bg-[#A78BFA]/10 text-[#A78BFA] border border-[#A78BFA]/30'
                          : 'bg-[#FF75D8]/10 text-[#FF75D8] border border-[#FF75D8]/30'
                      }`}
                    >
                      {item.mediaType.toUpperCase().slice(0, 3)}
                    </div>

                    <div>
                      <div className="text-sm font-bold text-white truncate max-w-sm md:max-w-md">
                        {item.fileName}
                      </div>
                      <div className="font-mono text-xs text-[#64748B] flex items-center gap-2 mt-0.5">
                        <span>{item.timestamp}</span>
                        <span>&bull;</span>
                        <span>{item.fileSizeFormatted}</span>
                        <span>&bull;</span>
                        <span className="text-[#94A3B8]">{item.details}</span>
                        {item.latencyMs !== undefined && (
                          <>
                            <span>&bull;</span>
                            <span className="text-[#62E8FF]">{(item.latencyMs / 1000).toFixed(2)}s</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between sm:justify-end gap-3 self-end sm:self-auto w-full sm:w-auto">
                    <span
                      className={`font-mono text-xs font-bold px-2.5 py-1 rounded-md uppercase tracking-wider ${
                        isAi
                          ? 'bg-[#FF75D8]/15 text-[#FF75D8] border border-[#FF75D8]/40'
                          : 'bg-[#62E8FF]/15 text-[#62E8FF] border border-[#62E8FF]/40'
                      }`}
                    >
                      {item.verdictDisplay} ({item.scoreDisplay})
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-12 rounded-xl bg-[#121424]/50 border border-white/5">
            <FileText className="w-10 h-10 text-[#64748B] mx-auto mb-3 opacity-60" />
            <div className="text-sm font-bold text-white mb-1">No Session Audit Records Found</div>
            <div className="text-xs text-[#94A3B8] max-w-xs mx-auto">
              Run analyses in the Image, Audio, or Video Labs to record completed evaluations in this session.
            </div>
          </div>
        )}
      </div>

      {/* Audit Detail Modal */}
      {selectedItem && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 z-50 animate-in fade-in">
          <div className="bg-[#191B30] border border-[#62E8FF]/40 rounded-2xl max-w-xl w-full p-6 relative shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-white/10 mb-4">
              <div className="font-bold text-base text-white flex items-center gap-2">
                <span>AUDIT RECORD //</span>
                <span className="font-mono text-[#62E8FF]">{selectedItem.id}</span>
              </div>
              <button
                onClick={() => setSelectedItem(null)}
                className="text-[#94A3B8] hover:text-white p-1 rounded-lg text-lg cursor-pointer"
              >
                &times;
              </button>
            </div>

            <div className="space-y-3 font-mono text-xs text-[#94A3B8]">
              <div className="flex justify-between">
                <span>FILE NAME:</span>
                <span className="text-white font-bold">{selectedItem.fileName}</span>
              </div>
              <div className="flex justify-between">
                <span>MEDIA TYPE:</span>
                <span className="text-[#62E8FF] uppercase">{selectedItem.mediaType}</span>
              </div>
              <div className="flex justify-between">
                <span>TIMESTAMP:</span>
                <span className="text-white">{selectedItem.timestamp}</span>
              </div>
              <div className="flex justify-between">
                <span>VERDICT:</span>
                <span className="text-white font-bold">{selectedItem.verdictDisplay}</span>
              </div>
              <div className="flex justify-between">
                <span>SCORE SUMMARY:</span>
                <span className="text-white font-bold">{selectedItem.scoreDisplay}</span>
              </div>
              <div className="flex justify-between">
                <span>DIAGNOSTICS:</span>
                <span className="text-[#A78BFA]">{selectedItem.details}</span>
              </div>
              {selectedItem.latencyMs !== undefined && (
                <div className="flex justify-between">
                  <span>MEASURED DURATION:</span>
                  <span className="text-[#62E8FF]">{(selectedItem.latencyMs / 1000).toFixed(2)} seconds</span>
                </div>
              )}
            </div>

            <div className="mt-6 pt-4 border-t border-white/10 flex justify-end">
              <button
                onClick={() => setSelectedItem(null)}
                className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 text-white font-bold text-xs font-mono cursor-pointer"
              >
                Close Audit Record
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
