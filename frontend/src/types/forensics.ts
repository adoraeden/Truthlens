export type MediaType = 'image' | 'audio' | 'video';

export type Verdict = 'AI-generated' | 'Likely real';

export interface ImageAnalysisResult {
  verdict: Verdict;
  score: number;
  logit?: number;
  image_size: [number, number];
  model: string;
  analysis_note?: string;
  preprocessing?: {
    resize_short_edge: number;
    center_crop: [number, number];
    color_mode: string;
  };
  filename?: string;
  inferenceDurationMs?: number;
}

export interface AudioAnalysisResult {
  verdict: Verdict;
  score: number;
  duration_analyzed_seconds: number;
  source_duration_seconds?: number;
  model_input_seconds?: number;
  analysis_region?: 'center_crop' | 'full_recording';
  analysis_start_seconds?: number;
  note: string;
  model: string;
  filename?: string;
  inferenceDurationMs?: number;
}

export interface FrameAnalysisResult {
  verdict: Verdict;
  score: number;
  logit: number;
  timestamp_seconds: number;
}

export interface VideoAnalysisResult {
  duration_seconds: number;
  frames_requested?: number;
  frames_analyzed: number;
  frame_timestamps_seconds?: number[];
  frame_sampling_method?: string;
  ai_flagged_frames: number;
  average_frame_score: number;
  frame_score_note?: string;
  frame_results: FrameAnalysisResult[];
  audio_analysis: AudioAnalysisResult | null;
  audio_track_present?: boolean;
  audio_status?: 'analyzed' | 'no_audio_stream' | 'audio_stream_unreadable';
  interpretation: string;
  filename?: string;
  inferenceDurationMs?: number;
}

export type AnalysisResult =
  | { type: 'image'; data: ImageAnalysisResult }
  | { type: 'audio'; data: AudioAnalysisResult }
  | { type: 'video'; data: VideoAnalysisResult };

export interface HistoryItem {
  id: string;
  timestamp: string;
  fileName: string;
  mediaType: MediaType;
  fileSizeFormatted: string;
  scoreDisplay: string;
  verdictDisplay: string;
  details: string;
  previewUrl?: string;
  latencyMs?: number;
  fullResult: AnalysisResult;
}

export interface BackendStatus {
  connected: boolean;
  endpoint: string;
  message: string;
  models?: Record<string, string>;
  latencyMs?: number;
}

export type TabType = 'dashboard' | 'image' | 'audio' | 'video' | 'history' | 'methodology';
