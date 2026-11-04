export type MediaType = 'image' | 'audio' | 'video';

export type Verdict = 'AI-generated' | 'Likely real' | 'Mixed evidence';

export type ImageForensicSection = 'attribution' | 'ela' | 'noise' | 'rgb' | 'compression';
export type AttributionDisplayMode = 'crop' | 'heatmap' | 'overlay';

export interface SpatialAttributionCell {
  row: number;
  column: number;
  x_start: number;
  y_start: number;
  x_end: number;
  y_end: number;
  positive_attribution_sum: number;
  percentage_of_positive_total: number;
}

export interface ImageVisualizationMap {
  map_png_base64: string;
  method: string;
  display_scale_max_pixel_difference: number;
  display_scale_percentile: number;
  note: string;
  limitations: string;
}

export interface ImageForensicVisualizations {
  diagnostic_preview_size: [number, number];
  ela: ImageVisualizationMap;
  noise_residual: ImageVisualizationMap;
  rgb: {
    pixel_count: number;
    source_size: [number, number];
    histograms: Record<'red' | 'green' | 'blue', number[]>;
    statistics: Record<'red' | 'green' | 'blue', {
      minimum: number;
      maximum: number;
      mean: number;
      standard_deviation: number;
    }>;
    method: string;
  };
  compression: {
    source_format: string;
    source_mode: string;
    source_size: [number, number];
    oriented_size: [number, number];
    exif_orientation: number | null;
    exif_field_count: number;
    exif_fields: Record<string, string | number | boolean | Array<string | number | boolean>>;
    decoder_metadata_fields: string[];
    jpeg: null | {
      quantization_tables: Array<{ table_id: number; values: number[] }>;
      jfif_version?: [number, number];
      jfif_unit?: number;
      jfif_density?: [number, number];
      progressive?: boolean;
    };
    quality_estimate: null;
    note: string;
  };
}

export interface ImageAnalysisResult {
  positive_class: 'AI-generated';
  verdict: Verdict;
  score: number;
  logit: number;
  decision_threshold: number;
  score_cutoff: number;
  score_method: string;
  score_calibrated_probability: false;
  calibration_status: 'available' | 'unavailable';
  calibration_method?: 'platt' | 'temperature';
  ai_generated_probability?: number;
  genuine_probability?: number;
  calibration_note?: string;
  image_size: [number, number];
  model: string;
  analysis_note?: string;
  explanation?: {
    method: string;
    scope: string;
    note: string;
    overlay_png_base64: string;
    crop_png_base64: string;
    heatmap_png_base64?: string;
    spatial_attribution?: {
      grid_size: number;
      positive_attribution_total: number;
      cells: SpatialAttributionCell[];
      strongest_regions: SpatialAttributionCell[];
      top_three_share_percent: number;
      distribution: 'concentrated' | 'distributed' | 'no_positive_attribution';
      distribution_rule: string;
    };
    diagnostic_preview_size?: [number, number];
    ela?: ImageVisualizationMap;
    noise_residual?: ImageVisualizationMap;
    rgb?: ImageForensicVisualizations['rgb'];
    compression?: ImageForensicVisualizations['compression'];
    visualizations?: ImageForensicVisualizations;
  };
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
  decision_threshold: number;
  score_cutoff: number;
  timestamp_seconds: number;
  explanation?: ImageAnalysisResult['explanation'];
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
  audio_status?: 'analyzed' | 'no_audio_stream' | 'audio_stream_unreadable' | 'audio_detector_unavailable';
  audio_status_note?: string;
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
  models?: Record<string, boolean>;
  audioUnavailableReason?: string;
  latencyMs?: number;
}

export type TabType = 'dashboard' | 'image' | 'audio' | 'video' | 'history' | 'methodology';
