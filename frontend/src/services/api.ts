
import {
  ImageAnalysisResult,
  AudioAnalysisResult,
  VideoAnalysisResult,
  BackendStatus,
} from '../types/forensics';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

async function getErrorMessage(response: Response): Promise<string> {
  const errorData = await response.json().catch(() => ({}));
  const detail = errorData.detail || errorData.error;

  return (
    (typeof detail === 'string' ? detail : undefined) ||
    `Detector service error (HTTP ${response.status}).`
  );
}

export async function checkBackendHealth(): Promise<BackendStatus> {
  const startTime = performance.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3000);

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error('Health check failed');
    }

    const data = await response.json();

    return {
      connected: data.status === 'online',
      endpoint: API_BASE_URL,
      message:
        data.status === 'online'
          ? 'Python Detector Microservice Online'
          : 'Python Detector Microservice Offline',
      models: data.detectors,
      latencyMs: Math.round(performance.now() - startTime),
    };
  } catch {
    return {
      connected: false,
      endpoint: API_BASE_URL,
      message:
        'Python Microservice Offline. Start the API server on port 8000.',
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function analyzeImage(
  file: File
): Promise<ImageAnalysisResult> {
  const startTime = performance.now();
  const formData = new FormData();

  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/detect/image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const data = await response.json();
  const result = data.result;

  return {
    verdict: result.verdict,
    score: Number(result.score),
    logit:
      result.logit !== undefined ? Number(result.logit) : undefined,
    image_size: result.image_size,
    model: result.model || 'ViT-Small/16',
    analysis_note: result.analysis_note,
    preprocessing: result.preprocessing,
    filename: data.filename || file.name,
    inferenceDurationMs: Math.round(performance.now() - startTime),
  };
}

export async function analyzeAudio(
  file: File
): Promise<AudioAnalysisResult> {
  const startTime = performance.now();
  const formData = new FormData();

  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/detect/audio`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const data = await response.json();
  const result = data.result;

  return {
    verdict: result.verdict,
    score: Number(result.score),
    duration_analyzed_seconds: Number(
      result.duration_analyzed_seconds || 4
    ),
    source_duration_seconds:
      result.source_duration_seconds !== undefined
        ? Number(result.source_duration_seconds)
        : undefined,
    model_input_seconds:
      result.model_input_seconds !== undefined
        ? Number(result.model_input_seconds)
        : undefined,
    analysis_region: result.analysis_region,
    analysis_start_seconds:
      result.analysis_start_seconds !== undefined
        ? Number(result.analysis_start_seconds)
        : undefined,
    note:
      result.note ||
      'Audio analysis uses the detector model output.',
    model: result.model || 'PellaV2',
    filename: data.filename || file.name,
    inferenceDurationMs: Math.round(performance.now() - startTime),
  };
}

export async function analyzeVideo(
  file: File
): Promise<VideoAnalysisResult> {
  const startTime = performance.now();
  const formData = new FormData();

  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/detect/video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const data = await response.json();
  const result = data.result;

  return {
    duration_seconds: Number(result.duration_seconds),
    frames_analyzed: Number(result.frames_analyzed),
    frames_requested:
      result.frames_requested !== undefined
        ? Number(result.frames_requested)
        : undefined,
    frame_timestamps_seconds: result.frame_timestamps_seconds,
    frame_sampling_method: result.frame_sampling_method,
    ai_flagged_frames: Number(result.ai_flagged_frames),
    average_frame_score: Number(result.average_frame_score),
    frame_score_note: result.frame_score_note,
    frame_results: result.frame_results || [],
    audio_analysis: result.audio_analysis || null,
    audio_track_present: result.audio_track_present,
    audio_status: result.audio_status,
    interpretation:
      result.interpretation ||
      'Video frames and audio are evaluated separately.',
    filename: data.filename || file.name,
    inferenceDurationMs: Math.round(performance.now() - startTime),
  };
}
