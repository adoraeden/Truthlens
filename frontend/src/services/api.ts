
import {
  ImageAnalysisResult,
  AudioAnalysisResult,
  VideoAnalysisResult,
  BackendStatus,
} from '../types/forensics';
import { ImageScoreFields, readImageCalibrationFields, readImageScoreFields } from './imageScore';

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
      audioUnavailableReason: data.audio_unavailable_reason,
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
  const scoreFields = readImageScoreFields(result);
  const calibrationFields = readImageCalibrationFields(result);

  return {
    ...scoreFields,
    ...calibrationFields,
    score_method: result.score_method,
    score_calibrated_probability: false,
    image_size: result.image_size,
    model: result.model || 'ViT-Small/16',
    analysis_note: result.analysis_note,
    explanation: result.explanation,
    preprocessing: result.preprocessing,
    filename: data.filename || file.name,
    inferenceDurationMs: Math.round(performance.now() - startTime),
  };
}

/**
 * Ask the backend to convert an audio file to a browser-compatible MP3 and
 * return an object URL that can be set directly on an <audio> element.
 *
 * The original file is NEVER sent to this endpoint for forensic analysis.
 * Always call analyzeAudio() with the original file for PellaV2 inference.
 *
 * The caller is responsible for calling URL.revokeObjectURL() on the returned
 * URL when it is no longer needed.
 *
 * Throws if the backend is unreachable or the conversion fails.
 */
export async function fetchAudioPreview(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/convert/audio-preview`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

/**
 * Ask the backend to convert a video file to a browser-compatible H.264 MP4
 * (yuv420p + faststart) and return an object URL for playback.
 *
 * The original file is NEVER sent to this endpoint for forensic analysis.
 * Always call analyzeVideo() with the original file for full forensic inference.
 */
export async function fetchVideoPreview(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/convert/video-preview`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);
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
  const frameResults = (result.frame_results || []).map((frame: ImageScoreFields & { timestamp_seconds?: unknown }) => ({
    ...readImageScoreFields(frame),
    timestamp_seconds: Number(frame.timestamp_seconds),
    explanation: frame.explanation,
  }));
  if (frameResults.some((frame: { timestamp_seconds: number }) => !Number.isFinite(frame.timestamp_seconds))) {
    throw new Error('The video detector returned an invalid frame timestamp. Please retry the analysis.');
  }

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
    frame_results: frameResults,
    audio_analysis: result.audio_analysis || null,
    audio_track_present: result.audio_track_present,
    audio_status: result.audio_status,
    audio_status_note: result.audio_status_note,
    interpretation:
      result.interpretation ||
      'Video frames and audio are evaluated separately.',
    filename: data.filename || file.name,
    inferenceDurationMs: Math.round(performance.now() - startTime),
  };
}
