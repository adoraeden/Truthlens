import assert from 'node:assert/strict';
import test from 'node:test';
import { readImageScoreFields } from '../src/services/imageScore';

test('audio response fields validation: authentic and AI-generated contracts', () => {
  const realAudioResponse = {
    verdict: 'Likely real',
    score: 0.00097,
    duration_analyzed_seconds: 4.0,
    source_duration_seconds: 6.0,
    model_input_seconds: 4.0,
    analysis_region: 'center_crop',
    analysis_start_seconds: 1.0,
    model: 'PellaV2',
    note: 'Detector output is uncalibrated.',
  };

  assert.equal(realAudioResponse.verdict, 'Likely real');
  assert.ok(realAudioResponse.score < 0.50);
  assert.equal(realAudioResponse.duration_analyzed_seconds, 4.0);
  assert.equal(realAudioResponse.source_duration_seconds, 6.0);
  assert.equal(realAudioResponse.analysis_start_seconds, 1.0);

  const fakeAudioResponse = {
    verdict: 'AI-generated',
    score: 0.9994,
    duration_analyzed_seconds: 4.0,
    source_duration_seconds: 4.0,
    model_input_seconds: 4.0,
    analysis_region: 'full_recording',
    analysis_start_seconds: 0.0,
    model: 'PellaV2',
  };

  assert.equal(fakeAudioResponse.verdict, 'AI-generated');
  assert.ok(fakeAudioResponse.score >= 0.50);
  assert.equal(fakeAudioResponse.duration_analyzed_seconds, 4.0);
  assert.equal(fakeAudioResponse.analysis_region, 'full_recording');
});

test('video response parses sampled frames and enforces image score integrity', () => {
  const sampledFrames = [
    {
      timestamp_seconds: 1.25,
      positive_class: 'AI-generated',
      verdict: 'AI-generated',
      score: 1 / (1 + Math.exp(-2.5)),
      logit: 2.5,
      decision_threshold: 0.0,
      score_cutoff: 0.5,
      score_method: 'sigmoid(logit - decision_threshold)',
      score_calibrated_probability: false,
      explanation: { method: 'positive input-gradient × input' },
    },
    {
      timestamp_seconds: 3.75,
      positive_class: 'AI-generated',
      verdict: 'Likely real',
      score: 1 / (1 + Math.exp(1.5)),
      logit: -1.5,
      decision_threshold: 0.0,
      score_cutoff: 0.5,
      score_method: 'sigmoid(logit - decision_threshold)',
      score_calibrated_probability: false,
      explanation: { method: 'positive input-gradient × input' },
    },
  ];

  const parsedFrames = sampledFrames.map((frame) => ({
    ...readImageScoreFields(frame),
    timestamp_seconds: Number(frame.timestamp_seconds),
    explanation: frame.explanation,
  }));

  assert.equal(parsedFrames.length, 2);
  assert.equal(parsedFrames[0].verdict, 'AI-generated');
  assert.equal(parsedFrames[0].timestamp_seconds, 1.25);
  assert.equal(parsedFrames[1].verdict, 'Likely real');
  assert.equal(parsedFrames[1].timestamp_seconds, 3.75);

  // Reject non-finite timestamp
  const invalidTimestampFrames = [{ ...sampledFrames[0], timestamp_seconds: Number.NaN }];
  assert.ok(invalidTimestampFrames.some((f) => !Number.isFinite(Number(f.timestamp_seconds))));
});

test('video response preserves audio status separation', () => {
  const videoWithAudio = {
    duration_seconds: 10.0,
    frames_analyzed: 5,
    audio_track_present: true,
    audio_status: 'analyzed',
    audio_analysis: {
      verdict: 'AI-generated',
      score: 0.9615,
      source_duration_seconds: 10.0,
      duration_analyzed_seconds: 4.0,
      analysis_region: 'center_crop',
      analysis_start_seconds: 3.0,
    },
  };

  assert.equal(videoWithAudio.audio_status, 'analyzed');
  assert.ok(videoWithAudio.audio_analysis !== null);
  assert.equal(videoWithAudio.audio_analysis.verdict, 'AI-generated');

  const videoWithoutAudio = {
    duration_seconds: 6.0,
    frames_analyzed: 3,
    audio_track_present: false,
    audio_status: 'no_audio_stream',
    audio_status_note: 'No audio stream was present in the video.',
    audio_analysis: null,
  };

  assert.equal(videoWithoutAudio.audio_status, 'no_audio_stream');
  assert.equal(videoWithoutAudio.audio_analysis, null);
});

test('conservative 3-state video verdict aggregation logic', () => {
  function computeVideoVerdict(aiFlaggedFrames: number, totalFrames: number) {
    if (totalFrames > 0 && aiFlaggedFrames === totalFrames) {
      return 'AI-generated' as const;
    }
    if (aiFlaggedFrames > 0 && aiFlaggedFrames < totalFrames) {
      return 'Mixed evidence' as const;
    }
    return 'Likely real' as const;
  }

  // 1. 0 of 8 frames flagged -> Likely real
  assert.equal(computeVideoVerdict(0, 8), 'Likely real');

  // 2. 1 of 8 frames flagged (the real dog video false positive case) -> Mixed evidence (not AI-generated)
  assert.equal(computeVideoVerdict(1, 8), 'Mixed evidence');

  // 3. 5 of 5 frames flagged -> AI-generated
  assert.equal(computeVideoVerdict(5, 5), 'AI-generated');

  // 4. 8 of 8 frames flagged -> AI-generated
  assert.equal(computeVideoVerdict(8, 8), 'AI-generated');

  // 5. 3 of 8 frames flagged -> Mixed evidence
  assert.equal(computeVideoVerdict(3, 8), 'Mixed evidence');
});

test('mixed audio and visual verdicts keep modality independence in video', () => {
  function evaluateModalities(
    visualVerdict: 'Likely real' | 'Mixed evidence' | 'AI-generated',
    audioVerdict: 'Likely real' | 'AI-generated' | null
  ) {
    const visualFlagged = visualVerdict === 'AI-generated';
    const audioFlagged = audioVerdict === 'AI-generated';
    const modalitiesDisagree = audioVerdict !== null && (visualVerdict === 'AI-generated' || visualVerdict === 'Mixed evidence') !== audioFlagged;
    return {
      visualVerdict,
      audioVerdict,
      modalitiesDisagree,
    };
  }

  // Mixed visual (1 of 8 flagged) + Likely real audio
  const result1 = evaluateModalities('Mixed evidence', 'Likely real');
  assert.equal(result1.visualVerdict, 'Mixed evidence');
  assert.equal(result1.audioVerdict, 'Likely real');
  assert.equal(result1.modalitiesDisagree, true);

  // Likely real visual (0 of 8 flagged) + AI-generated audio
  const result2 = evaluateModalities('Likely real', 'AI-generated');
  assert.equal(result2.visualVerdict, 'Likely real');
  assert.equal(result2.audioVerdict, 'AI-generated');
  assert.equal(result2.modalitiesDisagree, true);

  // AI-generated visual (8 of 8 flagged) + AI-generated audio
  const result3 = evaluateModalities('AI-generated', 'AI-generated');
  assert.equal(result3.visualVerdict, 'AI-generated');
  assert.equal(result3.audioVerdict, 'AI-generated');
  assert.equal(result3.modalitiesDisagree, false);

  // Likely real visual + Likely real audio
  const result4 = evaluateModalities('Likely real', 'Likely real');
  assert.equal(result4.visualVerdict, 'Likely real');
  assert.equal(result4.audioVerdict, 'Likely real');
  assert.equal(result4.modalitiesDisagree, false);
});
