export type ImageScoreFields = {
  positive_class?: unknown;
  verdict?: unknown;
  score?: unknown;
  logit?: unknown;
  decision_threshold?: unknown;
  score_cutoff?: unknown;
  score_method?: unknown;
  score_calibrated_probability?: unknown;
  explanation?: unknown;
  calibration_status?: unknown;
  calibration_method?: unknown;
  ai_generated_probability?: unknown;
  genuine_probability?: unknown;
  calibration_note?: unknown;
};

export function readImageCalibrationFields(result: ImageScoreFields) {
  if (result.calibration_status === 'unavailable' || result.calibration_status === undefined) {
    return {
      calibration_status: 'unavailable' as const,
      calibration_method: undefined,
      ai_generated_probability: undefined,
      genuine_probability: undefined,
      calibration_note: typeof result.calibration_note === 'string'
        ? result.calibration_note
        : 'Calibration unavailable.',
    };
  }
  const ai = typeof result.ai_generated_probability === 'number'
    ? result.ai_generated_probability
    : Number.NaN;
  const genuine = typeof result.genuine_probability === 'number'
    ? result.genuine_probability
    : Number.NaN;
  if (
    result.calibration_status !== 'available' ||
    !['platt', 'temperature'].includes(String(result.calibration_method)) ||
    !Number.isFinite(ai) || ai < 0 || ai > 1 ||
    !Number.isFinite(genuine) || genuine < 0 || genuine > 1 ||
    Math.abs(ai + genuine - 1) > 1e-6
  ) {
    throw new Error('The image detector returned invalid calibration data. Please retry the analysis.');
  }
  return {
    calibration_status: 'available' as const,
    calibration_method: result.calibration_method as 'platt' | 'temperature',
    ai_generated_probability: ai,
    genuine_probability: genuine,
    calibration_note: typeof result.calibration_note === 'string' ? result.calibration_note : '',
  };
}

export function readImageScoreFields(result: ImageScoreFields) {
  const score = typeof result.score === 'number' ? result.score : Number.NaN;
  const logit = typeof result.logit === 'number' ? result.logit : Number.NaN;
  const decisionThreshold = typeof result.decision_threshold === 'number'
    ? result.decision_threshold
    : Number.NaN;
  const scoreCutoff = typeof result.score_cutoff === 'number'
    ? result.score_cutoff
    : Number.NaN;
  const expectedVerdict = logit >= decisionThreshold ? 'AI-generated' : 'Likely real';
  const shiftedLogit = logit - decisionThreshold;
  const expectedScore = shiftedLogit >= 0
    ? 1 / (1 + Math.exp(-shiftedLogit))
    : Math.exp(shiftedLogit) / (1 + Math.exp(shiftedLogit));

  if (
    result.positive_class !== 'AI-generated' ||
    !Number.isFinite(score) || score < 0 || score > 1 ||
    !Number.isFinite(logit) || !Number.isFinite(decisionThreshold) ||
    !Number.isFinite(scoreCutoff) || scoreCutoff !== 0.5 ||
    result.verdict !== expectedVerdict ||
    Math.abs(score - expectedScore) > 1e-6 ||
    result.score_method !== 'sigmoid(logit - decision_threshold)' ||
    result.score_calibrated_probability !== false
  ) {
    throw new Error('The image detector returned inconsistent or invalid score data. Please retry the analysis.');
  }

  return {
    positive_class: 'AI-generated',
    verdict: expectedVerdict,
    score,
    logit,
    decision_threshold: decisionThreshold,
    score_cutoff: scoreCutoff,
  } as const;
}
