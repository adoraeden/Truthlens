import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readImageCalibrationFields, readImageScoreFields } from '../src/services/imageScore';

test('missing calibrator maps to unavailable without changing raw verdict', () => {
  const raw = {
    positive_class: 'AI-generated', verdict: 'AI-generated', score: 0.8807970779,
    logit: 4, decision_threshold: 2, score_cutoff: 0.5,
    score_method: 'sigmoid(logit - decision_threshold)', score_calibrated_probability: false,
  };
  assert.equal(readImageCalibrationFields(raw).calibration_status, 'unavailable');
  assert.equal(readImageScoreFields(raw).verdict, 'AI-generated');
});

test('calibrated probabilities validate bounds and complementarity', () => {
  const parsed = readImageCalibrationFields({
    calibration_status: 'available', calibration_method: 'platt',
    ai_generated_probability: 0.73, genuine_probability: 0.27,
  });
  assert.equal(parsed.ai_generated_probability, 0.73);
  assert.equal(parsed.genuine_probability, 0.27);
  assert.throws(() => readImageCalibrationFields({
    calibration_status: 'available', calibration_method: 'platt',
    ai_generated_probability: 1.1, genuine_probability: -0.1,
  }));
  assert.throws(() => readImageCalibrationFields({
    calibration_status: 'available', calibration_method: 'platt',
    ai_generated_probability: 0.6, genuine_probability: 0.3,
  }));
});
