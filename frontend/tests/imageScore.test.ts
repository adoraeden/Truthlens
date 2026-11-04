import assert from 'node:assert/strict';
import test from 'node:test';
import { readImageScoreFields } from '../src/services/imageScore';

const result = (overrides: Record<string, unknown> = {}) => ({
  positive_class: 'AI-generated',
  verdict: 'Likely real',
  score: 1 / (1 + Math.exp(-(0.386 - 2.0))),
  logit: 0.386,
  decision_threshold: 2.0,
  score_cutoff: 0.5,
  score_method: 'sigmoid(logit - decision_threshold)',
  score_calibrated_probability: false,
  ...overrides,
});

test('accepts consistent low score, logit cutoff and likely-real verdict', () => {
  const parsed = readImageScoreFields(result());
  assert.equal(parsed.verdict, 'Likely real');
  assert.equal(parsed.decision_threshold, 2.0);
  assert.equal(parsed.score_cutoff, 0.5);
  assert.ok(Math.abs(parsed.score - 0.1661) < 0.001);
});

test('accepts equality at the logit cutoff as the AI-positive boundary', () => {
  const parsed = readImageScoreFields(result({
    verdict: 'AI-generated',
    score: 0.5,
    logit: 2.0,
  }));
  assert.equal(parsed.verdict, 'AI-generated');
});

test('accepts the zero-logit default boundary and its threshold-relative score', () => {
  const parsed = readImageScoreFields(result({
    verdict: 'AI-generated',
    score: 0.5,
    logit: 0,
    decision_threshold: 0,
  }));
  assert.equal(parsed.decision_threshold, 0);
  assert.equal(parsed.verdict, 'AI-generated');
  assert.equal(parsed.score, 0.5);
});

test('rejects an inconsistent verdict, score, non-finite or out-of-range values', () => {
  for (const invalid of [
    result({ verdict: 'AI-generated' }),
    result({ positive_class: 'real' }),
    result({ score: 0.9 }),
    result({ logit: Number.NaN }),
    result({ decision_threshold: Number.POSITIVE_INFINITY }),
    result({ score: 1.01 }),
    result({ score_calibrated_probability: true }),
  ]) {
    assert.throws(() => readImageScoreFields(invalid));
  }
});
