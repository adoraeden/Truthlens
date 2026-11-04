import assert from 'node:assert/strict';
import test from 'node:test';
import {
  initialImageForensicsWorkspaceState,
  reduceImageForensicsWorkspaceState,
} from '../src/components/imageForensicsWorkspaceState';

test('forensic workspace switches sections and attribution views', () => {
  const elaState = reduceImageForensicsWorkspaceState(initialImageForensicsWorkspaceState, {
    type: 'select-section',
    section: 'ela',
  });
  assert.equal(elaState.section, 'ela');

  const heatmapState = reduceImageForensicsWorkspaceState(elaState, {
    type: 'select-section',
    section: 'attribution',
  });
  const overlayState = reduceImageForensicsWorkspaceState(heatmapState, {
    type: 'select-attribution-mode',
    mode: 'heatmap',
  });
  assert.equal(overlayState.section, 'attribution');
  assert.equal(overlayState.attributionMode, 'heatmap');
});

test('overlay opacity stays in range and can be adjusted', () => {
  const adjusted = reduceImageForensicsWorkspaceState(initialImageForensicsWorkspaceState, {
    type: 'set-overlay-opacity',
    opacity: 0.27,
  });
  assert.equal(adjusted.overlayOpacity, 0.27);
  assert.equal(reduceImageForensicsWorkspaceState(adjusted, {
    type: 'set-overlay-opacity', opacity: 2,
  }).overlayOpacity, 1);
  assert.equal(reduceImageForensicsWorkspaceState(adjusted, {
    type: 'set-overlay-opacity', opacity: -1,
  }).overlayOpacity, 0);
});
