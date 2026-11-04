import { AttributionDisplayMode, ImageForensicSection } from '../types/forensics';

export interface ImageForensicsWorkspaceState {
  section: ImageForensicSection;
  attributionMode: AttributionDisplayMode;
  overlayOpacity: number;
}

export type ImageForensicsWorkspaceAction =
  | { type: 'select-section'; section: ImageForensicSection }
  | { type: 'select-attribution-mode'; mode: AttributionDisplayMode }
  | { type: 'set-overlay-opacity'; opacity: number };

export const initialImageForensicsWorkspaceState: ImageForensicsWorkspaceState = {
  section: 'attribution',
  attributionMode: 'overlay',
  overlayOpacity: 0.55,
};

export function reduceImageForensicsWorkspaceState(
  state: ImageForensicsWorkspaceState,
  action: ImageForensicsWorkspaceAction,
): ImageForensicsWorkspaceState {
  if (action.type === 'select-section') {
    return { ...state, section: action.section };
  }
  if (action.type === 'select-attribution-mode') {
    return { ...state, attributionMode: action.mode };
  }
  return { ...state, overlayOpacity: Math.min(1, Math.max(0, action.opacity)) };
}
