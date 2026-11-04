import React, { useReducer } from 'react';
import { ChevronRight } from 'lucide-react';
import {
  AttributionDisplayMode,
  ImageAnalysisResult,
  ImageForensicSection,
  ImageForensicVisualizations,
  ImageVisualizationMap,
  SpatialAttributionCell,
} from '../types/forensics';
import {
  ImageForensicsWorkspaceAction,
  initialImageForensicsWorkspaceState,
  reduceImageForensicsWorkspaceState,
} from './imageForensicsWorkspaceState';

type Explanation = NonNullable<ImageAnalysisResult['explanation']>;

const sections: Array<{ id: ImageForensicSection; label: string; number: string }> = [
  { id: 'attribution', label: 'Model Attribution', number: '01' },
  { id: 'ela', label: 'Error Level Analysis', number: '02' },
  { id: 'noise', label: 'Noise Residual', number: '03' },
  { id: 'rgb', label: 'RGB Analysis', number: '04' },
  { id: 'compression', label: 'Compression Diagnostics', number: '05' },
];

function asset(base64?: string) {
  return base64 ? `data:image/png;base64,${base64}` : null;
}

function MethodologyDetails({
  children,
  title = 'Technical Methodology & Caveats',
}: {
  children: React.ReactNode;
  title?: string;
}) {
  const [open, setOpen] = React.useState(false);
  return (
    <details
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
      className="group rounded-xl border border-white/10 bg-[#090A14]/70 transition-colors"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between p-3.5 text-xs font-medium text-[#A6B1C5] hover:text-white transition-colors select-none">
        <span className="flex items-center gap-2">
          <ChevronRight
            className={`w-3.5 h-3.5 text-[#A78BFA] transition-transform duration-200 ${
              open ? 'rotate-90' : ''
            }`}
          />
          <span className="font-mono text-[11px] uppercase tracking-wider">{title}</span>
        </span>
        <span className="font-mono text-[10px] text-[#64748B] group-hover:text-[#94A3B8]">
          {open ? 'Hide details' : 'View details'}
        </span>
      </summary>
      <div className="border-t border-white/5 p-3.5 pt-2.5 space-y-2 text-xs leading-relaxed text-[#94A3B8]">
        {children}
      </div>
    </details>
  );
}

function EvidenceFigure({
  title,
  src,
  alt,
  empty,
  children,
}: {
  title: string;
  src?: string | null;
  alt: string;
  empty: string;
  children?: React.ReactNode;
}) {
  const [failed, setFailed] = React.useState(false);
  return (
    <figure className="min-w-0 space-y-2.5 rounded-2xl border border-white/10 bg-[#0E0F1C]/90 p-3.5 md:p-4">
      <figcaption className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#A78BFA]">
        {title}
      </figcaption>
      <div className="flex min-h-[260px] md:min-h-[380px] items-center justify-center overflow-hidden rounded-xl border border-white/5 bg-[#080912] p-2">
        {src && !failed ? (
          <img
            src={src}
            alt={alt}
            className="max-h-[460px] w-full object-contain rounded-lg"
            onError={() => setFailed(true)}
          />
        ) : children || (
          <p className="max-w-xs text-center text-xs text-[#8290A8]">{empty}</p>
        )}
      </div>
    </figure>
  );
}

function SourceComparison({
  originalUrl,
  title,
  diagnosticTitle,
  diagnosticSrc,
  diagnosticAlt,
  diagnosticEmpty,
  diagnosticContent,
}: {
  originalUrl: string | null;
  title: string;
  diagnosticTitle: string;
  diagnosticSrc?: string | null;
  diagnosticAlt: string;
  diagnosticEmpty: string;
  diagnosticContent?: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <EvidenceFigure
        title="Original Source Image"
        src={originalUrl}
        alt="Original uploaded image reference"
        empty="Original image preview is unavailable."
      />
      <EvidenceFigure
        title={diagnosticTitle || title}
        src={diagnosticSrc}
        alt={diagnosticAlt}
        empty={diagnosticEmpty}
      >
        {diagnosticContent}
      </EvidenceFigure>
    </div>
  );
}

function ColorLegend({
  gradient,
  low,
  high,
  label,
}: {
  gradient: string;
  low: string;
  high: string;
  label: string;
}) {
  return (
    <div
      className="rounded-xl border border-white/5 bg-[#090A14]/70 p-3 space-y-1.5"
      aria-label={`${label}: ${low} to ${high}`}
    >
      <div className="flex items-center justify-between text-[11px]">
        <span className="font-medium text-white">{label}</span>
        <span className="font-mono text-[10px] text-[#8290A8]">Scale legend</span>
      </div>
      <div
        className="h-2.5 rounded-full border border-white/10"
        style={{ background: gradient }}
      />
      <div className="flex justify-between font-mono text-[10px] text-[#A6B1C5]">
        <span>{low}</span>
        <span>{high}</span>
      </div>
    </div>
  );
}

function SpatialAttribution({
  cells,
  strongest,
  distribution,
  rule,
}: {
  cells: SpatialAttributionCell[];
  strongest: SpatialAttributionCell[];
  distribution: string;
  rule: string;
}) {
  const maximum = Math.max(0, ...cells.map((cell) => cell.percentage_of_positive_total));
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(220px,0.8fr)_1.2fr]">
      <div className="rounded-2xl border border-white/10 bg-[#0E0F1C]/90 p-4">
        <div className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#A78BFA]">
          4 × 4 Spatial Attribution Share
        </div>
        <div
          className="mt-3 grid grid-cols-4 gap-1.5"
          role="img"
          aria-label="Positive attribution percentage by spatial grid cell"
        >
          {cells.map((cell) => {
            const share = cell.percentage_of_positive_total;
            const intensity = maximum > 0 ? share / maximum : 0;
            return (
              <div
                key={`${cell.row}-${cell.column}`}
                className="flex aspect-square items-center justify-center rounded-md border border-white/10 font-mono text-[10px] text-white"
                style={{ backgroundColor: `rgba(255, 75, 150, ${0.08 + intensity * 0.84})` }}
                title={`Row ${cell.row}, column ${cell.column}: ${share.toFixed(2)}%`}
              >
                {share.toFixed(1)}%
              </div>
            );
          })}
        </div>
        <p className="mt-2.5 text-[10px] text-[#8290A8] leading-relaxed">
          Each tile shows the relative share of total positive gradient in that crop sector.
        </p>
      </div>

      <div className="rounded-2xl border border-white/10 bg-[#0E0F1C]/90 p-4 space-y-3">
        <div className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#A78BFA]">
          Strongest Contributing Sectors
        </div>
        {strongest.length ? (
          strongest.map((cell, index) => (
            <div key={`${cell.row}-${cell.column}`} className="flex items-center gap-3">
              <span className="font-mono text-xs text-[#FF75D8]">0{index + 1}</span>
              <span className="flex-1 text-sm text-white">Row {cell.row}, Column {cell.column}</span>
              <span className="font-mono text-sm text-[#A78BFA]">
                {cell.percentage_of_positive_total.toFixed(2)}%
              </span>
            </div>
          ))
        ) : (
          <p className="text-sm text-[#8290A8]">No positive attribution values produced.</p>
        )}
        <div className="border-t border-white/10 pt-2.5 text-xs">
          <span className="text-white font-medium">
            Pattern: {distribution.replaceAll('_', ' ')}
          </span>
          <p className="mt-1 text-[11px] text-[#8290A8]">{rule}</p>
        </div>
      </div>
    </div>
  );
}

function OverlayImage({
  crop,
  heatmap,
  opacity,
  onOpacityChange,
}: {
  crop: string;
  heatmap: string;
  opacity: number;
  onOpacityChange: (opacity: number) => void;
}) {
  return (
    <div className="w-full space-y-3">
      <div className="relative mx-auto aspect-square max-h-[460px] w-full overflow-hidden rounded-lg">
        <img
          src={crop}
          alt="Exact 384 by 384 model crop"
          className="absolute inset-0 h-full w-full object-contain"
        />
        <img
          src={heatmap}
          alt="Positive input-gradient attribution heatmap"
          className="absolute inset-0 h-full w-full object-contain"
          style={{ opacity }}
        />
      </div>
      <div className="flex items-center gap-3 px-2 text-xs text-[#A6B1C5]">
        <span className="shrink-0 font-mono text-[11px]">Overlay Opacity:</span>
        <input
          aria-label="Heatmap overlay opacity"
          type="range"
          min="0"
          max="100"
          value={Math.round(opacity * 100)}
          onChange={(event) => onOpacityChange(Number(event.target.value) / 100)}
          className="flex-1 accent-[#FF75D8] cursor-pointer"
        />
        <span className="font-mono text-xs font-bold text-[#FF75D8] w-10 text-right">
          {Math.round(opacity * 100)}%
        </span>
      </div>
    </div>
  );
}

function AttributionPanel({
  explanation,
  originalUrl,
  mode,
  opacity,
  dispatch,
}: {
  explanation: Explanation;
  originalUrl: string | null;
  mode: AttributionDisplayMode;
  opacity: number;
  dispatch: React.Dispatch<ImageForensicsWorkspaceAction>;
}) {
  const crop = asset(explanation.crop_png_base64);
  const heatmap = asset(explanation.heatmap_png_base64);
  const mapError = !crop || !heatmap;
  const viewButtons: Array<{ id: AttributionDisplayMode; label: string }> = [
    { id: 'crop', label: 'Model Crop' },
    { id: 'heatmap', label: 'Heatmap Only' },
    { id: 'overlay', label: 'Adjustable Overlay' },
  ];

  let diagnosticSrc: string | null | undefined;
  let diagnosticContent: React.ReactNode = null;
  let diagnosticTitle = 'Model Crop · 384 × 384 Input';
  if (mode === 'crop') {
    diagnosticSrc = crop;
  } else if (mode === 'heatmap') {
    diagnosticSrc = heatmap;
    diagnosticTitle = 'Positive Attribution Heatmap';
  } else {
    diagnosticTitle = 'Attribution Overlay on Model Crop';
    diagnosticContent =
      crop && heatmap ? (
        <OverlayImage
          crop={crop}
          heatmap={heatmap}
          opacity={opacity}
          onOpacityChange={(opacity) => dispatch({ type: 'set-overlay-opacity', opacity })}
        />
      ) : explanation.overlay_png_base64 ? (
        <img
          src={asset(explanation.overlay_png_base64) || ''}
          alt="Existing attribution overlay"
          className="max-h-[460px] w-full object-contain rounded-lg"
        />
      ) : undefined;
  }

  return (
    <div className="space-y-4">
      {/* Short Description */}
      <p className="text-xs text-[#94A3B8] leading-relaxed">
        Visualizes spatial regions within the 384×384 model crop that contributed most strongly toward the classifier verdict.
      </p>

      {/* Mode Controls */}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Attribution display mode">
        {viewButtons.map((button) => (
          <button
            key={button.id}
            type="button"
            data-testid={`attribution-mode-${button.id}`}
            aria-pressed={mode === button.id}
            onClick={() => dispatch({ type: 'select-attribution-mode', mode: button.id })}
            className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
              mode === button.id
                ? 'border-[#FF75D8]/60 bg-[#FF75D8]/15 text-white'
                : 'border-white/10 bg-white/[0.03] text-[#A6B1C5] hover:border-white/25'
            }`}
          >
            {button.label}
          </button>
        ))}
      </div>

      {/* Visualizations */}
      {mode !== 'overlay' ? (
        <SourceComparison
          originalUrl={originalUrl}
          title="Model Attribution"
          diagnosticTitle={diagnosticTitle}
          diagnosticSrc={diagnosticSrc}
          diagnosticAlt={
            mode === 'heatmap'
              ? 'Positive attribution heatmap aligned to model crop'
              : 'Exact resized and center-cropped model input'
          }
          diagnosticEmpty={
            mode === 'heatmap'
              ? 'Heatmap visualization is unavailable.'
              : 'Model crop preview is unavailable.'
          }
        />
      ) : (
        <SourceComparison
          originalUrl={originalUrl}
          title="Attribution Overlay"
          diagnosticTitle={diagnosticTitle}
          diagnosticAlt="Attribution overlay registered to model crop"
          diagnosticEmpty={mapError ? 'Attribution data unavailable.' : 'Attribution overlay unavailable.'}
          diagnosticContent={diagnosticContent}
        />
      )}

      {/* Legend */}
      <ColorLegend
        gradient="linear-gradient(90deg, #081d58 0%, #41b6c4 25%, #ffffbf 50%, #fdae61 75%, #d7191c 100%)"
        low="Zero / Low attribution"
        high="Peak in crop"
        label="Attribution Gradient Intensity"
      />

      {/* Spatial Attribution Grid — collapsed by default */}
      {explanation.spatial_attribution && (
        <details className="group rounded-xl border border-white/10 bg-[#090A14]/70 transition-colors">
          <summary className="flex cursor-pointer list-none items-center justify-between p-3.5 text-xs font-medium text-[#A6B1C5] hover:text-white transition-colors select-none">
            <span className="flex items-center gap-2">
              <ChevronRight
                className="w-3.5 h-3.5 text-[#FF75D8] transition-transform duration-200 group-open:rotate-90"
              />
              <span className="font-mono text-[11px] uppercase tracking-wider">Attribution Details</span>
            </span>
            <span className="font-mono text-[10px] text-[#64748B] group-hover:text-[#94A3B8]">
              4×4 grid · strongest sectors
            </span>
          </summary>
          <div className="border-t border-white/5 p-3.5 pt-2.5">
            <SpatialAttribution
              cells={explanation.spatial_attribution.cells}
              strongest={explanation.spatial_attribution.strongest_regions}
              distribution={explanation.spatial_attribution.distribution}
              rule={explanation.spatial_attribution.distribution_rule}
            />
          </div>
        </details>
      )}

      {/* Expandable Technical Methodology & Caveats */}
      <MethodologyDetails title="Attribution Methodology & Caveats">
        <p>
          <strong className="text-white">{explanation.method}:</strong> Approximates positive input-gradient saliency for the detector output. It indicates model attention, not causal proof or conclusive manipulation evidence.
        </p>
        <p>
          Heatmap values are normalized strictly within this individual crop. Relative intensities should not be compared as absolute strength across different images.
        </p>
        <p>
          Attribution aligns pixel-for-pixel with the 384 × 384 resized center crop evaluated by the classifier. Outer regions beyond this crop do not participate in gradient attribution.
        </p>
      </MethodologyDetails>
    </div>
  );
}

function MapSection({
  title,
  shortDescription,
  originalUrl,
  map,
  previewSize,
  gradient,
  low,
  high,
  legend,
  caveats,
}: {
  title: string;
  shortDescription: string;
  originalUrl: string | null;
  map?: ImageVisualizationMap;
  previewSize?: [number, number];
  gradient: string;
  low: string;
  high: string;
  legend: string;
  caveats: React.ReactNode;
}) {
  return (
    <div className="space-y-4">
      {/* Short Description */}
      <p className="text-xs text-[#94A3B8] leading-relaxed">{shortDescription}</p>

      {/* Prominent Visual Comparison */}
      <SourceComparison
        originalUrl={originalUrl}
        title={title}
        diagnosticTitle={
          previewSize
            ? `${title} Map · ${previewSize[0]} × ${previewSize[1]} preview`
            : `${title} Map`
        }
        diagnosticSrc={asset(map?.map_png_base64)}
        diagnosticAlt={`${title} diagnostic visualization`}
        diagnosticEmpty="Diagnostic map is not available for this analysis."
      />

      {/* Legend */}
      {map && <ColorLegend gradient={gradient} low={low} high={high} label={legend} />}

      {/* Expandable Technical Methodology & Caveats */}
      <MethodologyDetails title={`${title} Methodology & Caveats`}>
        <p>
          <strong className="text-white">Method:</strong> {map?.method || 'Diagnostic calculation unavailable.'}{' '}
          {map?.note}
        </p>
        {map?.limitations && (
          <p>
            <strong className="text-white">Limitations:</strong> {map.limitations}
          </p>
        )}
        {caveats}
      </MethodologyDetails>
    </div>
  );
}

function RGBHistogram({ rgb }: { rgb: NonNullable<ImageForensicVisualizations['rgb']> }) {
  const colors = { red: '#FB7185', green: '#34D399', blue: '#60A5FA' };
  const maximum = Math.max(1, ...Object.values(rgb.histograms).flat());
  const points = (values: number[]) =>
    values
      .map((count, index) => `${(index / 255) * 512},${156 - (count / maximum) * 148}`)
      .join(' ');

  return (
    <div className="w-full space-y-3 p-2 md:p-4">
      <svg
        viewBox="0 0 512 160"
        role="img"
        aria-label="Actual red, green, and blue pixel histograms"
        className="h-auto w-full overflow-visible"
      >
        <line x1="0" y1="156" x2="512" y2="156" stroke="#64748B" strokeWidth="1" />
        <line x1="0" y1="8" x2="0" y2="156" stroke="#64748B" strokeWidth="1" />
        {(['red', 'green', 'blue'] as const).map((channel) => (
          <polyline
            key={channel}
            data-testid={`rgb-histogram-${channel}`}
            fill="none"
            stroke={colors[channel]}
            strokeWidth="1.8"
            points={points(rgb.histograms[channel])}
          />
        ))}
      </svg>
      <div className="flex flex-wrap items-center gap-4 text-xs">
        {(['red', 'green', 'blue'] as const).map((channel) => (
          <span key={channel} className="flex items-center gap-2 capitalize" style={{ color: colors[channel] }}>
            <i className="h-2 w-4 rounded" style={{ backgroundColor: colors[channel] }} />
            {channel}
          </span>
        ))}
        <span className="ml-auto font-mono text-[10px] text-[#8290A8]">
          x: 0–255 value · y: frequency
        </span>
      </div>
      <p className="text-[10px] text-[#8290A8]">
        Computed across all {rgb.pixel_count.toLocaleString()} decoded pixels.
      </p>
    </div>
  );
}

function RGBPanel({
  rgb,
  originalUrl,
}: {
  rgb?: ImageForensicVisualizations['rgb'];
  originalUrl: string | null;
}) {
  return (
    <div className="space-y-4">
      {/* Short Description */}
      <p className="text-xs text-[#94A3B8] leading-relaxed">
        Analyzes color channel distributions and statistical variance across all decoded source pixels.
      </p>

      {/* Prominent Visual Comparison */}
      <SourceComparison
        originalUrl={originalUrl}
        title="RGB Channel Distribution"
        diagnosticTitle="Pixel Value Histograms (Red, Green, Blue)"
        diagnosticAlt="Actual RGB histogram computed from decoded image pixels"
        diagnosticEmpty="RGB histogram data is unavailable."
        diagnosticContent={rgb ? <RGBHistogram rgb={rgb} /> : undefined}
      />

      {/* Channel Statistics */}
      <div className="rounded-2xl border border-white/10 bg-[#0E0F1C]/90 p-4">
        <div className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#A78BFA] mb-3">
          Channel Statistics (0–255 scale)
        </div>
        {rgb ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {(['red', 'green', 'blue'] as const).map((channel) => {
              const stat = rgb.statistics[channel];
              const channelBorders = {
                red: 'border-[#FB7185]/20 bg-[#FB7185]/[0.03]',
                green: 'border-[#34D399]/20 bg-[#34D399]/[0.03]',
                blue: 'border-[#60A5FA]/20 bg-[#60A5FA]/[0.03]',
              };
              const textColors = {
                red: 'text-[#FB7185]',
                green: 'text-[#34D399]',
                blue: 'text-[#60A5FA]',
              };
              return (
                <div key={channel} className={`rounded-xl border p-3 ${channelBorders[channel]}`}>
                  <div className={`text-xs font-bold capitalize ${textColors[channel]}`}>
                    {channel} Channel
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 font-mono text-[11px] text-[#A6B1C5]">
                    <span>Min: <strong className="text-white">{stat.minimum}</strong></span>
                    <span>Max: <strong className="text-white">{stat.maximum}</strong></span>
                    <span>Mean: <strong className="text-white">{stat.mean.toFixed(1)}</strong></span>
                    <span>Std: <strong className="text-white">{stat.standard_deviation.toFixed(1)}</strong></span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-[#8290A8]">Channel statistics are unavailable.</p>
        )}
      </div>

      {/* Expandable Technical Methodology & Caveats */}
      <MethodologyDetails title="RGB Methodology & Caveats">
        <p>
          Channel statistics and histograms reflect raw decoded pixel intensity populations, not machine learning authenticity scores.
        </p>
        <p>
          Camera white balance, lighting conditions, tone grading, color profiles, and format conversion heavily influence channel balance. Similar histograms do not prove identical source, nor do unusual curves prove manipulation.
        </p>
      </MethodologyDetails>
    </div>
  );
}

function QuantizationTables({ tables }: { tables: Array<{ table_id: number; values: number[] }> }) {
  if (!tables.length) {
    return <p className="text-xs text-[#8290A8]">No JPEG quantization table was exposed by the decoder.</p>;
  }

  const allValues = tables.flatMap((table) => table.values);
  const minimum = Math.min(...allValues);
  const maximum = Math.max(...allValues);

  return (
    <div className="space-y-3">
      {tables.map((table) => (
        <div key={table.table_id} className="rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-white">Table {table.table_id}</span>
            <span className="font-mono text-[10px] text-[#8290A8]">64 DCT coefficients</span>
          </div>
          <div className="grid grid-cols-8 gap-1" aria-label={`JPEG quantization table ${table.table_id}`}>
            {table.values.map((value, index) => {
              const intensity = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum);
              return (
                <div
                  key={index}
                  className="flex aspect-square items-center justify-center rounded-sm font-mono text-[9px] text-white"
                  style={{ backgroundColor: `rgba(98, 232, 255, ${0.12 + intensity * 0.78})` }}
                  title={`Coefficient ${index + 1}: ${value}`}
                >
                  {value}
                </div>
              );
            })}
          </div>
        </div>
      ))}
      <div className="flex justify-between font-mono text-[10px] text-[#8290A8]">
        <span>Lowest: {minimum}</span>
        <span>Highest: {maximum}</span>
      </div>
    </div>
  );
}

function CompressionPanel({
  compression,
  originalUrl,
}: {
  compression?: ImageForensicVisualizations['compression'];
  originalUrl: string | null;
}) {
  const jpeg = compression?.jpeg;
  const exifEntries = Object.entries(compression?.exif_fields || {});
  const decoderFields = compression?.decoder_metadata_fields || [];

  const metadata = compression
    ? [
        ['Container Format', compression.source_format || 'Unknown'],
        ['Dimensions', `${compression.source_size[0]} × ${compression.source_size[1]}`],
        ['Color Mode', compression.source_mode || 'Unavailable'],
        ['EXIF Orientation', compression.exif_orientation ?? 'Not specified'],
      ]
    : [];

  return (
    <div className="space-y-4">
      {/* Short Description */}
      <p className="text-xs text-[#94A3B8] leading-relaxed">
        Inspects container format headers, EXIF metadata, and JPEG discrete cosine transform (DCT) quantization tables.
      </p>

      {/* Prominent Visual Comparison */}
      <SourceComparison
        originalUrl={originalUrl}
        title="Decoded Source Metadata"
        diagnosticTitle={
          jpeg
            ? 'JPEG Compression Metadata'
            : 'Container Format & Structural Metadata'
        }
        diagnosticAlt="Decoded image file structural properties"
        diagnosticEmpty="Compression metadata is unavailable."
        diagnosticContent={
          compression ? (
            <div className="w-full space-y-3 p-2 md:p-3">
              <div className="grid grid-cols-2 gap-2">
                {metadata.map(([name, value]) => (
                  <div key={name} className="rounded-lg border border-white/10 bg-white/[0.025] p-2.5">
                    <div className="font-mono text-[10px] uppercase tracking-wider text-[#8290A8]">{name}</div>
                    <div className="mt-1 font-mono text-xs font-semibold text-white">{value}</div>
                  </div>
                ))}
              </div>

              {/* Quantization Tables - Collapsed by default */}
              {jpeg?.quantization_tables.length ? (
                <details className="group rounded-xl border border-[#62E8FF]/20 bg-[#0E0F1C]/90 p-3 transition-colors">
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs font-semibold text-white select-none">
                    <span className="flex items-center gap-2">
                      <ChevronRight className="w-3.5 h-3.5 text-[#62E8FF] transition-transform duration-200 group-open:rotate-90" />
                      <span>JPEG Quantization Tables</span>
                    </span>
                    <span className="font-mono text-[10px] text-[#8290A8]">
                      {jpeg.quantization_tables.length} table{jpeg.quantization_tables.length === 1 ? '' : 's'} · click to view
                    </span>
                  </summary>
                  <div className="mt-3 border-t border-white/10 pt-3 space-y-3">
                    <QuantizationTables tables={jpeg.quantization_tables} />
                    <ColorLegend
                      gradient="linear-gradient(90deg, rgba(98,232,255,.12), rgba(98,232,255,.9))"
                      low="Lower coefficient"
                      high="Higher coefficient"
                      label="Quantization Matrix Value"
                    />
                  </div>
                </details>
              ) : (
                <div className="rounded-lg border border-white/10 bg-white/[0.02] p-2.5 text-xs text-[#94A3B8]">
                  {compression.source_format.toLowerCase() === 'unknown'
                    ? 'JPEG quantization tables unavailable for this format.'
                    : `JPEG quantization tables are not present in ${compression.source_format} format.`}
                </div>
              )}

              {/* Additional Metadata - Collapsed by default */}
              {(exifEntries.length > 0 || decoderFields.length > 0 || jpeg) && (
                <details className="group rounded-xl border border-white/10 bg-[#0E0F1C]/90 p-3 transition-colors">
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs font-semibold text-white select-none">
                    <span className="flex items-center gap-2">
                      <ChevronRight className="w-3.5 h-3.5 text-[#A78BFA] transition-transform duration-200 group-open:rotate-90" />
                      <span>EXIF & Decoder Fields</span>
                    </span>
                    <span className="font-mono text-[10px] text-[#8290A8]">
                      {exifEntries.length} EXIF · {decoderFields.length} decoder keys · click to view
                    </span>
                  </summary>
                  <div className="mt-3 border-t border-white/10 pt-3 space-y-3">
                    {exifEntries.length > 0 && (
                      <div>
                        <div className="font-mono text-[10px] uppercase tracking-wider text-[#A78BFA] mb-2">
                          EXIF Tags
                        </div>
                        <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
                          {exifEntries.map(([name, value]) => (
                            <div key={name} className="flex justify-between gap-3 border-b border-white/5 py-1">
                              <dt className="text-[#8290A8] truncate">{name}</dt>
                              <dd className="break-all text-right text-white font-mono text-[11px]">
                                {Array.isArray(value) ? value.join(', ') : String(value)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    )}
                    {decoderFields.length > 0 && (
                      <div>
                        <div className="font-mono text-[10px] uppercase tracking-wider text-[#A78BFA] mb-1">
                          Decoder Metadata Keys
                        </div>
                        <p className="break-words font-mono text-[11px] text-[#A6B1C5]">
                          {decoderFields.join(', ')}
                        </p>
                      </div>
                    )}
                    {jpeg && (
                      <p className="font-mono text-[11px] text-[#A6B1C5]">
                        {jpeg.progressive !== undefined
                          ? `Progressive JPEG: ${jpeg.progressive ? 'Yes' : 'No'}`
                          : ''}
                        {jpeg.jfif_version ? ` · JFIF ${jpeg.jfif_version.join('.')}` : ''}
                      </p>
                    )}
                  </div>
                </details>
              )}
            </div>
          ) : undefined
        }
      />

      {/* Expandable Technical Methodology & Caveats */}
      <MethodologyDetails title="Compression Methodology & Caveats">
        <p>
          {compression?.note || 'Container headers and quantization matrices are parsed directly from the uploaded file.'}
        </p>
        <p>
          Metadata can easily be stripped, rewritten, or synthesized during export, transit, or re-compression. The presence or absence of EXIF metadata does not prove or disprove artificial generation.
        </p>
      </MethodologyDetails>
    </div>
  );
}

export const ForensicVisualizationWorkspace: React.FC<{
  originalUrl: string | null;
  explanation?: Explanation;
}> = ({ originalUrl, explanation }) => {
  const [state, dispatch] = useReducer(
    reduceImageForensicsWorkspaceState,
    initialImageForensicsWorkspaceState
  );
  const visualizations = explanation?.visualizations;
  const ela = explanation?.ela || visualizations?.ela;
  const noise = explanation?.noise_residual || visualizations?.noise_residual;
  const rgb = explanation?.rgb || visualizations?.rgb;
  const compression = explanation?.compression || visualizations?.compression;
  const previewSize = explanation?.diagnostic_preview_size || visualizations?.diagnostic_preview_size;

  return (
    <section
      className="overflow-hidden rounded-2xl border border-[#A78BFA]/25 bg-gradient-to-br from-[#16182B] via-[#10111F] to-[#0A0B14] shadow-[0_20px_60px_-40px_rgba(167,139,250,0.4)]"
      aria-label="Image forensic visualization workspace"
    >
      {/* Workspace Header */}
      <div className="border-b border-white/10 p-5 md:p-6">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="font-mono text-[10px] tracking-[0.2em] uppercase text-[#A78BFA] font-semibold">
              Forensic Diagnostic Workspace
            </div>
            <h3 className="mt-1 text-lg md:text-xl font-bold text-white">
              Visual Evidence &amp; Artifact Diagnostics
            </h3>
          </div>
          <span className="font-mono text-[10px] text-[#8290A8] self-start md:self-auto">
            ViT-Small/16 · 5 Diagnostic Panels
          </span>
        </div>

        {/* Tab Navigation */}
        <div
          className="mt-4 flex gap-2 overflow-x-auto pb-1"
          role="tablist"
          aria-label="Forensic visualization sections"
        >
          {sections.map((section) => (
            <button
              key={section.id}
              type="button"
              role="tab"
              data-testid={`forensic-tab-${section.id}`}
              aria-selected={state.section === section.id}
              onClick={() => dispatch({ type: 'select-section', section: section.id })}
              className={`flex shrink-0 items-center gap-2 rounded-xl border px-3 py-2 text-left transition md:px-3.5 ${
                state.section === section.id
                  ? 'border-[#A78BFA]/60 bg-[#A78BFA]/15 text-white shadow-[0_0_15px_rgba(167,139,250,0.15)]'
                  : 'border-white/10 bg-white/[0.02] text-[#94A3B8] hover:border-white/20 hover:text-white'
              }`}
            >
              <span className="font-mono text-[10px] text-[#A78BFA] font-bold">{section.number}</span>
              <span className="text-xs font-semibold">{section.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Workspace Tab Panel */}
      <div
        className="p-4 md:p-6"
        role="tabpanel"
        aria-label={sections.find((section) => section.id === state.section)?.label}
      >
        {state.section === 'attribution' && explanation ? (
          <AttributionPanel
            explanation={explanation}
            originalUrl={originalUrl}
            mode={state.attributionMode}
            opacity={state.overlayOpacity}
            dispatch={dispatch}
          />
        ) : state.section === 'attribution' ? (
          <p className="rounded-xl border border-white/10 bg-white/[0.02] p-5 text-sm text-[#94A3B8]">
            Attribution data was not included in this response. The image verdict and score remain available.
          </p>
        ) : null}

        {state.section === 'ela' && (
          <MapSection
            title="Error Level Analysis"
            shortDescription="Highlights compression rate variance across image segments by re-compressing at known JPEG quality and measuring pixel differences."
            originalUrl={originalUrl}
            map={ela}
            previewSize={previewSize}
            gradient="linear-gradient(90deg, #120A26, #302A78, #2682BE, #2DCEBE, #FFEB6E, #F57D37, #BE2341)"
            low="0 difference"
            high={`${(ela?.display_scale_max_pixel_difference || 0).toFixed(2)} 8-bit levels`}
            legend="Mean Absolute RGB Error Delta"
            caveats={
              <p>
                ELA evaluates the delta between the uploaded image and an automated re-compression. Variations can arise naturally from textures, edges, or prior multi-stage saves.
              </p>
            }
          />
        )}

        {state.section === 'noise' && (
          <MapSection
            title="Noise Residual"
            shortDescription="Suppresses low-frequency image content to isolate high-frequency sensor noise and detect splicing or blending artifacts."
            originalUrl={originalUrl}
            map={noise}
            previewSize={previewSize}
            gradient="linear-gradient(90deg, #081230, #194B96, #1EA0BE, #41BE82, #E6DC46, #FA872D, #D2282D)"
            low="Low residual"
            high={`${(noise?.display_scale_max_pixel_difference || 0).toFixed(2)} 8-bit levels`}
            legend="High-Frequency Noise Residual"
            caveats={
              <p>
                The display uses per-image 99.5th-percentile scaling. Residual brightness is normalized within this specific image and does not represent an absolute cross-image scale.
              </p>
            }
          />
        )}

        {state.section === 'rgb' && <RGBPanel rgb={rgb} originalUrl={originalUrl} />}

        {state.section === 'compression' && (
          <CompressionPanel compression={compression} originalUrl={originalUrl} />
        )}
      </div>
    </section>
  );
};
