"""Image-only diagnostic visualizations derived from decoded source pixels."""

import base64
import io
import math

import numpy as np
from PIL import Image, ImageFilter
from PIL.ExifTags import TAGS


DIAGNOSTIC_LONG_EDGE = 768
ELA_JPEG_QUALITY = 90
NOISE_GAUSSIAN_RADIUS = 1.0
DISPLAY_PERCENTILE = 99.5
ATTRIBUTION_GRID_SIZE = 4


def _png_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _display_scale(values):
    peak = float(np.max(values)) if values.size else 0.0
    if not math.isfinite(peak) or peak <= 0:
        return 0.0
    percentile = float(np.percentile(values, DISPLAY_PERCENTILE))
    return max(percentile, 1e-12)


def _scaled_grayscale(values):
    scale = _display_scale(values)
    if scale == 0:
        pixels = np.zeros(values.shape, dtype=np.uint8)
    else:
        pixels = np.clip(values / scale * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L"), scale


def _scaled_colormap(values, colors):
    """Render a diagnostic intensity map with a display-only colour scale."""
    scale = _display_scale(values)

    if scale == 0:
        black = np.zeros((*values.shape, 3), dtype=np.uint8)
        return Image.fromarray(black, mode="RGB"), scale
    else:
        normalized = np.clip(values / scale, 0.0, 1.0)
        normalized = np.power(normalized, 0.85)

    stops = np.asarray(colors, dtype=np.float32)
    positions = np.linspace(0.0, 1.0, len(stops))

    pixels = np.stack([
        np.interp(normalized, positions, stops[:, channel])
        for channel in range(3)
    ], axis=-1).astype(np.uint8)

    return Image.fromarray(pixels, mode="RGB"), scale


ELA_COLORS = [
    (18, 10, 38),     # Low: dark purple
    (48, 42, 120),    # Blue
    (38, 130, 190),   # Cyan-blue
    (45, 205, 190),   # Cyan
    (255, 235, 110),  # Yellow
    (245, 125, 55),   # Orange
    (190, 35, 65),    # High: red
]

NOISE_COLORS = [
    (8, 18, 48),      # Low: navy
    (25, 75, 150),    # Blue
    (30, 160, 190),   # Cyan
    (65, 190, 130),   # Green
    (230, 220, 70),   # Yellow
    (250, 135, 45),   # Orange
    (210, 40, 45),    # High: red
]


def compute_ela(image, quality=ELA_JPEG_QUALITY):
    """Compare a decoded RGB preview with a documented JPEG round trip.

    ``image`` is expected to be the EXIF-corrected, source-derived preview.
    The returned scale is the 99.5th percentile of mean absolute channel
    difference in 8-bit pixel levels; the displayed map maps that value to
    white. This is a diagnostic view, not an authenticity test.
    """
    source = image.convert("RGB")
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as reencoded:
        reencoded_rgb = reencoded.convert("RGB")
        difference = np.abs(
            np.asarray(source, dtype=np.int16) - np.asarray(reencoded_rgb, dtype=np.int16)
        ).mean(axis=2, dtype=np.float32)
    return _scaled_colormap(difference, ELA_COLORS)


def compute_noise_residual(image, radius=NOISE_GAUSSIAN_RADIUS):
    """Return mean absolute RGB residual from a Gaussian-blurred image."""
    source = image.convert("RGB")
    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
    residual = np.abs(
        np.asarray(source, dtype=np.int16) - np.asarray(blurred, dtype=np.int16)
    ).mean(axis=2, dtype=np.float32)
    return _scaled_colormap(residual, NOISE_COLORS)


def compute_rgb_analysis(image):
    """Compute exact per-channel histograms and statistics from decoded pixels."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    count = rgb.width * rgb.height
    histograms = {}
    statistics = {}
    combined_histogram = rgb.histogram()
    for channel_index, channel_name in enumerate(("red", "green", "blue")):
        histogram = combined_histogram[channel_index * 256:(channel_index + 1) * 256]
        histograms[channel_name] = histogram
        total = sum(histogram)
        mean = sum(value * frequency for value, frequency in enumerate(histogram)) / total
        variance = sum(
            ((value - mean) ** 2) * frequency
            for value, frequency in enumerate(histogram)
        ) / total
        nonzero = [index for index, frequency in enumerate(histogram) if frequency]
        statistics[channel_name] = {
            "minimum": nonzero[0],
            "maximum": nonzero[-1],
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
        }
    return {
        "pixel_count": count,
        "source_size": [rgb.width, rgb.height],
        "histograms": histograms,
        "statistics": statistics,
        "method": "256-bin histograms and population statistics over all EXIF-corrected decoded RGB pixels",
    }


def compute_attribution_heatmap(attribution):
    """Colorize positive attribution and summarize actual values in a 4×4 grid."""
    values = np.asarray(attribution, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Attribution must be a two-dimensional spatial map")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    height, width = values.shape
    # Contrast-enhanced, per-image display normalization.
    # The strongest 2% of positive attribution values saturate at
    # the top of the colour scale, making important regions visible.
    # Raw attribution values and the 4x4 grid calculations remain unchanged.
    positive_values = values[values > 0]

    if positive_values.size:
        display_scale = float(np.percentile(positive_values, 98.0))
        display_scale = max(display_scale, float(np.finfo(np.float32).eps))
        normalized = np.clip(values / display_scale, 0.0, 1.0)
        normalized = np.power(normalized, 0.65)
    else:
        normalized = np.zeros_like(values)

    stops = np.asarray([
        [8, 29, 88],
        [65, 182, 196],
        [255, 255, 191],
        [253, 174, 97],
        [215, 25, 28],
    ], dtype=np.float32)
    positions = np.linspace(0.0, 1.0, len(stops))
    heat_pixels = np.stack([
        np.interp(normalized, positions, stops[:, channel])
        for channel in range(3)
    ], axis=-1).astype(np.uint8)
    heatmap = Image.fromarray(heat_pixels, mode="RGB")

    total = float(values.sum(dtype=np.float64))
    cells = []
    grid = ATTRIBUTION_GRID_SIZE
    for row in range(grid):
        y0, y1 = row * height // grid, (row + 1) * height // grid
        for column in range(grid):
            x0, x1 = column * width // grid, (column + 1) * width // grid
            cell_sum = float(values[y0:y1, x0:x1].sum(dtype=np.float64))
            cells.append({
                "row": row + 1,
                "column": column + 1,
                "x_start": x0,
                "y_start": y0,
                "x_end": x1,
                "y_end": y1,
                "positive_attribution_sum": cell_sum,
                "percentage_of_positive_total": cell_sum / total * 100.0 if total > 0 else 0.0,
            })
    strongest = sorted(
        cells,
        key=lambda cell: cell["positive_attribution_sum"],
        reverse=True,
    )[:3] if total > 0 else []
    top_three_share = sum(cell["percentage_of_positive_total"] for cell in strongest)
    if total <= 0:
        distribution = "no_positive_attribution"
    else:
        # This describes only the measured grid distribution; it is not a
        # confidence or authenticity threshold.
        distribution = "concentrated" if top_three_share >= 50.0 else "distributed"
    return heatmap, {
        "grid_size": grid,
        "positive_attribution_total": total,
        "cells": cells,
        "strongest_regions": strongest,
        "top_three_share_percent": top_three_share,
        "distribution": distribution,
        "distribution_rule": "concentrated when the three highest-sum cells contain at least 50% of positive attribution",
    }


def _source_preview(image):
    image = image if image.mode == "RGB" else image.convert("RGB")
    width, height = image.size
    scale = DIAGNOSTIC_LONG_EDGE / max(width, height)
    if scale < 1:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
    return image


def _jpeg_metadata(metadata):
    image_format = (metadata.get("source_format") or "").upper()
    if image_format not in {"JPEG", "JPG"}:
        return None
    tables = metadata.get("jpeg_quantization_tables") or []
    details = {"quantization_tables": tables}
    for source_key, target_key in (
        ("jfif_version", "jfif_version"),
        ("jfif_unit", "jfif_unit"),
        ("jfif_density", "jfif_density"),
        ("progressive", "progressive"),
    ):
        if source_key in metadata:
            details[target_key] = metadata[source_key]
    return details


def collect_source_metadata(opened_image):
    """Capture only source format fields actually exposed by Pillow."""
    exif = opened_image.getexif()
    def json_value(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (tuple, list)) and all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            return list(value)
        return None

    exif_fields = {}
    for tag_id, value in exif.items():
        safe_value = json_value(value)
        if safe_value is not None:
            exif_fields[TAGS.get(tag_id, f"Tag {tag_id}")] = safe_value
    metadata = {
        "source_format": opened_image.format,
        "source_mode": opened_image.mode,
        "source_size": [opened_image.width, opened_image.height],
        "exif_orientation": exif.get(274),
        "exif_field_count": len(exif),
        "exif_fields": exif_fields,
        "decoder_metadata_fields": sorted(str(key) for key in opened_image.info),
    }
    image_format = (opened_image.format or "").upper()
    if image_format in {"JPEG", "JPG"}:
        quantization = getattr(opened_image, "quantization", None)
        if isinstance(quantization, dict):
            metadata["jpeg_quantization_tables"] = [
                {"table_id": int(table_id), "values": [int(value) for value in values]}
                for table_id, values in sorted(quantization.items())
            ]
        info = opened_image.info
        for key in ("jfif_version", "jfif_unit", "jfif_density", "progressive"):
            if key in info:
                value = info[key]
                if isinstance(value, tuple):
                    value = list(value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    pass
                else:
                    continue
                metadata[key] = value
    return metadata


def build_visualizations(source_image, attribution, source_metadata, heatmap=None, spatial=None):
    """Generate map images and numeric diagnostics without further inference."""
    diagnostic_image = _source_preview(source_image)
    ela_image, ela_scale = compute_ela(diagnostic_image)
    noise_image, noise_scale = compute_noise_residual(diagnostic_image)
    if heatmap is None or spatial is None:
        heatmap, spatial = compute_attribution_heatmap(attribution)
    jpeg = _jpeg_metadata(source_metadata)
    return {
        "heatmap_png_base64": _png_base64(heatmap),
        "spatial_attribution": spatial,
        "diagnostic_preview_size": [diagnostic_image.width, diagnostic_image.height],
        "ela": {
            "map_png_base64": _png_base64(ela_image),
            "method": f"Mean absolute RGB pixel difference after JPEG re-encoding at quality {ELA_JPEG_QUALITY}, 4:2:0 subsampling",
            "display_scale_max_pixel_difference": ela_scale,
            "display_scale_percentile": DISPLAY_PERCENTILE,
            "note": "The decoded source is first resized to a maximum 768-pixel long edge; the display maps the 99.5th-percentile difference to white.",
            "limitations": "ELA responds to any difference from this new JPEG encoding. Prior recompression, editing, resizing, denoising, and ordinary image processing can change the map; it does not establish manipulation.",
        },
        "noise_residual": {
            "map_png_base64": _png_base64(noise_image),
            "method": f"Mean absolute RGB residual from Gaussian blur (radius {NOISE_GAUSSIAN_RADIUS:g})",
            "display_scale_max_pixel_difference": noise_scale,
            "display_scale_percentile": DISPLAY_PERCENTILE,
            "note": "The decoded source is first resized to a maximum 768-pixel long edge; the display maps the 99.5th-percentile residual to white.",
            "limitations": "Denoising, sharpening, camera image processing, resizing, and compression alter high-frequency residuals. The map is diagnostic and is not proof of synthesis or editing.",
        },
        "rgb": compute_rgb_analysis(source_image),
        "compression": {
            "source_format": source_metadata.get("source_format") or "unknown",
            "source_mode": source_metadata.get("source_mode") or "unknown",
            "source_size": list(source_metadata.get("source_size", source_image.size)),
            "oriented_size": [source_image.width, source_image.height],
            "exif_orientation": source_metadata.get("exif_orientation"),
            "exif_field_count": int(source_metadata.get("exif_field_count", 0)),
            "exif_fields": source_metadata.get("exif_fields", {}),
            "decoder_metadata_fields": source_metadata.get("decoder_metadata_fields", []),
            "jpeg": jpeg,
            "quality_estimate": None,
            "note": (
                "JPEG quantization tables and JFIF fields are shown only when present in a JPEG source. "
                "Quantization tables are not converted into an estimated encoder quality."
                if jpeg is not None
                else (
                    "The detected source is not JPEG; JPEG-specific compression fields are not applicable."
                    if source_metadata.get("source_format")
                    else "Source format is unknown; JPEG-specific fields are unavailable and no JPEG compression is inferred."
                )
            ),
        },
    }


def png_base64(image):
    return _png_base64(image)
