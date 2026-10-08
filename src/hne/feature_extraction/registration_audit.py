"""src/hne/feature_extraction/registration_audit.py

Audit of what feature extraction read for each tile. It reproduces the geometry of the
extraction that produced the current features (commit c1636ae: scale fallback to 1.0 and
crops clamped to the slide edge), so the audit shows what those features were computed
from, not what the hardened extractor would do now.

Works on any slide object with the OpenSlide interface (`dimensions`, `read_region`).
"""
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image, ImageDraw

from hne.feature_extraction.patching import PATCH_SPECS, PatchSpec, _has_enough_tissue, _tile_positions

LEGACY_FULLRES_MIN_WIDTH = 10000
LEGACY_MIN_TISSUE_FRACTION = 0.3
PANEL_WIDTH = 750          # each overlay panel; two panels side by side
OUTSIDE_COLOR = (225, 225, 225)
_CMAP = colormaps["viridis"]


def legacy_scale(scale_json: Optional[dict], slide_width: int, fullres_pixel_size: float, tile_size_px: int) -> dict:
    """Scale logic of the extraction at c1636ae, with its silent fallbacks made visible."""
    fallback_hit = False
    if scale_json is None:
        scale_factor, source, fallback_hit = 1.0, "fallback 1.0 (scalefactors_json.json unreadable)", True
    elif "regist_target_img_scalef" in scale_json:
        scale_factor, source = float(scale_json["regist_target_img_scalef"]), "regist_target_img_scalef"
    else:
        scale_factor, source, fallback_hit = 1.0, "fallback 1.0 (no regist_target_img_scalef)", True

    if slide_width >= LEGACY_FULLRES_MIN_WIDTH:
        current_scale, source, fallback_hit = 1.0, "none (slide treated as full resolution)", False
    else:
        current_scale = scale_factor

    return {
        "scale_used": current_scale,
        "scale_source": source,
        "fallback_hit": fallback_hit,
        "effective_um_per_px": fullres_pixel_size / current_scale,
        "effective_tile_size_px": int(round(tile_size_px * current_scale)),
    }


def legacy_patch_windows(x0: int, y0: int, tile_size_px: int, slide_w: int, slide_h: int,
                         effective_um_per_px: float, spec: PatchSpec) -> pd.DataFrame:
    """Every patch window of one tile: where it was meant to be and where the clamp moved it."""
    patch_px = max(1, round(spec.patch_fov_um / effective_um_per_px))
    stride_px = max(1, round(spec.stride_um / effective_um_per_px))
    rows = []
    for py in _tile_positions(y0, tile_size_px, patch_px, stride_px):
        for px in _tile_positions(x0, tile_size_px, patch_px, stride_px):
            read_x = max(0, min(px, slide_w - patch_px))
            read_y = max(0, min(py, slide_h - patch_px))
            rows.append({"px": px, "py": py, "read_x": read_x, "read_y": read_y,
                         "clamped": (read_x != px) or (read_y != py)})
    windows = pd.DataFrame(rows)
    windows.attrs["patch_px"] = patch_px
    return windows


def tile_origin(row: pd.Series, scale_used: float) -> tuple[int, int]:
    return (int(round(float(row["x_min_fullres"]) * scale_used)),
            int(round(float(row["y_min_fullres"]) * scale_used)))


def _color(value: float) -> tuple[int, int, int]:
    r, g, b, _ = _CMAP(float(np.clip(value, 0.0, 1.0)))
    return int(r * 255), int(g * 255), int(b * 255)


def _spot_colors(spots: pd.DataFrame, tiles_df: pd.DataFrame, tile_size_px_fullres: float) -> list:
    """Spot colour: its own tumor fraction if given, else the purity of its tumor tile (grey if none)."""
    if "tumor_fraction" in spots.columns:
        return [_color(v) if pd.notna(v) else (150, 150, 150) for v in spots["tumor_fraction"]]
    purity = {(int(r.tile_row), int(r.tile_col)): float(r.tile_purity) for r in tiles_df.itertuples()}
    rows = (spots["pxl_row_in_fullres"] // tile_size_px_fullres).astype(int)
    cols = (spots["pxl_col_in_fullres"] // tile_size_px_fullres).astype(int)
    return [_color(purity[key]) if key in purity else (150, 150, 150) for key in zip(rows, cols)]


def _draw_panel(image: Optional[Image.Image], image_size: tuple[int, int], spots_xy: np.ndarray, colors: list,
                tile_boxes: list, spot_radius_px: float, title: str) -> Image.Image:
    """
    One overlay panel in the pixel frame of `image`. The canvas grows to include spots and
    tiles that fall outside the image, so coordinates beyond the slide stay visible.
    """
    img_w, img_h = image_size
    xs = np.concatenate([[0, img_w], spots_xy[:, 0], [b for box in tile_boxes for b in (box[0], box[2])]])
    ys = np.concatenate([[0, img_h], spots_xy[:, 1], [b for box in tile_boxes for b in (box[1], box[3])]])
    min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
    zoom = PANEL_WIDTH / (max_x - min_x)
    canvas = Image.new("RGB", (PANEL_WIDTH, max(1, int(round((max_y - min_y) * zoom)))), OUTSIDE_COLOR)

    def to_canvas(x, y):
        return (x - min_x) * zoom, (y - min_y) * zoom

    if image is not None:
        small = image.convert("RGB").resize((max(1, int(round(img_w * zoom))), max(1, int(round(img_h * zoom)))), Image.BILINEAR)
        canvas.paste(small, tuple(int(round(v)) for v in to_canvas(0, 0)))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([*to_canvas(0, 0), *to_canvas(img_w, img_h)], outline=(220, 0, 0), width=2)   # image edge
    r = max(1.0, spot_radius_px * zoom)
    for (x, y), color in zip(spots_xy, colors):
        cx, cy = to_canvas(x, y)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    for box in tile_boxes:
        draw.rectangle([*to_canvas(box[0], box[1]), *to_canvas(box[2], box[3])], outline=(0, 0, 0), width=1)

    header = Image.new("RGB", (PANEL_WIDTH, 34), (255, 255, 255))
    ImageDraw.Draw(header).text((6, 4), title, fill=(0, 0, 0))
    out = Image.new("RGB", (PANEL_WIDTH, header.height + canvas.height), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(canvas, (0, header.height))
    return out


def audit_patient(
    patient_id: str,
    slide,
    scale_json: Optional[dict],
    spots: pd.DataFrame,
    tiles_df: pd.DataFrame,
    fullres_pixel_size: float,
    tile_size_px: int,
    output_dir: Path,
    hires_image: Optional[Image.Image] = None,
    n_sample_tiles: int = 3,
    model_name: str = "phikon_v2",
) -> dict:
    """
    Audit one patient: returns an inventory row and writes
      overlays/{patient}.jpg          left: the opened slide with spots and tiles where the extractor
                                      placed them; right: Space Ranger's own hires image with the same
                                      spots and tiles where they truly are (when `hires_image` is given)
      sample_crops/{patient}_{tile}.png   patches exactly as fed to the model, after resize
    """
    spec = PATCH_SPECS[model_name]
    output_dir = Path(output_dir)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_crops").mkdir(parents=True, exist_ok=True)

    slide_w, slide_h = slide.dimensions
    scale = legacy_scale(scale_json, slide_w, fullres_pixel_size, tile_size_px)
    s = scale["scale_used"]
    eff_tile = scale["effective_tile_size_px"]
    scale_json = scale_json or {}

    # === patch windows per tile, as the extraction placed them
    n_windows = n_clamped = n_partly = n_fully = 0
    windows_by_tile = {}
    for _, row in tiles_df.iterrows():
        x0, y0 = tile_origin(row, s)
        windows = legacy_patch_windows(x0, y0, eff_tile, slide_w, slide_h, scale["effective_um_per_px"], spec)
        windows_by_tile[row["tile_id"]] = windows
        n_windows += len(windows)
        n_clamped += int(windows["clamped"].sum())
        n_partly += int(windows["clamped"].any())
        n_fully += int(windows["clamped"].all())
    patch_px = next(iter(windows_by_tile.values())).attrs["patch_px"] if windows_by_tile else None

    # === does the opened slide have the size of the image the scale factor refers to?
    in_tissue = spots[spots["in_tissue"] == 1]
    spots_xy_slide = in_tissue[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float) * s
    outside = ((spots_xy_slide[:, 0] < 0) | (spots_xy_slide[:, 0] >= slide_w) |
               (spots_xy_slide[:, 1] < 0) | (spots_xy_slide[:, 1] >= slide_h))
    implied_fullres_w = implied_fullres_h = implied_target_w = None
    hires_scalef = scale_json.get("tissue_hires_scalef")
    if hires_image is not None and hires_scalef:
        implied_fullres_w = hires_image.size[0] / hires_scalef
        implied_fullres_h = hires_image.size[1] / hires_scalef
        if "regist_target_img_scalef" in scale_json:
            implied_target_w = implied_fullres_w * float(scale_json["regist_target_img_scalef"])

    row_out = {
        "patient_id": patient_id,
        "slide_width": slide_w,
        "slide_height": slide_h,
        "levels": getattr(slide, "level_count", None),
        "mpp_from_metadata": dict(getattr(slide, "properties", {})).get("openslide.mpp-x"),
        "regist_target_img_scalef": scale_json.get("regist_target_img_scalef"),
        "tissue_hires_scalef": hires_scalef,
        "spot_diameter_fullres": scale_json.get("spot_diameter_fullres"),
        **scale,
        "fullres_um_per_px": fullres_pixel_size,
        "patch_px_native": patch_px,
        "upsampling_to_model_input": (spec.patch_size_px / patch_px) if patch_px else None,
        "n_tiles": len(tiles_df),
        "n_tiles_partly_outside_slide": n_partly,
        "n_tiles_fully_outside_slide": n_fully,
        "n_patch_windows": n_windows,
        "n_patch_windows_clamped": n_clamped,
        "n_spots_in_tissue": len(in_tissue),
        "pct_spots_outside_slide": float(100 * outside.mean()) if len(outside) else None,
        "implied_fullres_width": implied_fullres_w,
        "implied_fullres_height": implied_fullres_h,
        "implied_regist_target_width": implied_target_w,
        "slide_matches_regist_target": (abs(slide_w - implied_target_w) / implied_target_w < 0.02) if implied_target_w else None,
    }

    # === overlay
    colors = _spot_colors(in_tissue, tiles_df, tile_size_px)
    spot_radius_fullres = float(scale_json.get("spot_diameter_fullres", 55.0 / fullres_pixel_size)) / 2
    slide_view = slide.read_region((0, 0), 0, (slide_w, slide_h)) if slide_w * slide_h <= 40_000_000 else slide.get_thumbnail((3000, 3000))
    view_zoom = slide_view.size[0] / slide_w
    boxes_fullres = tiles_df[["x_min_fullres", "y_min_fullres", "x_max_fullres", "y_max_fullres"]].to_numpy(dtype=float)
    left = _draw_panel(
        slide_view, slide_view.size, spots_xy_slide * view_zoom, colors,
        [tuple(b * s * view_zoom) for b in boxes_fullres], spot_radius_fullres * s * view_zoom,
        f"{patient_id}  EXTRACTOR VIEW: slide {slide_w}x{slide_h}, scale {s:.4f}, "
        f"{scale['effective_um_per_px']:.2f} um/px\n{row_out['pct_spots_outside_slide']:.0f}% of spots and "
        f"{n_partly}/{len(tiles_df)} tiles outside the slide (red box = slide edge)",
    )
    panels = [left]
    if hires_image is not None and hires_scalef:
        xy = in_tissue[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float) * hires_scalef
        panels.append(_draw_panel(
            hires_image, hires_image.size, xy, colors,
            [tuple(b * hires_scalef) for b in boxes_fullres], spot_radius_fullres * hires_scalef,
            "REFERENCE: Space Ranger tissue_hires_image, spots and tiles where they truly are\n"
            f"fullres image is {implied_fullres_w:.0f}x{implied_fullres_h:.0f} px at {fullres_pixel_size:.2f} um/px",
        ))
    overlay = Image.new("RGB", (PANEL_WIDTH * len(panels), max(p.height for p in panels)), (255, 255, 255))
    for i, panel in enumerate(panels):
        overlay.paste(panel, (PANEL_WIDTH * i, 0))
    overlay.save(output_dir / "overlays" / f"{patient_id}.jpg", quality=85)   # JPEG keeps 151 overlays committable

    # === sample patches exactly as fed to the model
    n_saved = 0
    tile_ids = list(windows_by_tile)
    picks = sorted({0, len(tile_ids) // 2, len(tile_ids) - 1})[:n_sample_tiles] if tile_ids else []
    for pick in picks:
        tile_id = tile_ids[pick]
        windows = windows_by_tile[tile_id]
        for w in windows.itertuples():
            crop = slide.read_region((int(w.read_x), int(w.read_y)), 0, (patch_px, patch_px)).convert("RGB")
            if _has_enough_tissue(crop, LEGACY_MIN_TISSUE_FRACTION):
                crop.resize((spec.patch_size_px, spec.patch_size_px), Image.BICUBIC).save(
                    output_dir / "sample_crops" / f"{patient_id}_{tile_id}.png")
                n_saved += 1
                break
    row_out["n_sample_crops"] = n_saved
    return row_out
