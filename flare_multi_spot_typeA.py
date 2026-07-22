#!/usr/bin/env python3
"""
Measure camera flare for 13 circular patches.

Folder convention:
  step1*.bmp: dark circular patches on a bright field
  step2*.bmp: black field with a bright segmented circular boundary
  step3*.bmp: dark circular patches on a bright field at longer exposure

Per circular patch:
  D70 = sqrt(W^2 + H^2) / 70
  YW1 = mean of a bright annulus outside the step1 patch, separated by D70
  YB2 = mean of the central dark region inside the step2 bright ring / 8
  YB3 = mean of the step3 dark patch inner region / 8
  flare = (YB3 - YB2) / YW1 * 100

The bright annulus has the same area as the reduced dark circular region.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


folder = r"D:\Data\ET_camera\NIL\data_260721_flare"
expected_patch_count = 13


def find_step_file(prefix: str) -> Path:
    matches = sorted(Path(folder).glob(f"{prefix}*.bmp"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {prefix}*.bmp file, found {len(matches)}")
    return matches[0]


def read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def otsu_threshold(values: np.ndarray) -> float:
    clipped = np.clip(values.ravel(), 0, 255).astype(np.uint8)
    hist = np.bincount(clipped, minlength=256).astype(np.float64)
    total = clipped.size
    sum_total = np.dot(np.arange(256), hist)

    sum_background = 0.0
    weight_background = 0.0
    best_variance = -1.0
    best_threshold = 0

    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = threshold

    return float(best_threshold)


def gaussian_blur_array(gray: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    return np.array(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def connected_components(mask: np.ndarray) -> list[dict]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask[start_y, start_x]:
                continue

            queue = deque([(start_x, start_y)])
            visited[start_y, start_x] = True
            pixels = []

            while queue:
                x, y = queue.popleft()
                pixels.append((x, y))
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx == x and ny == y:
                            continue
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            points = np.array(pixels, dtype=np.float32)
            xs = points[:, 0]
            ys = points[:, 1]
            components.append(
                {
                    "points": points,
                    "area": float(len(points)),
                    "bbox": (
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max()),
                        float(ys.max()),
                    ),
                    "center": (float(xs.mean()), float(ys.mean())),
                }
            )

    return components


def find_dark_patches(gray: np.ndarray) -> tuple[list[dict], np.ndarray]:
    height, width = gray.shape
    blurred = gaussian_blur_array(gray, radius=1.1)

    margin_x = int(width * 0.08)
    margin_y = int(height * 0.08)
    roi = blurred[margin_y : height - margin_y, margin_x : width - margin_x]
    # The field has strong lens falloff, so Otsu can merge darker background
    # regions with the patches. A low percentile isolates the dark disks.
    threshold = float(np.percentile(roi, 5.0))

    mask = blurred <= threshold
    search_mask = np.zeros_like(mask, dtype=bool)
    search_mask[margin_y : height - margin_y, margin_x : width - margin_x] = True
    mask &= search_mask

    components = connected_components(mask)
    image_area = width * height
    candidates = []
    for component in components:
        x0, y0, x1, y1 = component["bbox"]
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        area = component["area"]
        if area < 0.0012 * image_area or area > 0.025 * image_area:
            continue
        if bbox_w < 8 or bbox_h < 8:
            continue
        aspect = max(bbox_w, bbox_h) / max(1.0, min(bbox_w, bbox_h))
        if aspect > 1.45:
            continue

        radius = (bbox_w + bbox_h) / 4.0
        fill_ratio = area / (math.pi * radius * radius)
        if fill_ratio < 0.45 or fill_ratio > 1.25:
            continue

        component["radius"] = float(radius)
        component["fill_ratio"] = float(fill_ratio)
        candidates.append(component)

    if len(candidates) < expected_patch_count:
        raise RuntimeError(
            f"Found {len(candidates)} circular patches; expected {expected_patch_count}."
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    patches = candidates[:expected_patch_count]
    patches.sort(key=lambda item: (item["center"][1], item["center"][0]))
    for patch_id, patch in enumerate(patches, start=1):
        patch["id"] = patch_id

    return patches, mask


def find_step2_center_circle(gray: np.ndarray) -> tuple[dict, np.ndarray]:
    height, width = gray.shape
    blurred = gaussian_blur_array(gray, radius=0.8)

    roi_fraction = 0.45
    roi_w = int(width * roi_fraction)
    roi_h = int(height * roi_fraction)
    x0 = (width - roi_w) // 2
    y0 = (height - roi_h) // 2
    x1 = x0 + roi_w
    y1 = y0 + roi_h
    roi = blurred[y0:y1, x0:x1]

    threshold = max(80.0, float(np.percentile(roi, 99.0)) * 0.65)
    mask = np.zeros_like(gray, dtype=bool)
    mask[y0:y1, x0:x1] = blurred[y0:y1, x0:x1] >= threshold

    ys, xs = np.where(mask)
    if len(xs) < 8:
        raise RuntimeError("Could not find the bright circular boundary in step2.")

    cx = float(xs.mean())
    cy = float(ys.mean())
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    inner_edge_radius = float(np.percentile(distances, 5))
    outer_edge_radius = float(np.percentile(distances, 95))

    return {
        "id": 0,
        "center": (cx, cy),
        "radius": inner_edge_radius,
        "outer_radius": outer_edge_radius,
    }, mask


def disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cx, cy = center
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius


def annulus_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    outer = disk_mask(shape, center, outer_radius)
    inner = disk_mask(shape, center, inner_radius)
    return outer & ~inner


def masked_mean(gray: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    pixels = gray[mask]
    if pixels.size == 0:
        return float("nan"), 0
    return float(pixels.mean()), int(pixels.size)


def measure_step1_patch(gray: np.ndarray, patch: dict, d70: float) -> dict:
    radius = patch["radius"]
    dark_radius = radius - d70
    if dark_radius <= 1:
        raise RuntimeError(f"D70 is too large for patch {patch['id']}.")

    bright_inner = radius + d70
    bright_outer = math.sqrt(bright_inner * bright_inner + dark_radius * dark_radius)

    dark = disk_mask(gray.shape, patch["center"], dark_radius)
    bright = annulus_mask(gray.shape, patch["center"], bright_inner, bright_outer)
    dark_mean, dark_pixels = masked_mean(gray, dark)
    bright_mean, bright_pixels = masked_mean(gray, bright)

    return {
        "dark_mean": dark_mean,
        "dark_pixels": dark_pixels,
        "bright_mean": bright_mean,
        "bright_pixels": bright_pixels,
        "inner_radius": dark_radius,
        "bright_inner_radius": bright_inner,
        "bright_outer_radius": bright_outer,
    }


def measure_dark_patch(gray: np.ndarray, patch: dict, d70: float) -> dict:
    radius = patch["radius"]
    dark_radius = radius - d70
    if dark_radius <= 1:
        raise RuntimeError(f"D70 is too large for patch {patch.get('id', 'center')}.")

    dark = disk_mask(gray.shape, patch["center"], dark_radius)
    dark_mean, dark_pixels = masked_mean(gray, dark)
    return {
        "dark_mean": dark_mean,
        "dark_pixels": dark_pixels,
        "inner_radius": dark_radius,
    }


def match_patches(reference: list[dict], detected: list[dict]) -> list[dict]:
    remaining = detected.copy()
    matched = []
    for ref in reference:
        best_index = min(
            range(len(remaining)),
            key=lambda i: np.linalg.norm(
                np.array(remaining[i]["center"]) - np.array(ref["center"])
            ),
        )
        patch = remaining.pop(best_index)
        patch["id"] = ref["id"]
        matched.append(patch)
    matched.sort(key=lambda item: item["id"])
    return matched


def draw_circle(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: float, color: str) -> None:
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=1)


def draw_patch_overlay(
    image: Image.Image,
    patches: list[dict],
    measurements: dict[int, dict],
    output_path: Path,
    *,
    show_bright: bool,
) -> None:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    overlay = Image.new("RGBA", marked.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for patch in patches:
        patch_id = patch["id"]
        measurement = measurements[patch_id]
        cx, cy = patch["center"]

        draw_circle(draw, patch["center"], patch["radius"], "lime")
        draw_circle(draw, patch["center"], measurement["inner_radius"], "yellow")
        r = measurement["inner_radius"]
        overlay_draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(255, 255, 0, 55),
        )

        if show_bright:
            draw_circle(draw, patch["center"], measurement["bright_inner_radius"], "cyan")
            draw_circle(draw, patch["center"], measurement["bright_outer_radius"], "cyan")

        draw.text((cx - 4, cy - 5), str(patch_id), fill="red")

    marked = Image.alpha_composite(marked.convert("RGBA"), overlay).convert("RGB")
    marked.save(output_path)


def draw_step2_overlay(
    image: Image.Image,
    center_patch: dict,
    measurement: dict,
    output_path: Path,
) -> None:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw_circle(draw, center_patch["center"], center_patch["outer_radius"], "cyan")
    draw_circle(draw, center_patch["center"], center_patch["radius"], "lime")
    draw_circle(draw, center_patch["center"], measurement["inner_radius"], "yellow")
    cx, cy = center_patch["center"]
    draw.text((cx - 34, cy + 12), f"YB2 {measurement['dark_mean'] / 8.0:.3f}", fill="yellow")
    marked.save(output_path)


def write_mask(mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_path)


def main() -> None:
    folder_path = Path(folder)
    step1_path = find_step_file("step1")
    step2_path = find_step_file("step2")
    step3_path = find_step_file("step3")

    step1_gray = read_gray(step1_path)
    step2_gray = read_gray(step2_path)
    step3_gray = read_gray(step3_path)
    step1_rgb = read_rgb(step1_path)
    step2_rgb = read_rgb(step2_path)
    step3_rgb = read_rgb(step3_path)

    height, width = step1_gray.shape
    d70 = float(np.hypot(height, width) / 70.0)

    step1_patches, step1_mask = find_dark_patches(step1_gray)
    step3_patches_raw, step3_mask = find_dark_patches(step3_gray)
    step3_patches = match_patches(step1_patches, step3_patches_raw)

    step2_circle, step2_mask = find_step2_center_circle(step2_gray)
    step2_measurement = measure_dark_patch(step2_gray, step2_circle, d70)
    yb2 = step2_measurement["dark_mean"] / 8.0

    step1_measurements = {}
    step3_measurements = {}
    rows = []

    for step1_patch, step3_patch in zip(step1_patches, step3_patches):
        patch_id = step1_patch["id"]
        step1_measurement = measure_step1_patch(step1_gray, step1_patch, d70)
        step3_measurement = measure_dark_patch(step3_gray, step3_patch, d70)
        yw1 = step1_measurement["bright_mean"]
        yb3 = step3_measurement["dark_mean"] / 8.0
        flare = (yb3 - yb2) / yw1 * 100.0

        step1_measurements[patch_id] = step1_measurement
        step3_measurements[patch_id] = step3_measurement
        rows.append(
            {
                "patch_id": patch_id,
                "x_step1": step1_patch["center"][0],
                "y_step1": step1_patch["center"][1],
                "radius_step1": step1_patch["radius"],
                "x_step3": step3_patch["center"][0],
                "y_step3": step3_patch["center"][1],
                "radius_step3": step3_patch["radius"],
                "YW1": yw1,
                "YB2": yb2,
                "YB3": yb3,
                "flare_percent": flare,
            }
        )

    csv_path = folder_path / "circular_patch_flare_results.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    draw_patch_overlay(
        step1_rgb,
        step1_patches,
        step1_measurements,
        folder_path / f"{step1_path.name}_circular_regions.png",
        show_bright=True,
    )
    draw_step2_overlay(
        step2_rgb,
        step2_circle,
        step2_measurement,
        folder_path / f"{step2_path.name}_YB2_region.png",
    )
    draw_patch_overlay(
        step3_rgb,
        step3_patches,
        step3_measurements,
        folder_path / f"{step3_path.name}_YB3_regions.png",
        show_bright=False,
    )
    write_mask(step1_mask, folder_path / f"{step1_path.name}_patch_mask.png")
    write_mask(step2_mask, folder_path / f"{step2_path.name}_bright_mask.png")
    write_mask(step3_mask, folder_path / f"{step3_path.name}_patch_mask.png")

    print(f"Folder: {folder}")
    print(f"D70: {d70:.3f} pixels")
    print(f"YB2: {yb2:.6f}")
    print()
    print("patch_id, YW1, YB2, YB3, flare_percent")
    for row in rows:
        print(
            f"{row['patch_id']:2d}, "
            f"{row['YW1']:.6f}, "
            f"{row['YB2']:.6f}, "
            f"{row['YB3']:.6f}, "
            f"{row['flare_percent']:.6f}"
        )
    print()
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
