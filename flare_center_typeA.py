#%%
#!/usr/bin/env python3
"""
Find the four corners of a dark rectangle in an image and save a marked preview.

Usage:
    python find_dark_rectangle_corners.py input.png
    python find_dark_rectangle_corners.py input.png --output marked.png

Dependencies:
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import os

# folder = r'D:\Data\ET_camera\FlareTest\data_260624'
# fname0 = r'trans_white_DM250_CLR76_exp3_5ms_gain0dB.bmp'
# fname1 = r'trans_black_DM250_CLR76_exp28ms_gain0dB.bmp'
# fname2 = r'trans_white_DM250_CLR76_exp28ms_gain0dB.bmp'

folder = r'D:\Data\ET_camera\NIL\data_260717_flare'
fname0 = r'step1_exp04ms_gainx1_led920_850nm.bmp'
fname1 = r'step2_exp3_2ms_gainx1_led920_850nm.bmp'
fname2 = r'step3_exp1_6ms_gainx1_led920_850nm.bmp'


REGION_COLORS = {
    "dark_inner": (0, 255, 255),
    "bright_top": (255, 128, 0),
    "bright_right": (255, 0, 255),
    "bright_bottom": (0, 165, 255),
    "bright_left": (255, 255, 0),
}


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def make_mask_from_contour(image_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def find_dark_rectangle(
    image: np.ndarray,
    *,
    search_fraction: float = 0.35,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    image_center = np.array([width / 2.0, height / 2.0])

    # Work from the central part of the image because the target rectangle is
    # known to be there. This keeps a dark object on the side from winning.
    roi_w = int(width * search_fraction)
    roi_h = int(height * search_fraction)
    roi_x0 = max(0, (width - roi_w) // 2)
    roi_y0 = max(0, (height - roi_h) // 2)
    roi_x1 = min(width, roi_x0 + roi_w)
    roi_y1 = min(height, roi_y0 + roi_h)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    center_roi = blurred[roi_y0:roi_y1, roi_x0:roi_x1]
    otsu_threshold, _ = cv2.threshold(
        center_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Otsu separates the dark central rectangle from its brighter surroundings.
    # A percentile blend can become too permissive when the background saturates.
    threshold_value = float(otsu_threshold)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[blurred <= threshold_value] = 255

    search_mask = np.zeros_like(mask)
    search_mask[roi_y0:roi_y1, roi_x0:roi_x1] = 255
    mask = cv2.bitwise_and(mask, search_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No dark rectangle-like region was found.")

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_fraction * width * height:
            continue
        if area > max_area_fraction * width * height:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        rect_area = rw * rh
        fill_ratio = area / rect_area if rect_area else 0
        center_distance = np.linalg.norm(np.array([cx, cy]) - image_center)
        box = cv2.boxPoints(rect)
        center_inside = cv2.pointPolygonTest(box, tuple(image_center), False) >= 0

        x, y, w, h = cv2.boundingRect(contour)
        center_overlap_x = max(0, min(x + w, roi_x1) - max(x, roi_x0)) / max(1, w)
        center_overlap_y = max(0, min(y + h, roi_y1) - max(y, roi_y0)) / max(1, h)
        center_overlap = center_overlap_x * center_overlap_y

        # Strongly prefer the region that contains the image center. Area still
        # matters, but no longer overwhelms the central target.
        center_bonus = 8.0 if center_inside else 1.0
        distance_penalty = 1.0 + 4.0 * center_distance / max(width, height)
        score = area * fill_ratio * center_overlap * center_bonus / distance_penalty
        candidates.append((score, rect, contour))

    if not candidates:
        raise RuntimeError("Dark regions were found, but none looked like a rectangle.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    rect = candidates[0][1]
    contour = candidates[0][2]
    box = cv2.boxPoints(rect)
    return order_corners(box), contour, mask


def find_dark_rectangle_from_bright_border(
    image: np.ndarray,
    *,
    search_fraction: float = 0.35,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    image_center = np.array([width / 2.0, height / 2.0])

    roi_w = int(width * search_fraction)
    roi_h = int(height * search_fraction)
    roi_x0 = max(0, (width - roi_w) // 2)
    roi_y0 = max(0, (height - roi_h) // 2)
    roi_x1 = min(width, roi_x0 + roi_w)
    roi_y1 = min(height, roi_y0 + roi_h)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    center_roi = blurred[roi_y0:roi_y1, roi_x0:roi_x1]
    bright_threshold = max(80.0, float(np.percentile(center_roi, 99.0)) * 0.65)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[blurred >= bright_threshold] = 255

    search_mask = np.zeros_like(mask)
    search_mask[roi_y0:roi_y1, roi_x0:roi_x1] = 255
    mask = cv2.bitwise_and(mask, search_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 4 or max(w, h) < 12:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        components.append(
            {
                "contour": contour,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "distance": float(np.linalg.norm(np.array([cx, cy]) - image_center)),
            }
        )

    if not components:
        raise RuntimeError("No bright border components were found near the image center.")

    horizontal = [c for c in components if c["w"] >= 2.0 * c["h"]]
    vertical = [c for c in components if c["h"] >= 2.0 * c["w"]]
    if len(horizontal) < 2 or len(vertical) < 2:
        raise RuntimeError(
            "Could not find two horizontal and two vertical bright border segments."
        )

    horizontal.sort(key=lambda c: (abs(c["cx"] - image_center[0]), c["distance"]))
    vertical.sort(key=lambda c: (abs(c["cy"] - image_center[1]), c["distance"]))
    top, bottom = sorted(horizontal[:2], key=lambda c: c["cy"])
    left, right = sorted(vertical[:2], key=lambda c: c["cx"])

    # The dark rectangle boundary is inferred from the inner edges of the bright
    # border segments. The white border has dark gaps at the corners, so these
    # four inner lines are more reliable than trying to find a closed contour.
    left_x = float(left["x"] + left["w"] - 1)
    right_x = float(right["x"])
    top_y = float(top["y"] + top["h"] - 1)
    bottom_y = float(bottom["y"])

    corners = np.array(
        [[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y]],
        dtype=np.float32,
    )

    selected_contours = [top["contour"], right["contour"], bottom["contour"], left["contour"]]
    return corners, selected_contours, mask


def polygon_from_local_rect(
    local_to_image: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    local_points = np.array(
        [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(local_points, local_to_image)[0]


def mean_inside_polygon(gray: np.ndarray, polygon: np.ndarray) -> tuple[float, int]:
    region_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(region_mask, np.round(polygon).astype(np.int32), 255)
    pixel_count = int(cv2.countNonZero(region_mask))
    if pixel_count == 0:
        return float("nan"), 0
    return float(cv2.mean(gray, mask=region_mask)[0]), pixel_count


def calculate_region_measurements(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    d70_divisor: float = 70.0,
) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    d70 = float(np.hypot(height, width) / d70_divisor)

    top_width = float(np.linalg.norm(corners[1] - corners[0]))
    bottom_width = float(np.linalg.norm(corners[2] - corners[3]))
    left_height = float(np.linalg.norm(corners[3] - corners[0]))
    right_height = float(np.linalg.norm(corners[2] - corners[1]))
    dark_width = (top_width + bottom_width) / 2.0
    dark_height = (left_height + right_height) / 2.0
    reduced_width = dark_width - 2.0 * d70
    reduced_height = dark_height - 2.0 * d70

    if reduced_width <= 0 or reduced_height <= 0:
        raise RuntimeError(
            f"D70 ({d70:.2f}) is too large for the detected rectangle "
            f"({dark_width:.2f} x {dark_height:.2f})."
        )

    local_rect = np.array(
        [[0, 0], [dark_width, 0], [dark_width, dark_height], [0, dark_height]],
        dtype=np.float32,
    )
    local_to_image = cv2.getPerspectiveTransform(local_rect, corners.astype(np.float32))

    region_specs = [
        ("dark_inner", d70, d70, dark_width - d70, dark_height - d70),
        ("bright_top", d70, -d70 - reduced_height, dark_width - d70, -d70),
        (
            "bright_right",
            dark_width + d70,
            d70,
            dark_width + d70 + reduced_width,
            dark_height - d70,
        ),
        (
            "bright_bottom",
            d70,
            dark_height + d70,
            dark_width - d70,
            dark_height + d70 + reduced_height,
        ),
        ("bright_left", -d70 - reduced_width, d70, -d70, dark_height - d70),
    ]

    regions = []
    bright_values = []
    bright_pixel_counts = []
    for name, x0, y0, x1, y1 in region_specs:
        polygon = polygon_from_local_rect(local_to_image, x0, y0, x1, y1)
        mean_value, pixel_count = mean_inside_polygon(gray, polygon)
        regions.append(
            {
                "name": name,
                "polygon": polygon,
                "mean": mean_value,
                "pixel_count": pixel_count,
            }
        )
        if name.startswith("bright_") and pixel_count > 0:
            bright_values.append(mean_value * pixel_count)
            bright_pixel_counts.append(pixel_count)

    dark_mean = regions[0]["mean"]
    bright_mean = sum(bright_values) / sum(bright_pixel_counts)
    ratio = bright_mean / dark_mean if dark_mean != 0 else float("inf")

    return {
        "W": width,
        "H": height,
        "D70": d70,
        "dark_width": dark_width,
        "dark_height": dark_height,
        "DW": reduced_width,
        "DH": reduced_height,
        "dark_mean": dark_mean,
        "bright_mean": bright_mean,
        "bright_dark_ratio": ratio,
        "regions": regions,
    }


def calculate_dark_only_measurement(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    value_name: str,
    d70_divisor: float = 70.0,
) -> dict:
    measurements = calculate_region_measurements(image, corners, d70_divisor=d70_divisor)
    dark_region = measurements["regions"][0]
    dark_value_divided_by_8 = measurements["dark_mean"] / 8.0
    return {
        **measurements,
        value_name: dark_value_divided_by_8,
        "regions": [dark_region],
    }


def draw_marked_image(
    image: np.ndarray,
    corners: np.ndarray,
    contour: np.ndarray | list[np.ndarray] | None = None,
    measurements: dict | None = None,
) -> np.ndarray:
    marked = image.copy()
    int_corners = np.round(corners).astype(int)
    image_height, image_width = marked.shape[:2]
    min_image_size = min(image_width, image_height)
    is_small_image = min_image_size < 700
    line_thickness = 1 if is_small_image else 3
    point_radius = 4 if is_small_image else 8
    font_scale = 0.32 if is_small_image else 0.55
    text_thickness = 1 if is_small_image else 2

    if isinstance(contour, list):
        cv2.drawContours(marked, contour, -1, (255, 0, 0), line_thickness)
    elif contour is not None:
        cv2.drawContours(marked, [contour], -1, (255, 0, 0), line_thickness)
    cv2.polylines(
        marked,
        [int_corners],
        isClosed=True,
        color=(0, 255, 0),
        thickness=line_thickness,
    )

    if measurements is not None:
        overlay = marked.copy()
        for region in measurements["regions"]:
            color = REGION_COLORS[region["name"]]
            polygon = np.round(region["polygon"]).astype(np.int32)
            cv2.fillConvexPoly(overlay, polygon, color)
        marked = cv2.addWeighted(overlay, 0.22, marked, 0.78, 0)

        for region in measurements["regions"]:
            color = REGION_COLORS[region["name"]]
            polygon = np.round(region["polygon"]).astype(np.int32)
            cv2.polylines(
                marked,
                [polygon],
                isClosed=True,
                color=color,
                thickness=line_thickness,
            )
            label_xy = np.round(region["polygon"].mean(axis=0)).astype(int)
            region_label = region["name"]
            if is_small_image:
                region_label = region_label.replace("dark_inner", "dark")
                region_label = region_label.replace("bright_", "")
            cv2.putText(
                marked,
                f"{region_label} {region['mean']:.1f}",
                (int(label_xy[0]) - (26 if is_small_image else 70), int(label_xy[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                text_thickness,
                cv2.LINE_AA,
            )

    labels = ["TL", "TR", "BR", "BL"]
    for label, (x, y) in zip(labels, int_corners):
        cv2.circle(marked, (int(x), int(y)), point_radius, (0, 0, 255), -1)
        corner_text = label if is_small_image else f"{label} ({int(x)}, {int(y)})"
        cv2.putText(
            marked,
            corner_text,
            (int(x) + point_radius + 2, int(y) - point_radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34 if is_small_image else 0.6,
            (0, 0, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return marked


def print_corners(corners: np.ndarray) -> None:
    print("Corners, ordered clockwise from top-left:")
    for name, (x, y) in zip(["top_left", "top_right", "bottom_right", "bottom_left"], corners):
        print(f"{name}: ({x:.1f}, {y:.1f})")


def print_region_measurements(measurements: dict) -> None:
    print("Region measurements from grayscale image:")
    print(f"W: {measurements['W']} pixels")
    print(f"H: {measurements['H']} pixels")
    print(f"D70: {measurements['D70']:.3f} pixels")
    print(f"Detected dark boundary size: {measurements['dark_width']:.3f} x {measurements['dark_height']:.3f} pixels")
    print(f"Reduced region size DW x DH: {measurements['DW']:.3f} x {measurements['DH']:.3f} pixels")
    for region in measurements["regions"]:
        print(
            f"{region['name']}: mean={region['mean']:.3f}, "
            f"pixels={region['pixel_count']}"
        )
    if "bright_mean" in measurements and len(measurements["regions"]) > 1:
        print(f"Average bright mean: {measurements['bright_mean']:.3f}")
        print(f"Dark mean: {measurements['dark_mean']:.3f}")
        print(f"Bright / dark ratio: {measurements['bright_dark_ratio']:.6f}")
    for value_name in ("YB2", "YB3"):
        if value_name in measurements:
            print(f"Dark mean / 8, {value_name}: {measurements[value_name]:.6f}")


def main() -> None:
    # parser = argparse.ArgumentParser()
    # parser.add_argument("image", type=Path, help="Path to the input image")
    # parser.add_argument(
    #     "--output",
    #     type=Path,
    #     help="Path for the marked output image. Defaults to '<input>_corners.png'.",
    # )
    # args = parser.parse_args()

    fname = os.path.join(folder, fname0)
    image = read_image(fname)

    corners, contour, mask = find_dark_rectangle(image)
    measurements = calculate_region_measurements(image, corners)

    # output = args.output
    output = os.path.join(folder, f"{fname0}_regions.png")
    mask_output = os.path.join(folder, f"{fname0}_mask.png")
    # if output is None:
    #     output = args.image.with_name(f"{args.image.stem}_corners.png")

    marked = draw_marked_image(image, corners, contour, measurements)
    if not cv2.imwrite(str(output), marked):
        raise SystemExit(f"Could not write output image: {output}")
    if not cv2.imwrite(str(mask_output), mask):
        raise SystemExit(f"Could not write mask image: {mask_output}")

    print(f"===== {fname0} =====")
    print_corners(corners)
    print()
    print_region_measurements(measurements)
    print()
    print(f"Marked image written to: {output}")
    print(f"Mask image written to: {mask_output}")

    fname_black = os.path.join(folder, fname1)
    black_image = read_image(fname_black)
    black_corners, black_contour, black_mask = find_dark_rectangle_from_bright_border(black_image)
    black_measurements = calculate_dark_only_measurement(
        black_image,
        black_corners,
        value_name="YB2",
    )

    black_output = os.path.join(folder, f"{fname1}_YB2_region.png")
    black_mask_output = os.path.join(folder, f"{fname1}_bright_border_mask.png")
    black_marked = draw_marked_image(
        black_image,
        black_corners,
        black_contour,
        black_measurements,
    )
    if not cv2.imwrite(str(black_output), black_marked):
        raise SystemExit(f"Could not write output image: {black_output}")
    if not cv2.imwrite(str(black_mask_output), black_mask):
        raise SystemExit(f"Could not write mask image: {black_mask_output}")

    print()
    print(f"===== {fname1} =====")
    print_corners(black_corners)
    print()
    print_region_measurements(black_measurements)
    print()
    print(f"Marked image written to: {black_output}")
    print(f"Bright border mask image written to: {black_mask_output}")

    fname_white_long = os.path.join(folder, fname2)
    white_long_image = read_image(fname_white_long)
    white_long_corners, white_long_contour, white_long_mask = find_dark_rectangle(
        white_long_image
    )
    white_long_measurements = calculate_dark_only_measurement(
        white_long_image,
        white_long_corners,
        value_name="YB3",
    )

    white_long_output = os.path.join(folder, f"{fname2}_YB3_region.png")
    white_long_mask_output = os.path.join(folder, f"{fname2}_mask.png")
    white_long_marked = draw_marked_image(
        white_long_image,
        white_long_corners,
        white_long_contour,
        white_long_measurements,
    )
    if not cv2.imwrite(str(white_long_output), white_long_marked):
        raise SystemExit(f"Could not write output image: {white_long_output}")
    if not cv2.imwrite(str(white_long_mask_output), white_long_mask):
        raise SystemExit(f"Could not write mask image: {white_long_mask_output}")

    print()
    print(f"===== {fname2} =====")
    print_corners(white_long_corners)
    print()
    print_region_measurements(white_long_measurements)
    print()
    print(f"Marked image written to: {white_long_output}")
    print(f"Mask image written to: {white_long_mask_output}")

    yw1 = measurements["bright_mean"]
    yb2 = black_measurements["YB2"]
    yb3 = white_long_measurements["YB3"]
    flare = (yb3 - yb2) / yw1 * 100.0

    print()
    print("===== Camera flare =====")
    print(f"YW1, average white region from {fname0}: {yw1:.6f}")
    print(f"YB2, dark value from {fname1} / 8: {yb2:.6f}")
    print(f"YB3, dark value from {fname2} / 8: {yb3:.6f}")
    print(f"flare = (YB3 - YB2) / YW1 * 100 = {flare:.6f}%")


if __name__ == "__main__":
    main()

# %%
