#!/usr/bin/env python3
"""
识别培养皿中菌苔扩散范围，使用绿色标注边界。

流程：
  灰度图 → 蒙版非黑区域 → OTSU + 偏移量 → 形态学清理
  → 取含中心最大连通域 → Canny 边界提取 → 膨胀加粗 → 绿色叠加
"""

import os, sys, cv2
import numpy as np
from pathlib import Path

# ── 可调参数 ──
THRESHOLD_OFFSET = 10   # OTSU 阈值上调量（越大标注越保守）
LINE_THICKNESS = 4       # 绿色描边线宽度（px）

# ── 路径 ──
DEFAULT_INPUT = (
    "/Volumes/科研 2/mcr-1 影响菌毛系统的表达/游泳运动"
    "/20250511 加药后游泳运动/result_bg_black_gray"
)
DEFAULT_OUTPUT = os.path.normpath(
    os.path.join(DEFAULT_INPUT, "..", "result_lawn_only")
)


def detect_lawn(gray, offset=THRESHOLD_OFFSET, line_w=LINE_THICKNESS):
    """
    检测菌苔边界，返回标注后的 BGR 图。
    
    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (H, W)，黑底。
    offset : int
        OTSU 阈值上调量。
    line_w : int
        绿色边界线宽度（px）。
    
    Returns
    -------
    result : np.ndarray
        叠加绿色边界后的 BGR 图 (H, W, 3)。
    info : dict
        {'otsu_val', 'adjusted_th', 'lawn_area', 'lawn_pct'}
    """
    h, w = gray.shape[:2]

    # 1. 只分析培养皿内部（排除黑色背景）
    dish_mask = np.uint8(gray > 10) * 255
    dish_only = cv2.bitwise_and(gray, gray, mask=dish_mask)

    # 2. OTSU 自适应阈值 + 偏移
    otsu_val, _ = cv2.threshold(dish_only, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = int(otsu_val)
    adjusted_th = min(otsu_val + offset, 255)
    _, bin_img = cv2.threshold(dish_only, adjusted_th, 255, cv2.THRESH_BINARY)

    # 3. 形态学清理
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)

    # 4. 取含图像中心的最大连通域
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = w // 2, h // 2
    candidates = [
        c for c in cnts
        if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0
    ]
    if not candidates:
        candidates = cnts
    largest = max(candidates, key=cv2.contourArea)
    lawn_area = int(cv2.contourArea(largest))

    # 5. 实心蒙版 → Canny 单像素边缘 → 膨胀加粗
    lawn_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(lawn_mask, [largest], -1, 255, -1)
    edge = cv2.Canny(lawn_mask, 10, 50)
    thick_edge = cv2.dilate(
        edge, np.ones((3, 3), np.uint8), iterations=line_w // 2
    )

    # 6. 叠加绿色边界
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result[thick_edge > 0] = (0, 255, 0)

    info = {
        'otsu_val': otsu_val,
        'adjusted_th': adjusted_th,
        'lawn_area': lawn_area,
        'lawn_pct': lawn_area / (h * w) * 100,
    }
    return result, info


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    in_path, out_path = Path(in_dir), Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted([
        p for p in in_path.glob("*.png") if not p.stem.startswith("._")
    ])

    print(f"输入: {in_path.resolve()}")
    print(f"输出: {out_path.resolve()}")
    print(f"OTSU 偏移: +{THRESHOLD_OFFSET}  线宽: {LINE_THICKNESS}px")
    print(f"共 {len(files)} 张\n")

    for i, f in enumerate(files, 1):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [{i}/{len(files)}] ⚠️  {f.name}: 跳过")
            continue

        result, info = detect_lawn(gray)
        cv2.imwrite(str(out_path / f.name), result)

        if i == 1 or i % 10 == 0 or i == len(files):
            print(
                f"  [{i}/{len(files)}] {f.name}  "
                f"OTSU={info['otsu_val']} -> {info['adjusted_th']}  "
                f"菌苔={info['lawn_pct']:.0f}%"
            )

    print(f"\n完成: {len(files)} 张")


if __name__ == "__main__":
    main()
