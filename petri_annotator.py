#!/usr/bin/env python3
"""
Petri Lawn Annotator - 培养皿菌苔与纸片自动标注工具

流程：
  1. 读取灰度图 → 菌苔绿色边界标注
  2. 纸片红色圆形标注
  3. 输出合并结果

依赖: opencv-python, numpy
"""

import os, sys
import cv2
import numpy as np
from pathlib import Path


# ============================================================
#  参数
# ============================================================

# 菌苔检测
LAWN_THRESHOLD_OFFSET = 10     # OTSU 阈值上调量（越大越保守）
LAWN_LINE_WIDTH = 4            # 绿色边界线宽（px）

# 纸片检测
PAPER_MIN_RADIUS = 10          # 纸片最小半径（px）
PAPER_MAX_RADIUS = 80          # 纸片最大半径（px）
PAPER_LINE_WIDTH = 3           # 红色边界线宽（px）


# ============================================================
#  菌苔检测
# ============================================================

def detect_lawn(gray, offset=LAWN_THRESHOLD_OFFSET, line_w=LAWN_LINE_WIDTH):
    """
    检测菌苔边界，返回标注后的 BGR 图和信息。
    
    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (H, W)，黑底。
    offset : int
        OTSU 阈值上调量。
    line_w : int
        绿色边界线宽（px）。
    
    Returns
    -------
    result : np.ndarray
        叠加绿色边界后的 BGR 图 (H, W, 3)。
    info : dict
        {'otsu_val', 'adjusted_th', 'lawn_area', 'lawn_pct'}
    """
    h, w = gray.shape[:2]

    # 只分析培养皿内部（排除黑色背景）
    dish_mask = np.uint8(gray > 10) * 255
    dish_only = cv2.bitwise_and(gray, gray, mask=dish_mask)

    # OTSU 自适应阈值 + 偏移
    otsu_val, _ = cv2.threshold(dish_only, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = int(otsu_val)
    adjusted_th = min(otsu_val + offset, 255)
    _, bin_img = cv2.threshold(dish_only, adjusted_th, 255, cv2.THRESH_BINARY)

    # 形态学清理
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)

    # 取含图像中心的最大连通域
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

    # 实心蒙版 → Canny 单像素边缘 → 膨胀加粗
    lawn_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(lawn_mask, [largest], -1, 255, -1)
    edge = cv2.Canny(lawn_mask, 10, 50)
    thick_edge = cv2.dilate(
        edge, np.ones((3, 3), np.uint8), iterations=line_w // 2
    )

    # 叠加绿色边界
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result[thick_edge > 0] = (0, 255, 0)

    info = {
        'otsu_val': otsu_val,
        'adjusted_th': adjusted_th,
        'lawn_area': lawn_area,
        'lawn_pct': round(lawn_area / (h * w) * 100, 1),
    }
    return result, info


# ============================================================
#  纸片检测
# ============================================================

def detect_paper(gray):
    """
    用 HoughCircles 检测白色纸片圆形。
    
    Parameters
    ----------
    gray : np.ndarray
        输入灰度图 (H, W)。
    
    Returns
    -------
    (cx, cy, r) : tuple
        圆心坐标和半径，检测失败返回 None。
    """
    h, w = gray.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    # 在中心区域搜索纸片圆
    margin = 120
    x1, y1 = max(0, img_cx - margin), max(0, img_cy - margin)
    x2, y2 = min(w, img_cx + margin), min(h, img_cy + margin)
    roi = gray[y1:y2, x1:x2]

    for dp in [1.0, 1.2, 1.5]:
        for p2 in [15, 20, 25, 30]:
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=dp,
                minDist=30, param1=80, param2=p2,
                minRadius=PAPER_MIN_RADIUS, maxRadius=PAPER_MAX_RADIUS,
            )
            if circles is not None:
                c = np.round(circles[0][0]).astype(int)
                rx, ry, rr = c[0] + x1, c[1] + y1, c[2]

                # 验证：纸片与周围环带的亮度差应显著
                outer_r = min(int(rr * 1.5), min(w, h) // 2)
                inner_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(inner_mask, (rx, ry), max(rr, 1), 255, -1)
                outer_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(outer_mask, (rx, ry), outer_r, 255, -1)
                ring = cv2.bitwise_xor(outer_mask, inner_mask)
                inner_b = cv2.mean(gray, mask=inner_mask)[0]
                outer_b = cv2.mean(gray, mask=ring)[0]

                if abs(inner_b - outer_b) > 15 and inner_b > 50:
                    return rx, ry, rr

    return None


def draw_paper_circle(img, cx, cy, r, line_w=PAPER_LINE_WIDTH):
    """在 BGR 图上绘制红色纸片圆圈。"""
    pts = []
    for deg in range(360):
        rad = np.deg2rad(deg)
        px = int(round(cx + r * np.cos(rad)))
        py = int(round(cy + r * np.sin(rad)))
        pts.append([[px, py]])
    contour = np.array(pts, dtype=np.int32)
    cv2.drawContours(img, [contour], -1, (0, 0, 255), line_w)
    return img


# ============================================================
#  完整流水线
# ============================================================

def process_image(gray, lawn_offset=LAWN_THRESHOLD_OFFSET):
    """
    对单张灰度图执行完整流水线：菌苔标注 + 纸片标注。
    
    Parameters
    ----------
    gray : np.ndarray
        输入灰度图。
    lawn_offset : int
        OTSU 阈值上调量。
    
    Returns
    -------
    result : np.ndarray
        标注后的 BGR 图。
    info : dict
        检测信息。
    """
    # Step 1: 菌苔绿色标注
    result, info = detect_lawn(gray, offset=lawn_offset)

    # Step 2: 纸片红色标注
    paper = detect_paper(gray)
    if paper is not None:
        cx, cy, r = paper
        result = draw_paper_circle(result, cx, cy, r)
        info['paper_detected'] = True
        info['paper_cx'] = cx
        info['paper_cy'] = cy
        info['paper_r'] = r
    else:
        info['paper_detected'] = False

    return result, info


# ============================================================
#  CLI
# ============================================================

def main():
    default_input = (
        "/Volumes/科研 2/mcr-1 影响菌毛系统的表达/游泳运动"
        "/20250511 加药后游泳运动/result_bg_black_gray"
    )
    default_output = os.path.normpath(
        os.path.join(default_input, "..", "result_combined")
    )

    in_dir = sys.argv[1] if len(sys.argv) > 1 else default_input
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else LAWN_THRESHOLD_OFFSET

    in_path, out_path = Path(in_dir), Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted([
        p for p in in_path.glob("*.png") if not p.stem.startswith("._")
    ])
    print(f"输入: {in_path.resolve()}")
    print(f"输出: {out_path.resolve()}")
    print(f"参数: OTSU偏移=+{offset}  菌苔线宽={LAWN_LINE_WIDTH}px  纸片线宽={PAPER_LINE_WIDTH}px")
    print(f"共 {len(files)} 张\n")

    for i, f in enumerate(files, 1):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [{i}/{len(files)}] ⚠️  {f.name}: 跳过")
            continue

        result, info = process_image(gray, lawn_offset=offset)
        cv2.imwrite(str(out_path / f.name), result)

        flag_paper = "✅纸片" if info.get('paper_detected') else "⚠️无纸片"
        if i == 1 or i % 10 == 0 or i == len(files):
            print(
                f"  [{i}/{len(files)}] {f.name}  "
                f"菌苔={info['lawn_pct']}%({info['lawn_area']}px²)  "
                f"{flag_paper}"
            )

    print(f"\n完成: {len(files)} 张")


if __name__ == "__main__":
    main()
