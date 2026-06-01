#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_ground_truth.py -- Baidu API auto face detection + interactive pose annotation

Usage:
  python auto_ground_truth.py detect           # Auto-detect faces via Baidu API, fill cx/cy/is_edge
  python auto_ground_truth.py label rater1     # Rater 1: interactive pose labeling
  python auto_ground_truth.py label rater2     # Rater 2: interactive pose labeling
  python auto_ground_truth.py resolve          # Resolve conflicts -> true_pose / true_final_status
  python auto_ground_truth.py status           # Show annotation progress

Compared to manual auto_annotator.py:
  - No mouse clicking needed; Baidu API auto-detects face positions
  - Shows detected face boxes as reference during pose labeling
  - Incremental save, supports pause/resume
"""
import cv2
import csv
import os
import sys
import time
import numpy as np

from baidu_face import BaiduFaceAPI
from utils import baidu_api_limiter

IMAGE_DIR = "class_images"
CSV_PATH = "classroom_ground_truth.csv"
CROP_SIZE = 150
PANEL_W = 500
YAW_THRESHOLD = 21  # yaw threshold for distraction detection

# ================= 核心修改区：对齐论文的4种状态 =================
POSE_LABELS = {
    "1": "Positive Interaction",
    "2": "Focused Listening",
    "3": "Low Mood",
    "4": "Distracted"
}
# =================================================================

FIELDNAMES = ["image_path", "true_cx", "true_cy", "is_edge",
              "rater1_pose", "rater2_pose", "true_pose", "true_final_status",
              "detected_yaw", "detected_pitch", "pred_pose_off", "pred_pose_on", "pred_final"]

def load_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        return [], FIELDNAMES
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or FIELDNAMES
        rows = list(reader)
    # Ensure all FIELDNAMES keys exist in every row
    for r in rows:
        for k in FIELDNAMES:
            if k not in r:
                r[k] = ""
    return rows, list(fieldnames)
def save_csv(rows, fieldnames=None):
    if not rows:
        return
    fn = fieldnames or FIELDNAMES
    all_keys = list(fn)
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    tmp = CSV_PATH + ".tmp"
    for attempt in range(3):
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=all_keys)
                w.writeheader()
                w.writerows(rows)
            import os as _os
            _os.replace(tmp, CSV_PATH)
            return
        except PermissionError:
            time.sleep(0.5)
    # Use a completely different file path
    import shutil, uuid
    alt_path = os.path.join(os.path.dirname(os.path.abspath(CSV_PATH)) if os.path.dirname(CSV_PATH) else os.getcwd(), 
                            "gt_" + str(uuid.uuid4().hex[:8]) + ".csv")
    with open(alt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print("  [saved to " + os.path.basename(alt_path) + " instead]")


def get_face_crop(img, cx, cy, bbox_w=0, bbox_h=0, size=CROP_SIZE):
    h, w = img.shape[:2]
    if bbox_w > 0 and bbox_h > 0:
        half = max(bbox_w, bbox_h) // 2 + 20
    else:
        half = size // 2
    x1, y1 = max(0, int(cx) - half), max(0, int(cy) - half)
    x2, y2 = min(w, int(cx) + half), min(h, int(cy) + half)
    crop = img[y1:y2, x1:x2].copy()
    target = max(size, bbox_w + 40, bbox_h + 40) if bbox_w > 0 else size
    if crop.shape[0] < target or crop.shape[1] < target:
        padded = np.zeros((target, target, 3), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


# Step 1: Auto-detect faces via Baidu API
# ============================================================
def auto_detect():
    print("=" * 55)
    print("  Step 1: Baidu API Auto Face Detection")
    print("=" * 55)

    rows, fieldnames = load_csv()

    detected_images = set()
    for row in rows:
        detected_images.add(row["image_path"])

    valid_exts = {".jpg", ".jpeg", ".png"}
    all_images = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            all_images.append(f)

    all_images.sort(key=lambda x: (not x.startswith("eval_"), x))

    new_images = [img for img in all_images
                  if f"{IMAGE_DIR}/{img}" not in detected_images]

    if not new_images:
        total_imgs = len(detected_images)
        print(f"All {total_imgs} images already detected. Total: {len(rows)} records.")
        return

    print(f"Found {len(all_images)} images, {len(new_images)} new to detect")
    print(f"Baidu API QPS ~1.5, estimated {len(new_images) * 0.8:.0f}s")
    print()

    bd = BaiduFaceAPI()
    new_count = 0

    for i, img_name in enumerate(new_images):
        img_path_full = f"{IMAGE_DIR}/{img_name}"
        print(f"[{i + 1}/{len(new_images)}] Detecting: {img_name} ... ", end="", flush=True)

        baidu_api_limiter.wait()

        img = cv2.imread(img_path_full)
        if img is None:
            print("FAIL - cannot read")
            continue
        ih, iw = img.shape[:2]

        faces = bd.detect_faces(img_path_full)

        if not faces:
            print("no faces found")
            continue

        valid_faces = [f for f in faces if f.get("face_probability", 0) >= 0.3]

        added = 0
        for face in valid_faces:
            loc = face.get("location", {})
            left = loc.get("left", 0)
            top = loc.get("top", 0)
            width = loc.get("width", 0)
            height = loc.get("height", 0)

            cx = int(left + width / 2)
            cy = int(top + height / 2)
            is_edge = 1 if (cx < iw / 3 or cx > 2 * iw / 3) else 0

            rows.append({
                "image_path": f"{IMAGE_DIR}/{img_name}",
                "true_cx": str(cx),
                "true_cy": str(cy),
                "is_edge": str(is_edge),
                "rater1_pose": "",
                "rater2_pose": "",
                "true_pose": "",
                "true_final_status": "",
                "detected_yaw": "",
                "detected_pitch": "",
                "pred_pose_off": "",
                "pred_pose_on": "",
                "pred_final": ""
            })
        if (i + 1) % 5 == 0:
            save_csv(rows, fieldnames)

    save_csv(rows, fieldnames)
    print(f"\nDone! Added {new_count} records. "
          f"Total: {len(rows)} records, {len(detected_images) + len(new_images)} images.")


# ============================================================
# Step 2: Interactive pose annotation
# ============================================================
def draw_face_info(img_display, rows, img_path, current_idx, rater_column=None):
    for i, row in enumerate(rows):
        if row["image_path"] != img_path:
            continue
        cx = int(row["true_cx"])
        cy = int(row["true_cy"])
        is_edge = row.get("is_edge", "0")
        if rater_column:
            pose = row.get(rater_column, "")
        else:
            pose = row.get("rater1_pose", "") or row.get("rater2_pose", "")

        if i == current_idx:
            color, thickness = (0, 255, 0), 3
        elif pose:
            color, thickness = (0, 255, 255), 1
        elif is_edge == "1":
            color, thickness = (0, 0, 255), 1
        else:
            color, thickness = (255, 150, 0), 1

        cv2.circle(img_display, (cx, cy), 14, color, thickness)
        label = str(i)
        if pose:
            # 取首字母缩写展示在小圆圈旁边 (如 P, F, L, D)
            label += f":{pose[0]}"
        cv2.putText(img_display, label, (cx + 16, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def annotate_poses(rater="rater1_pose"):
    rows, fieldnames = load_csv()

    if not rows:
        print("CSV is empty. Run 'detect' first.")
        return

    rater_label = "Rater 1" if rater == "rater1_pose" else "Rater 2"
    done = sum(1 for r in rows if r.get(rater, "") != "")
    print(f"\n=== {rater_label} Pose Annotation ===")
    print(f"Total: {len(rows)}, Done: {done}, Remaining: {len(rows) - done}")
    print(f"Keys: 1=Positive Interaction  2=Focused Listening  3=Low Mood  4=Distracted")
    print(f"      N=Next  P=Prev  D=Clear  S=Skip  Q=Save+Quit")
    print(f"Colors: green=current, red=edge(unlabeled), "
          f"orange=center(unlabeled), yellow=labeled")

    cv2.namedWindow("Auto Ground Truth", cv2.WINDOW_NORMAL)
    idx = 0
    while idx < len(rows):
        row = rows[idx]
        img_path_full = row["image_path"]
        if not os.path.exists(img_path_full):
            img_path_full = os.path.join(IMAGE_DIR,
                                         os.path.basename(row["image_path"]))

        img = cv2.imread(img_path_full)
        if img is None:
            idx += 1
            continue

        ih, iw = img.shape[:2]
        cx = int(row["true_cx"])
        cy = int(row["true_cy"])
        is_edge = row.get("is_edge", "0")

        display = img.copy()
        draw_face_info(display, rows, row["image_path"], idx, rater)

        crop = get_face_crop(img, cx, cy, size=CROP_SIZE)
        crop_big = cv2.resize(crop, (PANEL_W, int(CROP_SIZE * 2 * PANEL_W / (CROP_SIZE * 2))))
        ch, cw = crop_big.shape[:2]
        cv2.line(crop_big, (cw // 2, 0), (cw // 2, ch), (0, 255, 255), 1)
        cv2.line(crop_big, (0, ch // 2), (cw, ch // 2), (0, 255, 255), 1)

        # Build info panel - hide other rater's labels for independence
        if rater == "rater1_pose":
            r1_display = f"Rater1: [{row.get('rater1_pose', '')}]"
            r2_display = "Rater2: [hidden]"
        else:
            r1_display = "Rater1: [hidden]"
            r2_display = f"Rater2: [{row.get('rater2_pose', '')}]"

        # 调整了面板高度以适应新增的选项
        info_lines = [
            f"Image: {os.path.basename(row['image_path'])}",
            f"Row: {idx + 1}/{len(rows)}  Pos: ({cx}, {cy})",
            f"Edge: {'Yes' if is_edge == '1' else 'No'}",
            r1_display,
            r2_display,
            "--- Current ---",
            f"{rater_label}: [{row.get(rater, '')}]",
            "",
            "1: Positive Interaction   2: Focused Listening",
            "3: Low Mood               4: Distracted",
            f"S: Skip  D: Clear  N/P: Prev/Next  Q: Save+Quit",
        ]

        # 增加黑色底板的高度从 240 到 280
        panel = np.zeros((280, 500, 3), dtype=np.uint8)
        for j, line in enumerate(info_lines):
            color = (0, 255, 0) if j == 6 else (255, 255, 255)
            cv2.putText(panel, line, (10, 22 + j * 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        right = np.vstack([crop_big, panel])
        if right.shape[0] < display.shape[0]:
            pad = np.zeros((display.shape[0] - right.shape[0], 500, 3),
                           dtype=np.uint8)
            right = np.vstack([right, pad])
        elif right.shape[0] > display.shape[0]:
            display = cv2.resize(
                display, (int(iw * right.shape[0] / ih), right.shape[0]))

        combined = np.hstack([display, right])
        cv2.imshow("Auto Ground Truth", combined)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)
        elif key == ord("d"):
            rows[idx][rater] = ""
            print(f"  [{idx + 1}] cleared")
        # 增加对按键 4 的监听
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            label = POSE_LABELS[chr(key)]
            rows[idx][rater] = label
            done = sum(1 for r in rows if r[rater] != "")
            print(f"  [{idx + 1}] {rater_label} = {label}  "
                  f"({done}/{len(rows)})")
            idx += 1
        elif key == ord("s"):
            idx += 1

    cv2.destroyAllWindows()
    save_csv(rows, fieldnames)
    done = sum(1 for r in rows if r[rater] != "")
    print(f"\nSaved. {rater_label}: {done}/{len(rows)} done.")


# ============================================================
# Step 3: Resolve conflicts -> true_pose / true_final_status
# ============================================================
def resolve_true_labels():
    rows, fieldnames = load_csv()
    if not rows:
        print("CSV is empty.")
        return

    conflicts = []
    agreed = 0
    for i, row in enumerate(rows):
        r1 = row.get("rater1_pose", "")
        r2 = row.get("rater2_pose", "")
        if r1 and r2:
            if r1 == r2:
                rows[i]["true_pose"] = r1
                agreed += 1
            else:
                conflicts.append(i)

    print(f"\n=== Conflict Resolution ===")
    print(f"Agreed: {agreed}, Conflicts: {len(conflicts)}")

    if conflicts:
        print(f"Keys: 1=Accept Rater1  2=Accept Rater2  Q=Skip")
        cv2.namedWindow("Conflict Resolver", cv2.WINDOW_NORMAL)
        for cidx, idx in enumerate(conflicts):
            row = rows[idx]
            img_path = os.path.join(IMAGE_DIR,
                                    os.path.basename(row["image_path"]))
            img = cv2.imread(img_path)
            if img is None:
                continue

            cx = int(row["true_cx"])
            cy = int(row["true_cy"])

            crop = get_face_crop(img, cx, cy, size=CROP_SIZE)
            crop_big = cv2.resize(crop, (CROP_SIZE * 3, CROP_SIZE * 3))

            info = np.zeros((120, CROP_SIZE * 3, 3), dtype=np.uint8)
            cv2.putText(
                info,
                f"Row {idx + 1}: R1=[{row['rater1_pose']}] vs "
                f"R2=[{row['rater2_pose']}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(
                info,
                f"[{cidx + 1}/{len(conflicts)}]  1:Rater1  2:Rater2  Q:Skip",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info, f"Edge: {row.get('is_edge', '?')}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 200, 0), 1)

            display = np.vstack([crop_big, info])
            cv2.imshow("Conflict Resolver", display)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("1"):
                    rows[idx]["true_pose"] = row["rater1_pose"]
                    break
                elif key == ord("2"):
                    rows[idx]["true_pose"] = row["rater2_pose"]
                    break
                elif key == ord("q"):
                    break
        cv2.destroyAllWindows()

    for i, row in enumerate(rows):
        if row.get("true_pose", ""):
            rows[i]["true_final_status"] = row["true_pose"]

    save_csv(rows, fieldnames)
    final_done = sum(1 for r in rows if r.get("true_pose", "") != "")
    print(f"\nDone! true_pose filled: {final_done}/{len(rows)}. Saved.")


# ============================================================
# Status viewer
# ============================================================
def show_status():
    rows, _ = load_csv()
    unique_images = set(r['image_path'] for r in rows)
    print("")
    print("=" * 45)
    print("  Ground Truth Status")
    print("=" * 45)
    print(f"  Images:       {len(unique_images)}")
    print(f"  Face records: {len(rows)}")
    for col, label in [("rater1_pose", "Rater1 Pose   "),
                        ("rater2_pose", "Rater2 Pose   "),
                        ("true_pose", "Final Pose    "),
                        ("true_final_status", "Final Status  ")]:
        done = sum(1 for r in rows if r.get(col, "") != "")
        bar_max = max(len(rows), 1)
        bar_len = done * 20 // bar_max
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {label}: {done:3d}/{len(rows):3d}  [{bar}]")
    print("=" * 45)
# Step 4: Compare predictions to ground truth
# ============================================================
def run_prediction():
    """
    用百度 API 预测每条人脸记录的姿态/状态，填入 pred_* 列
    方便在 CSV 中和人工标注逐条对比，定位问题
    """
    from config import CAMERA_FOV_ANGLE_HALF

    rows, fieldnames = load_csv()
    if not rows:
        print("CSV is empty. Run 'detect' first.")
        return

    bd = BaiduFaceAPI()

    # 按图片分组，每张图只调一次 API
    img_groups = {}
    for i, row in enumerate(rows):
        img_path = row["image_path"]
        if img_path not in img_groups:
            img_groups[img_path] = []
        img_groups[img_path].append(i)

    total = len(img_groups)
    print(f"\nPredicting {len(rows)} faces across {total} images...")

    for img_idx, (img_path, row_indices) in enumerate(img_groups.items()):
        print(f"  [{img_idx+1}/{total}] {os.path.basename(img_path)} ... ", end="", flush=True)

        # Check if image exists
        if not os.path.exists(img_path):
            alt_path = os.path.join(IMAGE_DIR, os.path.basename(img_path))
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                print("image not found, keep existing data")
                continue

        baidu_api_limiter.wait()
        faces = bd.detect_faces(img_path)

        img = cv2.imread(img_path)
        if img is None:
            print("skip (cannot read)")
            continue
        ih, iw = img.shape[:2]

        matched = 0
        for idx in row_indices:
            row = rows[idx]
            true_cx = int(row["true_cx"])
            true_cy = int(row["true_cy"])

            matched_face = None
            for f in faces:
                loc = f.get("location", {})
                left = loc.get("left", 0)
                top = loc.get("top", 0)
                w = loc.get("width", 0)
                h = loc.get("height", 0)
                if left <= true_cx <= left + w and top <= true_cy <= top + h:
                    matched_face = f
                    break

            if not matched_face:
                rows[idx]["pred_pose_off"] = "NO_FACE"
                rows[idx]["pred_pose_on"] = "NO_FACE"
                rows[idx]["pred_final"] = "NO_FACE"
                continue

            matched += 1
            angle = matched_face.get("angle", {})
            pitch = angle.get("pitch", 0)
            yaw = angle.get("yaw", 0)

            rows[idx]["detected_yaw"] = str(round(yaw, 1))
            rows[idx]["detected_pitch"] = str(round(pitch, 1))

            loc = matched_face["location"]
            left = loc.get("left", 0)
            top = loc.get("top", 0)
            w = loc.get("width", 0)
            h = loc.get("height", 0)

            # AYC OFF
            if pitch < -25:
                pose_off = "Head Down"
            elif abs(yaw) > YAW_THRESHOLD:
                pose_off = "Distracted"
            else:
                pose_off = "Attentive"

            # AYC ON
            cx = left + w / 2
            expected_yaw = ((cx / iw) - 0.5) * 2 * CAMERA_FOV_ANGLE_HALF
            yaw_corrected = yaw - expected_yaw

            if pitch < -25:
                pose_on = "Head Down"
            elif abs(yaw_corrected) > YAW_THRESHOLD:
                pose_on = "Distracted"
            else:
                pose_on = "Attentive"

            # Emotion (from Baidu API)
            from emotion_models import BaiduEmotionModel
            em = BaiduEmotionModel()
            face_crop = img[max(0, int(top)):int(top+h), max(0, int(left)):int(left+w)]
            emotion_cn, _ = em.analyze(face_crop, baidu_face_data=matched_face)

            # Final status
            if pose_on == "Attentive":
                if emotion_cn == "Positive":
                    pred_final = "Actively Interacting"
                elif emotion_cn == "Negative":
                    pred_final = "Low Spirits"
                else:
                    pred_final = "Attentive Listening"
            else:
                pred_final = pose_on

            rows[idx]["pred_pose_off"] = pose_off
            rows[idx]["pred_pose_on"] = pose_on
            rows[idx]["pred_final"] = pred_final

        print(f"{matched}/{len(row_indices)} matched")

        if (img_idx + 1) % 5 == 0:
            save_csv(rows, fieldnames)

    save_csv(rows, fieldnames)

    # Summary
    print(f"\n{'='*55}")
    print(f"  Prediction Summary (vs ground truth)")
    print(f"{'='*55}")

    has_gt = sum(1 for r in rows if r.get("true_pose", ""))
    has_pred = sum(1 for r in rows if r.get("pred_pose_on", ""))
    matched_cnt = sum(1 for r in rows if r.get("pred_pose_on", "") not in ("", "NO_FACE"))
    agree = sum(1 for r in rows if r.get("true_pose") and r.get("pred_pose_on")
                and r["true_pose"] == r["pred_pose_on"])
    disagree = sum(1 for r in rows if r.get("true_pose") and r.get("pred_pose_on")
                   and r["true_pose"] != r["pred_pose_on"])

    if has_gt and matched_cnt:
        print(f"  有标注: {has_gt}  有预测: {has_pred}  匹配成功: {matched_cnt}")
        print(f"  一致: {agree}  不一致: {disagree}  一致率: {agree/matched_cnt:.1%}")
        print()
        print(f"  Head Down 预测总数: {sum(1 for r in rows if r.get('pred_pose_on') == 'Head Down')}")
        print(f"  Distracted 预测总数: {sum(1 for r in rows if r.get('pred_pose_on') == 'Distracted')}")
        print(f"  Attentive 预测总数: {sum(1 for r in rows if r.get('pred_pose_on') == 'Attentive')}")
    else:
        print(f"  预测完成，等待人工标注后做对比。")

    print(f"\n  CSV saved with pred_* columns. Open in Excel to browse.")


def review_comparison():
    """
    可视化对比模式：逐条展示人脸 + 人工标注 vs 百度预测结果
    """
    rows, fieldnames = load_csv()
    if not rows:
        print("CSV is empty.")
        return

    has_pred = any(r.get("pred_pose_on", "") for r in rows)
    if not has_pred:
        print("No predictions found. Run 'predict' first.")
        return

    cv2.namedWindow("Prediction Review", cv2.WINDOW_NORMAL)
    idx = 0
    while idx < len(rows):
        row = rows[idx]
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, os.path.basename(row["image_path"]))

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        ih, iw = img.shape[:2]
        cx = int(row["true_cx"])
        cy = int(row["true_cy"])

        display = img.copy()
        cv2.circle(display, (cx, cy), 14, (0, 255, 0), 2)
        cv2.putText(display, f"#{idx}", (cx + 16, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        crop = get_face_crop(img, cx, cy, size=CROP_SIZE)
        crop_big = cv2.resize(crop, (CROP_SIZE * 3, CROP_SIZE * 3))

        gt = row.get("true_pose", "-") or "-"
        r1 = row.get("rater1_pose", "-") or "-"
        r2 = row.get("rater2_pose", "-") or "-"
        p_off = row.get("pred_pose_off", "-") or "-"
        p_on = row.get("pred_pose_on", "-") or "-"
        p_final = row.get("pred_final", "-") or "-"

        agree = "Y" if (gt != "-" and gt == p_on) else "N" if gt != "-" else "?"

        info_lines = [
            f"{os.path.basename(row['image_path'])}  Row {idx+1}/{len(rows)}",
            f"Coord: ({cx}, {cy})  Edge: {row.get('is_edge','?')}",
            "",
            f"Rater1: [{r1}]  Rater2: [{r2}]",
            f"True:   [{gt}]",
            "",
            f"Pred (AYC OFF): [{p_off}]",
            f"Pred (AYC ON):  [{p_on}]",
            f"Pred Final:     [{p_final}]",
            "",
            f"Agree? {agree}",
            f"N: Next  P: Prev  Q: Quit",
        ]
        panel = np.zeros((280, CROP_SIZE * 3, 3), dtype=np.uint8)
        for j, line in enumerate(info_lines):
            color = (255, 255, 255)
            if "True:" in line:
                color = (0, 255, 0)
            elif "Agree" in line:
                color = (0, 255, 0) if agree == "Y" else (0, 0, 255) if agree == "N" else (200, 200, 0)
            elif "AYC ON" in line:
                color = (255, 200, 0)
            cv2.putText(panel, line, (10, 22 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        display_resized = cv2.resize(display, (int(iw * (CROP_SIZE * 3) / ih), CROP_SIZE * 3))
        combined = np.hstack([display_resized, np.vstack([crop_big, panel])])
        cv2.imshow("Prediction Review", combined)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()

def run_evaluation():
    """
    基于 CSV 中的标注和预测数据，输出完整评估报告：
      - Cohen's Kappa（标注者间信度）
      - AYC 边缘 FPR 对比
      - 四状态分类报告 + 混淆矩阵
    """
    try:
        from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("缺少依赖: pip install scikit-learn matplotlib seaborn")
        return

    rows, _ = load_csv()
    if not rows:
        print("CSV 为空，请先运行 detect。")
        return

    # 检查是否有预测数据
    has_pred = any(r.get("pred_pose_on", "") for r in rows)
    if not has_pred:
        print("尚未运行预测，先自动执行 predict ...")
        run_prediction()
        rows, _ = load_csv()

    import pandas as pd
    df = pd.DataFrame(rows)

    # 过滤有效行
    df_valid = df[df["pred_pose_on"].notna() & (df["pred_pose_on"] != "") & (df["pred_pose_on"] != "NO_FACE")].copy()

    if len(df_valid) == 0:
        print("没有有效的预测数据。")
        return

    print("\n" + "=" * 55)
    print("  定量评估报告")
    print("=" * 55)

    # --- 1) Cohen's Kappa ---
    has_r1 = df_valid["rater1_pose"].notna() & (df_valid["rater1_pose"] != "")
    has_r2 = df_valid["rater2_pose"].notna() & (df_valid["rater2_pose"] != "")
    both_rated = has_r1 & has_r2

    if both_rated.sum() > 0:
        kappa = cohen_kappa_score(
            df_valid.loc[both_rated, "rater1_pose"],
            df_valid.loc[both_rated, "rater2_pose"]
        )
        print(f"\n1. 评分者间信度 (Cohen's Kappa)")
        print(f"   样本数: {both_rated.sum()}")
        print(f"   Kappa = {kappa:.3f}  (通常 >0.75 高度一致, >0.40 中等一致)")
    else:
        print("\n1. 评分者间信度: 缺少双评数据（需先执行 label rater1 + label rater2）")

    # --- 2) AYC 消融实验 ---
    # --- 2) AYC ablation study ---
    edge_cases = df_valid[df_valid["is_edge"].astype(str) == "1"]
    has_gt = df_valid["true_pose"].notna() & (df_valid["true_pose"] != "")

    if len(edge_cases) > 0 and has_gt.any():
        edge_with_gt = edge_cases[edge_cases["true_pose"].notna()].copy()
        if len(edge_with_gt) > 0:
            total_edge = len(edge_with_gt)
            truly_distracted = edge_with_gt[edge_with_gt["true_pose"] == "Distracted"]
            n_true_dist = len(truly_distracted)
            truly_attentive = edge_with_gt[edge_with_gt["true_pose"].isin(["Actively Interacting", "Attentive Listening"])]
            n_true_att = len(truly_attentive)

            tp_off = len(truly_distracted[truly_distracted["pred_pose_off"] == "Distracted"])
            fp_off = len(truly_attentive[truly_attentive["pred_pose_off"] == "Distracted"])
            rec_off = tp_off / n_true_dist if n_true_dist > 0 else 0
            fpr_off = fp_off / n_true_att if n_true_att > 0 else 0

            tp_on = len(truly_distracted[truly_distracted["pred_pose_on"] == "Distracted"])
            fp_on = len(truly_attentive[truly_attentive["pred_pose_on"] == "Distracted"])
            rec_on = tp_on / n_true_dist if n_true_dist > 0 else 0
            fpr_on = fp_on / n_true_att if n_true_att > 0 else 0

            acc_off = (edge_with_gt["pred_pose_off"] == edge_with_gt["true_pose"]).sum() / total_edge
            acc_on = (edge_with_gt["pred_pose_on"] == edge_with_gt["true_pose"]).sum() / total_edge

            print()
            print("2. AYC Edge Performance Comparison")
            print(f"   {'Metric':35s} {'Raw Pose':15s} {'AYC-Corr':15s} {'Change':15s}")
            print(f"   {'-'*80}")
            print(f"   {'Edge samples':35s} {total_edge:>10d}")
            print(f"   {'Distracted detected (TP)':35s} {tp_off:>10d} {tp_on:>10d}")
            print(f"   {'Distraction Recall':35s} {rec_off:>10.1%} {rec_on:>10.1%} {rec_on-rec_off:>+7.1%}")
            print(f"   {'False Pos Rate (attentive)':35s} {fpr_off:>10.1%} {fpr_on:>10.1%} {fpr_on-fpr_off:>+7.1%}")
            print(f"   {'Edge Accuracy':35s} {acc_off:>10.1%} {acc_on:>10.1%} {acc_on-acc_off:>+7.1%}")
        else:
            print()
            print("2. AYC: edge samples lack ground truth")
    else:
        print()
        print("2. AYC: no edge samples or ground truth data")

    # --- 3) 四状态分类报告 ---
    has_final = has_gt

    # 标签归一化：将同义标签统一为4类标准名
    LABEL_MAP = {
        'Positive Interaction': 'Actively Interacting',
        'Focused Listening': 'Attentive Listening',
        'Low Mood': 'Low Spirits',
    }
    df_valid['true_pose'] = df_valid['true_pose'].replace(LABEL_MAP)
    df_valid['true_final_status'] = df_valid['true_final_status'].replace(LABEL_MAP)
    df_valid['pred_pose_off'] = df_valid['pred_pose_off'].replace(LABEL_MAP)
    df_valid['pred_pose_on'] = df_valid['pred_pose_on'].replace(LABEL_MAP)
    df_valid['pred_final'] = df_valid['pred_final'].replace(LABEL_MAP)
    if has_final.any():
        labels = ["Actively Interacting", "Attentive Listening", "Low Spirits", "Distracted"]
        y_true = df_valid.loc[has_final, "true_final_status"].replace("Head Down", "Distracted")
        y_pred = df_valid.loc[has_final, "pred_final"].replace("Head Down", "Distracted")

        if len(y_true) > 0:
            print(f"\n3. 四状态融合分类报告 (true_final_status vs pred_final)")
            print(f"   样本数: {len(y_true)}")
            print()
            print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels)
            plt.title("Confusion Matrix - Classroom Engagement Status")
            plt.ylabel("True Status")
            plt.xlabel("Predicted Status")
            plt.tight_layout()
            plt.savefig("confusion_matrix_results.png", dpi=300)
            plt.close()
            print("   混淆矩阵已保存: confusion_matrix_results.png")

            # 总体准确率
            acc = (y_true == y_pred).mean()
            print(f"\n   总体准确率 (Accuracy): {acc:.2%}")

            # 按表展示前10条不一致项
            mismatches = df_valid[has_final].copy()
            mismatches["y_true"] = mismatches["true_final_status"].replace("Head Down", "Distracted")
            mismatches["y_pred"] = mismatches["pred_final"].replace("Head Down", "Distracted")
            mismatches = mismatches[mismatches["y_true"] != mismatches["y_pred"]]
            if len(mismatches) > 0:
                print(f"   不一致项共 {len(mismatches)} 条")
        else:
            print("\n3. 四状态分类: 无有效数据")
    else:
        print("\n3. 四状态分类: 缺少标注 (需先执行 resolve)")

    print(f"\n{'=' * 55}")
    print("  评估完成。")
    print(f"{'=' * 55}")
# Entry point (交互式菜单版)
# ============================================================
if __name__ == "__main__":
    while True:
        print("\n" + "=" * 45)
        print("  自动标注系统 (Auto Ground Truth)")
        print("=" * 45)
        print("  1. 自动检测人脸位置 (detect)")
        print("  2. Rater 1 交互标注姿态 (label rater1)")
        print("  3. Rater 2 交互标注姿态 (label rater2)")
        print("  4. 冲突裁决 (resolve)")
        print("  5. 查看标注进度 (status)")
        print("  6. 百度 API 预测并填入CSV (predict)")
        print("  7. 跑评估: Kappa + AYC + 混淆矩阵 (evaluate)")
        print("  8. 可视化对比人工 vs 预测 (review)")
        print("  0. 退出程序 (exit)")
        print("=" * 45)

        choice = input("请输入选项数字并回车 (0-8): ").strip()

        if choice == "1":
            auto_detect()
        elif choice == "2":
            annotate_poses("rater1_pose")
        elif choice == "3":
            annotate_poses("rater2_pose")
        elif choice == "4":
            resolve_true_labels()
        elif choice == "5":
            show_status()
        elif choice == "6":
            run_prediction()
        elif choice == "7":
            run_evaluation()
        elif choice == "8":
            review_comparison()
        elif choice == "0" or choice.lower() == "q":
            print("已退出程序。")
            break
        else:
            print("无效的输入，请重新选择。")
