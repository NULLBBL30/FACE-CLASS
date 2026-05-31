# processor.py
import cv2
import numpy as np
import logging
import concurrent.futures
import time
from PIL import Image, ImageDraw, ImageFont
from config import *

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, face_api, emotion_model):
        self.face_api = face_api
        self.emotion_model = emotion_model

        try:
            self.font = ImageFont.truetype("simhei.ttf", 40)
        except IOError:
            logger.warning("simhei.ttf not found. Falling back to default font.")
            self.font = ImageFont.load_default()

    def _filter_face(self, f, img_w, img_h):
        if f.get("face_probability", 0) < FACE_PROB_THRESHOLD:
            return None, None

        loc = f["location"]
        left, top, w, h = int(loc["left"]), int(loc["top"]), int(loc["width"]), int(loc["height"])

        area = w * h
        if not ((img_w * img_h * MIN_FACE_RATIO) <= area <= (img_w * img_h * MAX_FACE_RATIO)):
            return None, None
        return (left, top, w, h), area

    def _process_single_face(self, f, img, iw, ih):
        bbox, area = self._filter_face(f, iw, ih)
        if not bbox: return None

        left, top, w, h = bbox
        margin_w, margin_h = int(w * FACE_MARGIN_RATIO), int(h * FACE_MARGIN_RATIO)
        crop_left, crop_top = max(0, left - margin_w), max(0, top - margin_h)
        face_crop_np = img[crop_top:top + h + margin_h, crop_left:left + w + margin_w]

        # 调用在 app.py 中注入的限流器
        if hasattr(self, 'api_limiter'):
            self.api_limiter.wait()
        else:
            time.sleep(0.6)

        # 核心修改：移除人脸搜索(身份识别)，仅进行情绪和姿态分析
        emotion_cn, _ = self.emotion_model.analyze(face_crop_np, baidu_face_data=f)

        angle = f.get("angle", {})
        pitch = angle.get("pitch", 0)
        detected_yaw = angle.get("yaw", 0)

        cx = left + w / 2
        cy = top + h / 2  # 提取人脸中心Y坐标用于后续全局动态聚类

        fov_angle_half = globals().get('CAMERA_FOV_ANGLE_HALF', 35.0)
        expected_yaw = ((cx / iw) - 0.5) * 2 * fov_angle_half
        yaw_corrected = detected_yaw - expected_yaw

        # ================= 核心修改区：严格对齐论文的 4 种状态 =================
        # 第一步：判定基础姿态 (Focused 还是 Distracted)
        if pitch < -25 or abs(yaw_corrected) > 30:
            pose_status = "Distracted"
        else:
            pose_status = "Focused"

        # 第二步：结合情绪模型映射最终状态
        if pose_status == "Focused":
            if emotion_cn == "Positive":
                final_status = "Positive Interaction"
                nature = "Positive"
            elif emotion_cn == "Negative":
                final_status = "Low Mood"
                nature = "Negative"
            else:
                final_status = "Focused Listening"
                nature = "Positive"
        else:
            final_status = "Distracted"
            nature = "Negative"
        # ======================================================================

        return {
            "bbox": bbox,
            "status": final_status,
            "nature": nature,
            "cx": cx,
            "cy": cy
        }

    def process(self, img_path, output_path):
        faces = self.face_api.detect_faces(img_path)
        img = cv2.imread(img_path)
        if img is None: return [], []

        ih, iw = img.shape[:2]
        students_data = []
        valid_results = []

        # 启动多线程并发执行API请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_results = [executor.submit(self._process_single_face, f, img, iw, ih) for f in faces]
            for future in concurrent.futures.as_completed(future_results):
                res = future.result()
                if res:
                    valid_results.append(res)

        # ================= 全局动态空间聚类 (九宫格) =================
        if valid_results:
            y_coords = [res["cy"] for res in valid_results]
            y_min, y_max = min(y_coords), max(y_coords)

            # 自适应判断：如果画面纵深极小(只有一排)，统一视为"中排"；否则使用动态分位数切分
            if y_max - y_min < ih * 0.15:
                y_33, y_66 = float('inf'), -float('inf')
            else:
                y_33 = np.percentile(y_coords, 33)
                y_66 = np.percentile(y_coords, 66)

            for res in valid_results:
                cx, cy = res["cx"], res["cy"]

                # 判定排数 (Y坐标越小越靠上，即越在后排)
                if cy <= y_33:
                    row = "Back Row"
                elif cy <= y_66:
                    row = "Mid Row"
                else:
                    row = "Front Row"

                # 判定左右列
                if cx < iw / 3:
                    col = "Left"
                elif cx < 2 * iw / 3:
                    col = "Center"
                else:
                    col = "Right"

                res["row"] = row
                res["col"] = col
                res["pos"] = f"{col} {row}"

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for res in valid_results:
            left, top, w, h = res["bbox"]
            status = res["status"]
            box_color = (0, 255, 0) if res["nature"] == "Positive" else (255, 0, 0)

            draw.rectangle([left, top, left + w, top + h], outline=box_color, width=2)

            text_content = f"{status}"  # 仅保留状态文本，去除名字

            try:
                text_w = int(draw.textlength(text_content, font=self.font))
            except AttributeError:
                text_w = draw.textsize(text_content, font=self.font)[0]

            text_x = left
            if text_x + text_w > iw:
                text_x = iw - text_w - 10

            text_y = max(0, top - 45)

            draw.text((text_x, text_y), text_content, font=self.font, fill=box_color)

            students_data.append({
                "status": status,
                "nature": res["nature"],
                "pos": res["pos"],
                "row": res["row"],
                "col": res["col"]
            })

        img_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_final)

        summary = self._generate_structured_summary(students_data)
        return students_data, summary

    @staticmethod
    def _generate_structured_summary(students):
        summary = []
        if not students: return ["Cannot generate assessment: No clear and valid face data detected."]

        def get_active_rate(group):
            if not group: return 0
            return len([s for s in group if s["nature"] == "Positive"]) / len(group)

        # 1. 整体大盘分析
        avg_rate = get_active_rate(students)
        if avg_rate > 0.7:
            summary.append(f"• Overall Status: Excellent class condition, the vast majority of students are highly attentive (Active Rate {avg_rate:.0%}).")
        elif avg_rate > 0.5:
            summary.append(f"• Overall Status: Good condition, over half the students ({avg_rate:.0%}) are attentive.")
        else:
            summary.append(f"• Overall Status: Low activity and attention ({avg_rate:.0%}), suggest adjusting the teaching pace to attract attention.")

        # 2. 纵深对比 (前/中/后排)
        rows = {"Front Row": [], "Mid Row": [], "Back Row": []}
        for s in students: rows[s.get("row", "Front Row")].append(s)

        row_rates = {k: get_active_rate(v) for k, v in rows.items() if v}
        if row_rates:
            worst_row = min(row_rates, key=row_rates.get)
            best_row = max(row_rates, key=row_rates.get)
            if row_rates[worst_row] < 0.5:
                summary.append(
                    f"• Row Distribution: [{worst_row}] shows more distracted behaviors (Attentive rate only {row_rates[worst_row]:.0%}), requiring more monitoring; whereas [{best_row}] performs the best.")

        # 3. 横向分布分析 (左/中/右)
        cols = {"Left": [], "Center": [], "Right": []}
        for s in students: cols[s.get("col", "Center")].append(s)

        col_rates = {k: get_active_rate(v) for k, v in cols.items() if v}
        if col_rates:
            best_col = max(col_rates, key=col_rates.get)
            worst_col = min(col_rates, key=col_rates.get)
            if col_rates[best_col] - col_rates[worst_col] > SUMMARY_DIFF_THRESHOLD:
                summary.append(
                    f"• Column Distribution: [{best_col}] shows significantly better attention. Recommend focusing more eye contact and questions toward [{worst_col}].")

        # 4. 九宫格细分盲区挖掘 (高阶分析)
        zones = {}
        for s in students:
            zones.setdefault(s["pos"], []).append(s)

        bad_zones = []
        for zone_name, zone_students in zones.items():
            if len(zone_students) >= 3 and get_active_rate(zone_students) <= 0.3:
                bad_zones.append(f"{zone_name} ({len(zone_students)} students)")

        if bad_zones:
            summary.append(
                f"• ⚠️ Blind Spot Warning: The algorithm detected clustered negative status in [{', '.join(bad_zones)}], which are the core interactive blind spots of this class.")

        return summary