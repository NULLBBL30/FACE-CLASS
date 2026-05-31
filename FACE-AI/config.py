# config.py
import os
from dotenv import load_dotenv

load_dotenv() # 建议在同级目录建立 .env 文件写入真实密钥
# ================= 百度云配置 =================

API_KEY = os.getenv("")
SECRET_KEY = os.getenv("")
GROUP_ID = "class_students"

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ID_FOLDER = os.path.join(BASE_DIR, "id_images")
CLASS_FOLDER = os.path.join(BASE_DIR, "class_images")
RESULT_FOLDER = os.path.join(BASE_DIR, "result_images")

# ================= 业务逻辑参数优化 =================
FACE_THRESHOLD = 80                 # 人脸匹配分数阈值
FACE_PROB_THRESHOLD = 0.45          # 人脸检测置信度
MIN_FACE_RATIO = 0.0001             # 支持检测占画面0.1%的远端小人脸
MAX_FACE_RATIO = 0.5
FACE_ASPECT_RATIO_MIN = 0.5
FACE_ASPECT_RATIO_MAX = 2.0
FACE_MARGIN_RATIO = 0.3

# ================= 结构化评估统计阈值 =================
SUMMARY_ACTIVE_THRESHOLD = 0.6      # 判断"大多积极"的占比阈值 
SUMMARY_NEUTRAL_THRESHOLD = 0.7     # 定义后排"中性"占比阈值
SUMMARY_DIFF_THRESHOLD = 0.2       # 定义左右参与度差异大于 15% 视为有明显差异

# 空间位置与姿态分析阈值
POS_FRONT_AREA_RATIO = 0.012        # 前排面积占比阈值
POS_FRONT_TOP_RATIO = 0.5           # 前排高度(Y轴)占比阈值
CAMERA_FOV_ANGLE_HALF = 35.0        # 摄像头预估水平视角的一半（根据实际广角微调）