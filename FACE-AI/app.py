# app.py
from flask import Flask, request, jsonify, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler
import os, time, logging, threading, atexit
from config import *
from baidu_face import BaiduFaceAPI
from emotion_models import BaiduEmotionModel
from processor import ImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)


for folder in [CLASS_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)


class RateLimiter:
    def __init__(self, max_qps):
        self.interval = 1.0 / max_qps
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now - self.last_call < self.interval:
                time.sleep(self.interval - (now - self.last_call))
            self.last_call = time.time()


baidu_api_limiter = RateLimiter(max_qps=1.5)
bd_api = BaiduFaceAPI()
# 移除 bd_api.init_group()，不再需要维护百度云人脸组
emotion_model = BaiduEmotionModel()
processor = ImageProcessor(bd_api, emotion_model)
processor.api_limiter = baidu_api_limiter


def cleanup_old_files():
    now = time.time()
    for folder in [CLASS_FOLDER, RESULT_FOLDER]:
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path) and os.stat(path).st_mtime < (now - 86400):
                try:
                    os.remove(path)
                except:
                    pass


scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_files, trigger="interval", hours=2)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index1.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        ts = str(int(time.time()))
        c_path = os.path.join(CLASS_FOLDER, f"c_{ts}.jpg")
        r_path = os.path.join(RESULT_FOLDER, f"r_{ts}.jpg")

        file.save(c_path)
        students, summary = processor.process(c_path, r_path)

        return jsonify({
            "status": "success",
            "img": f"/result/r_{ts}.jpg",
            "students": students,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)