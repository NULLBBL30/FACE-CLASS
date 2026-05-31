# utils.py
import time
import threading

class RateLimiter:
    """简单的线程安全限流器，用于控制 API QPS"""
    def __init__(self, max_qps):
        self.interval = 1.0 / max_qps
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

# 百度免费人脸识别 QPS 限制为 2
baidu_api_limiter = RateLimiter(max_qps=1.5)