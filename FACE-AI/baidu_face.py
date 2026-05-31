# baidu_face.py
import requests
import base64
import cv2
import logging
from config import API_KEY, SECRET_KEY, GROUP_ID, FACE_THRESHOLD

logger = logging.getLogger(__name__)

class BaiduFaceAPI:
    def __init__(self):
        self._token = None
        self._get_token()

    def _get_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        data = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        try:
            res = requests.post(url, data=data, timeout=10).json()
            if "access_token" in res:
                self._token = res["access_token"]
                logger.info("百度云Token刷新成功")
            else:
                logger.error(f"Token获取失败，请检查密钥: {res}")
        except Exception as e:
            logger.error(f"Token请求异常: {e}")
            raise

    def _request(self, endpoint, json_data, retry=True):
        url = f"https://aip.baidubce.com{endpoint}?access_token={self._token}"
        try:
            res = requests.post(url, json=json_data, timeout=10).json()
            if res.get("error_code") in [110, 111] and retry:
                logger.warning("Token失效，尝试刷新...")
                self._get_token()
                return self._request(endpoint, json_data, retry=False)
            return res
        except Exception as e:
            return {"error_code": -1, "error_msg": str(e)}

    @staticmethod
    def np_to_b64(img_np):
        _, buffer = cv2.imencode('.jpg', img_np)
        return base64.b64encode(buffer).decode()

    @staticmethod
    def file_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def init_group(self):
        return self._request("/rest/2.0/face/v3/faceset/group/add", {"group_id": GROUP_ID})

    def add_user(self, stu_id, name_cn, image_path):
        data = {
            "image": self.file_to_b64(image_path),
            "image_type": "BASE64",
            "group_id": GROUP_ID,
            "user_id": stu_id,
            "user_info": name_cn, # 注册时将姓名写入云端
            "quality_control": "NONE"
        }
        return self._request("/rest/2.0/face/v3/faceset/user/add", data)

    def search_user(self, img_np):
        data = {
            "image": self.np_to_b64(img_np),
            "image_type": "BASE64",
            "group_id_list": GROUP_ID,
            "user_top_num": 1,
            "quality_control": "NORMAL"
        }
        res = self._request("/rest/2.0/face/v3/search", data)
        if res.get("error_code") == 0:
            user = res["result"]["user_list"][0]
            display_name = user.get("user_info") or user["user_id"]
            return display_name if user["score"] >= FACE_THRESHOLD else "未知"
        return "未知"

    def detect_faces(self, image_path):
        data = {
            "image": self.file_to_b64(image_path),
            "image_type": "BASE64",
            "face_field": "emotion,location,face_probability,angle",
            "max_face_num": 120,
            "min_face_size": 15
        }
        res = self._request("/rest/2.0/face/v3/detect", data)
        return res.get("result", {}).get("face_list", [])

    def get_users(self):
        """仅获取云端所有学号"""
        res = self._request("/rest/2.0/face/v3/faceset/group/getusers", {"group_id": GROUP_ID})
        if res.get("error_code") == 0:
            return res.get("result", {}).get("user_id_list", [])
        return []

    def get_user_detail(self, stu_id):
        """新增：获取云端用户的详细信息（包含姓名）"""
        res = self._request("/rest/2.0/face/v3/faceset/user/get", {
            "group_id": GROUP_ID,
            "user_id": stu_id
        })
        if res.get("error_code") == 0:
            user_list = res.get("result", {}).get("user_list", [])
            return user_list[0].get("user_info", "未知") if user_list else "未知"
        return "未知"

    def delete_user(self, stu_id):
        return self._request("/rest/2.0/face/v3/faceset/user/delete", {"group_id": GROUP_ID, "user_id": stu_id})

    def update_user(self, stu_id, new_name):
        data = {
            "group_id": GROUP_ID,
            "user_id": stu_id,
            "user_info": new_name
        }
        return self._request("/rest/2.0/face/v3/faceset/user/update", data)