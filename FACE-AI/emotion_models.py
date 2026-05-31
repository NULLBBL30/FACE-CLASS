# emotion_models.py (策略层架构保持不变)
from abc import ABC, abstractmethod

class BaseEmotionModel(ABC):

    name = "Base"
    @abstractmethod
    def analyze(self, img_np, baidu_face_data=None):
        pass

class BaiduEmotionModel(BaseEmotionModel):
    """
    基于百度 API 检测结果进行积极与中性情绪映射的具体策略实现
    """
    name = "Baidu"

    @staticmethod
    def _map_baidu_emotion(baidu_type):

        mapping = {
            "happy": ("Positive", "Positive"),
            "laugh": ("Positive", "Positive"),
            "surprise": ("Positive", "Positive"),

            "neutral": ("Neutral", "Neutral"),
            "calm": ("Neutral", "Neutral"),

            "angry": ("Negative", "Negative"),
            "sad": ("Negative", "Negative"),
            "fear": ("Negative", "Negative"),
            "disgust": ("Negative", "Negative")
        }

        return mapping.get(baidu_type, ("Neutral", "Neutral"))

    def analyze(self, img_np, baidu_face_data=None):
        """核心业务层调用此方法进行映射"""
        if baidu_face_data is None:
            return "Unknown", "Neutral"
        emotion_cn, emotion_type = self._map_baidu_emotion(baidu_face_data.get("emotion", {}).get("type", "neutral"))
        return emotion_cn, emotion_type