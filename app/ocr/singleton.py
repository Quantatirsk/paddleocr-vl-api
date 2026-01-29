"""
PaddleOCR 单例模式实现
"""
import threading
import logging
from paddleocr import PaddleOCRVL
from ..config import VLLM_SERVER_URL

logger = logging.getLogger(__name__)


class PaddleOCRSingleton:
    """使用单例模式初始化 PaddleOCR（避免重复加载模型）"""
    _instance = None
    _pipeline = None
    _lock = threading.Lock()  # 线程锁，确保线程安全

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            logger.info("Initializing PaddleOCR pipeline...")
            cls._pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=VLLM_SERVER_URL
            )
            logger.info("PaddleOCR pipeline initialized successfully")
        return cls._pipeline

    @classmethod
    def get_lock(cls):
        """获取锁对象"""
        return cls._lock
