# asr_module.py（轻量流式版）
import os
import time
import torch
import numpy as np
import sounddevice as sd
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging
import threading


# -------------------------- 日志终极配置（彻底解决重复） --------------------------
def setup_logger(name: str) -> logging.Logger:
    """
    全局日志初始化函数（确保每个logger仅初始化一次）
    :param name: logger名称
    :return: 配置好的logger
    """
    # 获取logger，禁止传递给父级（防止重复输出）
    logger = logging.getLogger(name)
    logger.propagate = False  # 核心：禁止日志向上传播
    logger.setLevel(logging.INFO)

    # 仅在无处理器时添加
    if logger.handlers:
        return logger

    # 配置处理器
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# 初始化ASR模块logger
logger = setup_logger("ASRModule")


# -------------------------- 配置类 --------------------------
@dataclass
class ASRConfig:
    """ASR模块配置类"""
    model_path: str = r"E:\models\sense-voice-small"
    vad_model_path: str = r"E:\models\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    language: str = "zh_cn"
    use_itn: bool = True
    batch_size_s: int = 60
    merge_vad: bool = True
    merge_length_s: int = 15
    vad_max_single_segment_time: int = 30000
    sampling_rate: int = 16000
    channels: int = 1
    default_record_duration: int = 5
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 新增流式配置
    stream_chunk_duration: float = 0.2  # 流式分片时长（200ms）
    silence_timeout: int = 3  # 静音超时自动结束（3秒）
    max_record_duration: int = 60  # 最大录音时长（60秒）
    # 新增：有效语音检测阈值配置（默认0.001）
    voice_detection_threshold: float = 0.001  # 有效语音能量阈值


# -------------------------- ASR核心类 --------------------------
class SenseVoiceASR:
    """SenseVoice ASR模块（优化日志+流式识别）"""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.model: Optional[AutoModel] = None
        self._validate_config()
        # 流式识别状态变量
        self.is_recording = False
        self.audio_buffer = []
        self.last_speech_time = time.time()
        self.full_recognized_text = ""
        self.record_thread: Optional[threading.Thread] = None
        self.stop_flag = False

    def _validate_config(self):
        """校验配置（精简日志输出）"""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"ASR模型路径不存在: {self.config.model_path}")
        if not os.path.exists(self.config.vad_model_path):
            raise FileNotFoundError(f"VAD模型路径不存在: {self.config.vad_model_path}")
        logger.debug(f"ASR配置校验通过，设备: {self.config.device}")  # 降级为debug，减少输出

    def load_model(self) -> None:
        """加载模型（精简日志，防止重复加载）"""
        if self.model is not None:
            logger.warning("ASR模型已加载，跳过重复加载")
            return

        try:
            logger.info("开始加载ASR模型...")
            self.model = AutoModel(
                model=self.config.model_path,
                vad_model=self.config.vad_model_path,
                vad_kwargs={"max_single_segment_time": self.config.vad_max_single_segment_time},
                device=self.config.device,
                disable_update=True
            )
            logger.info("ASR模型加载成功")
        except Exception as e:
            logger.error(f"ASR模型加载失败: {str(e)}", exc_info=True)
            raise

    def unload_model(self) -> None:
        """释放模型资源"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("ASR模型已释放")

    def record_audio(self, duration: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """原有录音接口（保留，兼容旧代码）"""
        record_duration = duration or self.config.default_record_duration
        logger.info(f"开始录音 {record_duration} 秒，请说话...")

        try:
            audio_data = sd.rec(
                int(record_duration * self.config.sampling_rate),
                samplerate=self.config.sampling_rate,
                channels=self.config.channels,
                dtype="float32"
            )
            sd.wait()
            logger.info("录音结束")

            # 格式标准化
            if self.config.channels > 1:
                audio_data = np.mean(audio_data, axis=1)
            else:
                audio_data = audio_data.flatten()
            audio_data = audio_data.astype(np.float32)

            return audio_data, self.config.sampling_rate
        except Exception as e:
            logger.error(f"音频采集失败: {str(e)}", exc_info=True)
            raise

    def recognize_audio(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """原有识别接口（保留，兼容旧代码）"""
        if self.model is None:
            raise RuntimeError("ASR模型未加载，请先调用load_model()")

        target_language = language or self.config.language
        logger.debug(f"开始识别语音（语言: {target_language}）")  # 降级为debug

        try:
            res = self.model.generate(
                input=audio_data,
                cache={},
                language=target_language,
                use_itn=self.config.use_itn,
                batch_size_s=self.config.batch_size_s,
                merge_vad=self.config.merge_vad,
                merge_length_s=self.config.merge_length_s,
            )
            text = rich_transcription_postprocess(res[0]["text"])
            logger.info(f"语音识别结果: {text}")
            return text
        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}", exc_info=True)
            return ""

    def recognize_from_mic(self, duration: Optional[int] = None, language: Optional[str] = None) -> str:
        """原有一站式接口（保留，兼容旧代码）"""
        audio_data, _ = self.record_audio(duration)
        return self.recognize_audio(audio_data, language)

    def _stream_record_worker(self, language: Optional[str]):
        """流式录音+识别线程（内部方法）"""
        chunk_samples = int(self.config.stream_chunk_duration * self.config.sampling_rate)
        start_time = time.time()
        target_language = language or self.config.language
        cache = {}

        # 初始化音频流
        with sd.InputStream(
                samplerate=self.config.sampling_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=chunk_samples
        ) as stream:
            logger.info(
                f"开始流式录音（静音{self.config.silence_timeout}秒/最长{self.config.max_record_duration}秒自动结束），请说话...")

            while not self.stop_flag:
                # 1. 检查最大时长
                if time.time() - start_time > self.config.max_record_duration:
                    logger.info(f"达到最大录音时长{self.config.max_record_duration}秒，自动结束")
                    break

                # 2. 读取音频分片
                audio_chunk, overflowed = stream.read(chunk_samples)
                if overflowed:
                    logger.warning("音频缓冲区溢出，可能丢失部分数据")

                # 3. 格式标准化
                if self.config.channels > 1:
                    audio_chunk = np.mean(audio_chunk, axis=1)
                else:
                    audio_chunk = audio_chunk.flatten()
                audio_chunk = audio_chunk.astype(np.float32)

                # 4. 检测是否有有效语音（简单能量检测）
                audio_energy = np.sqrt(np.mean(np.square(audio_chunk)))
                if audio_energy > self.config.voice_detection_threshold:  # 有有效语音
                    self.last_speech_time = time.time()
                    self.audio_buffer.append(audio_chunk)

                    # 5. 流式识别
                    full_audio = np.concatenate(self.audio_buffer) if self.audio_buffer else np.array([])
                    if len(full_audio) > 0:
                        try:
                            res = self.model.generate(
                                input=full_audio,
                                cache=cache,
                                language=target_language,
                                use_itn=self.config.use_itn,
                                batch_size_s=self.config.batch_size_s,
                                merge_vad=self.config.merge_vad,
                                merge_length_s=self.config.merge_length_s,
                            )
                            current_text = rich_transcription_postprocess(res[0]["text"])

                            # 6. 逐句拼接输出（核心要求）
                            if current_text != self.full_recognized_text:
                                self.full_recognized_text = current_text
                                print(self.full_recognized_text)  # 逐次输出完整拼接的句子
                        except Exception as e:
                            logger.error(f"流式识别出错: {str(e)}", exc_info=True)
                else:
                    # 7. 检查静音超时
                    if time.time() - self.last_speech_time > self.config.silence_timeout:
                        logger.info(f"检测到{self.config.silence_timeout}秒无语音输入，自动结束录音")
                        break

                # 8. 小延迟，降低CPU占用
                time.sleep(0.01)

        # 录音结束，清理状态
        self.is_recording = False
        stream.stop()

    def stream_recognize_from_mic(self, language: Optional[str] = None) -> str:
        """
        新增：轻量流式识别接口（按enter开始，满足3个核心要求）
        :param language: 识别语言
        :return: 完整识别结果
        """
        if self.model is None:
            raise RuntimeError("ASR模型未加载，请先调用load_model()")

        # 重置状态
        self.is_recording = True
        self.audio_buffer = []
        self.last_speech_time = time.time()
        self.full_recognized_text = ""
        self.stop_flag = False

        # 启动流式录音线程
        self.record_thread = threading.Thread(target=self._stream_record_worker, args=(language,))
        self.record_thread.start()

        # 等待录音线程结束
        self.record_thread.join()

        # 返回完整识别结果
        logger.info(f"流式识别最终结果: {self.full_recognized_text}")
        return self.full_recognized_text

    def stop_stream_recording(self):
        """手动停止流式录音"""
        self.stop_flag = True
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join()
        self.is_recording = False
        logger.info("已手动停止流式录音")


# -------------------------- 模块使用示例 --------------------------
def main():
    """ASR模块测试入口（新增流式测试）"""
    # 1. 初始化配置（可自定义参数）
    custom_config = ASRConfig(
        default_record_duration=15,
        language="zh_cn",
        silence_timeout=3,  # 3秒静音自动结束
        max_record_duration=60  # 最长60秒
    )

    # 2. 初始化ASR模块
    asr_module = SenseVoiceASR(custom_config)

    try:
        # 3. 加载模型
        asr_module.load_model()

        # 4. 流式识别测试
        logger.info("\n=== SenseVoice 流式ASR模块测试 ===")
        logger.info("按回车开始流式录音（3秒静音/60秒最长自动结束），按Ctrl+C退出")
        while True:
            input("按回车开始流式录音...")
            result = asr_module.stream_recognize_from_mic()
            logger.info("-" * 60)
            logger.info(f"最终识别结果: {result if result else '未识别到有效语音'}")
            logger.info("-" * 60)

    except KeyboardInterrupt:
        logger.info("\n用户终止程序，开始清理资源...")
        asr_module.stop_stream_recording()
    except Exception as e:
        logger.error(f"程序异常: {str(e)}", exc_info=True)
    finally:
        # 5. 释放模型资源
        asr_module.unload_model()
        logger.info("程序正常退出")


if __name__ == "__main__":
    main()