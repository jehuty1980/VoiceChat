import os
import re
import queue
import threading
import time
import torch
import sounddevice as sd
import numpy as np
import tempfile
from typing import Optional, List, Union
import logging

# -------------------------- 基础配置与日志 --------------------------
# 解决Windows临时文件路径问题
tempfile.tempdir = os.environ.get('TEMP', tempfile.tempdir)


# 复用全局日志初始化函数
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# 初始化TTS模块logger
logger = setup_logger("TTSModule")


# -------------------------- 配置类（统一管理参数） --------------------------
class TTSConfig:
    """TTS模块配置类，集中管理参数"""
    # 内置音色列表（只读）
    supported_speakers = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

    def __init__(
            self,
            model_path: str = r'E:\models\CosyVoice-300M-SFT',
            default_speaker: str = "中文女",
            min_segment_length: int = 15,
            play_extra_delay: float = 0.5,  # 播放额外等待时间（秒）
            queue_timeout: float = 1.0  # 音频队列获取超时（秒）
    ):
        self.model_path = model_path
        self.default_speaker = default_speaker
        self.current_speaker = self.default_speaker
        self.min_segment_length = min_segment_length
        self.play_extra_delay = play_extra_delay
        self.queue_timeout = queue_timeout



# -------------------------- TTS核心模块类 --------------------------
class CosyVoiceTTS:
    """CosyVoice TTS模块，支持分句合成、异步播放"""

    def __init__(self, config: Optional[TTSConfig] = None):
        """
        初始化TTS模块
        :param config: TTS配置对象，None则使用默认配置
        """
        self.config = config or TTSConfig()
        self.model = None  # 延迟加载模型
        self.audio_queue = queue.Queue()  # 音频播放队列
        self.is_running = False  # 播放/生成状态标记
        self.model_lock = threading.Lock()  # 模型调用锁
        self.play_thread: Optional[threading.Thread] = None  # 播放线程
        # 数字转汉字映射表（简单转换）
        self.num2han = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }

    def load_model(self) -> None:
        """加载CosyVoice模型（仅接收模型路径，无其他参数）"""
        if self.model is not None:
            logger.warning("TTS模型已加载，无需重复初始化")
            return

        # 延迟导入AutoModel，避免初始化时依赖问题
        from cosyvoice.cli.cosyvoice import AutoModel

        try:
            logger.info(f"开始加载TTS模型，路径: {self.config.model_path}")
            # 严格按要求：仅传递model_dir参数
            self.model = AutoModel(model_dir=self.config.model_path)
            logger.info("TTS模型加载成功")
        except Exception as e:
            logger.error(f"TTS模型加载失败: {str(e)}", exc_info=True)
            raise

    def unload_model(self) -> None:
        """释放模型资源"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("TTS模型已释放")

    def _filter_emoji(self, text: str) -> str:
        """
        温和过滤表情符号，只删emoji，不删文字
        """
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U0001f900-\U0001f9ff"
            u"\u2600-\u2b55"
            u"\ufe0f"
            "]+",
            re.UNICODE
        )
        cleaned = emoji_pattern.sub('', text)
        # 把多余空格变成单个空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _num2han_simple(self, text: str) -> str:
        """
        新增：简单将阿拉伯数字转换为汉字（逐位转换，如102→一零二）
        :param text: 过滤表情后的文本
        :return: 数字转汉字后的文本（仅用于TTS合成，不影响文字输出）
        """
        result = []
        for char in text:
            # 逐字符判断，是数字则转汉字，否则保留原样
            result.append(self.num2han.get(char, char))
        return ''.join(result)

    def _filter_markdown(self, text: str) -> str:
        """
        新增：过滤markdown特殊符号（仅移除标记，保留文本内容）
        处理场景：标题符号(#)、加粗(**/__)、斜体(*/_)、链接([](url))、代码(`)、列表符号(-/*)等
        """
        import re
        # 1. 移除链接（保留链接文本，删除url）
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # 2. 移除代码块/行标记
        text = re.sub(r'```[\s\S]*?```', '', text)  # 多行代码块
        text = re.sub(r'`([^`]+)`', r'\1', text)  # 单行代码
        # 3. 移除标题符号（#）、列表符号（-/*）
        text = re.sub(r'^[#*-]+ ', '', text, flags=re.MULTILINE)
        # 4. 移除加粗/斜体符号（**/__/*/_）
        text = re.sub(r'(\*\*|__|\*|_)', '', text)
        # 5. 移除多余空格（过滤后可能产生的冗余空格）
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_text(self, text: str) -> List[str]:
        """
        按标点分句（内部方法，无冗余输出）- 优化：加入感叹号，支持中英文标点
        :param text: 原始文本
        :return: 处理后的分句列表
        """
        # 优化：加入感叹号（中文！+英文!），同时支持中英文标点分割，保留标点
        pattern = r'([^，。；！!]+[，。；！!]?)'
        raw_segments = [seg.strip() for seg in re.findall(pattern, text) if seg.strip()]

        if not raw_segments:
            return []

        # 合并过短分句（保持原有逻辑不变）
        processed_segments = []
        current_seg = ""
        for seg in raw_segments:
            current_seg += seg
            if len(current_seg) >= self.config.min_segment_length:
                processed_segments.append(current_seg)
                current_seg = ""

        # 处理最后一段（保持原有逻辑不变）
        if current_seg:
            if processed_segments and len(current_seg) < self.config.min_segment_length:
                processed_segments[-1] += current_seg
            else:
                processed_segments.append(current_seg)

        return processed_segments

    def _audio_play_worker(self, sample_rate: int) -> None:
        """音频播放线程（内部方法，无冗余输出）"""
        try:
            while self.is_running:
                try:
                    audio_data = self.audio_queue.get(timeout=self.config.queue_timeout)
                    if audio_data is None:  # 播放结束标记
                        break
                    # 播放音频
                    sd.play(audio_data, samplerate=sample_rate)
                    # 等待播放完成（加额外延迟）
                    play_duration = len(audio_data) / sample_rate
                    time.sleep(play_duration + self.config.play_extra_delay)
                    sd.stop()
                except queue.Empty:
                    continue
        except Exception as e:
            logger.error(f"音频播放线程异常: {str(e)}", exc_info=True)
        finally:
            self.is_running = False

    def text_to_speech(
            self,
            text: str,
            speaker: Optional[str] = None,
            play_audio: bool = True,
            keep_original: bool = True  # 新增：是否保留原始文本（用于文字输出）
    ) -> Union[List[np.ndarray], None]:
        """
        核心TTS接口：文本转语音 - 新增：数字转汉字（仅用于TTS，不影响文字输出）
        :param text: 待合成文本（原始LLM输出）
        :param speaker: 音色，None则使用默认配置
        :param play_audio: 是否立即播放（False则仅生成音频数据不播放）
        :param keep_original: 是否保留原始文本（用于文字输出，不参与TTS处理）
        :return: 音频数据列表（每个元素为单分句的numpy数组），失败返回None
        """
        # 前置校验
        if self.model is None:
            logger.error("TTS模型未加载，请先调用load_model()")
            return None

        original_text = text.strip()  # 保存原始文本（用于文字输出，不修改）
        if not original_text:
            logger.warning("待合成文本为空，跳过TTS")
            return None

        # 步骤1：过滤表情符号（仅用于TTS处理，不修改原始文本）
        tts_text = self._filter_emoji(original_text)
        # 步骤2：过滤markdown标记
        tts_text = self._filter_markdown(tts_text)
        # 步骤3：数字转汉字（仅用于TTS合成，不影响文字输出）
        tts_text = self._num2han_simple(tts_text)

        # 校验音色（保持原有逻辑不变）
        target_speaker = speaker or self.config.default_speaker
        if target_speaker not in self.config.supported_speakers:
            logger.error(f"无效音色: {target_speaker}，支持的音色: {self.config.supported_speakers}")
            return None

        # 分句处理（使用优化后的_split_text方法，基于处理后的tts_text）
        text_segments = self._split_text(tts_text)
        if not text_segments:
            logger.warning("文本分句后无有效内容")
            return None

        # 准备播放（如需播放）（保持原有逻辑不变）
        sample_rate = self.model.sample_rate
        audio_data_list = []  # 存储生成的音频数据

        if play_audio:
            # 清空队列，初始化播放状态
            while not self.audio_queue.empty():
                self.audio_queue.get()
            self.is_running = True
            # 启动播放线程
            self.play_thread = threading.Thread(target=self._audio_play_worker, args=(sample_rate,))
            self.play_thread.start()

        # 分句生成语音（加锁保证单线程调用）（使用处理后的tts_text分句）
        logger.info(f"开始合成语音，共{len(text_segments)}个分句，音色: {target_speaker}")
        for idx, seg in enumerate(text_segments):
            try:
                with self.model_lock:
                    # 非流式生成完整语音（保证音质）
                    audio_chunks = []
                    for chunk in self.model.inference_sft(seg, target_speaker, stream=False):
                        audio_chunks.append(chunk['tts_speech'])

                    if audio_chunks:
                        # 拼接并转换为numpy数组
                        audio = torch.cat(audio_chunks, dim=1).squeeze().cpu().numpy()
                        audio_data_list.append(audio)
                        # 放入播放队列（如需播放）
                        if play_audio:
                            self.audio_queue.put(audio)
                        logger.debug(f"第{idx + 1}个分句合成完成，长度: {len(seg)}字")
                    else:
                        logger.warning(f"第{idx + 1}个分句'{seg}'未生成音频")
            except Exception as e:
                logger.error(f"第{idx + 1}个分句合成失败: {str(e)}", exc_info=True)
                continue

        # 等待播放完成（如需播放）（保持原有逻辑不变）
        if play_audio and self.is_running:
            self.audio_queue.put(None)  # 发送结束标记
            self.play_thread.join()
            self.is_running = False
            logger.info("所有语音片段播放完成")

        return audio_data_list if audio_data_list else None

    def get_supported_speakers(self) -> List[str]:
        """获取支持的音色列表（对外接口）"""
        return self.config.supported_speakers.copy()


# -------------------------- 模块使用示例 --------------------------
def main():
    """TTS模块测试入口（精简输出）"""
    # 1. 初始化配置
    tts_config = TTSConfig(
        model_path=r'E:\models\CosyVoice-300M-SFT',
        default_speaker="中文女",
        min_segment_length=15
    )

    # 2. 初始化TTS模块
    tts_module = CosyVoiceTTS(tts_config)

    try:
        # 3. 加载模型
        tts_module.load_model()

        logger.info("\n=== CosyVoice TTS模块测试 ===")
        logger.info(f"支持的音色: {tts_module.get_supported_speakers()}")
        logger.info("输入 'quit'/'exit' 退出 | 输入 'list' 查看音色 | 其他文本为合成内容")
        logger.info("-" * 50)

        while True:
            user_input = input("\n请输入合成文本: ").strip()

            # 退出条件
            if user_input.lower() in ['quit', 'exit']:
                logger.info("程序退出")
                break

            # 查看音色
            if user_input.lower() == 'list':
                logger.info(f"支持的音色: {tts_module.get_supported_speakers()}")
                continue

            # 合成并播放语音（测试：文字输出用原样，TTS用处理后的）
            print(f"\n【文字输出（原样）】: {user_input}")
            tts_module.text_to_speech(user_input)

    except KeyboardInterrupt:
        logger.info("用户终止程序，清理资源...")
    except Exception as e:
        logger.error(f"程序异常: {str(e)}", exc_info=True)
    finally:
        # 释放模型
        tts_module.unload_model()


if __name__ == '__main__':
    # 极简依赖检查（仅报错，无冗余提示）
    try:
        import torch
        import sounddevice
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError as e:
        logger.error(f"缺少依赖: {str(e)}，请安装对应包")
        exit(1)

    main()