import logging
import warnings
from typing import Optional

# -------------------------- 全局终极配置（彻底杜绝重复） --------------------------
# 1. 强制清空所有已有日志处理器
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# 2. 过滤所有第三方警告
warnings.filterwarnings("ignore")

# 3. 全局日志仅初始化一次
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)

# 4. 屏蔽所有第三方库日志
logging.getLogger("onnxruntime").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)
logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("funasr").setLevel(logging.CRITICAL)
logging.getLogger("cosyvoice").setLevel(logging.CRITICAL)


# -------------------------- 导入优化后的模块 --------------------------
from asr_module import SenseVoiceASR, ASRConfig
from llm_module import LlamaServerLLM, LLMConfig
from tts_module import CosyVoiceTTS, TTSConfig


# -------------------------- 全链路联动核心类 --------------------------
class VoiceAssistant:
    """ASR+LLM+TTS全链路语音助手"""

    def __init__(
            self,
            asr_config: Optional[ASRConfig] = None,
            llm_config: Optional[LLMConfig] = None,
            tts_config: Optional[TTSConfig] = None,
            default_session_id: str = "voice_assistant_session",
            default_tts_speaker: str = "中文女",
            record_duration: int = 5
    ):
        """
        初始化语音助手
        :param asr_config: ASR配置
        :param llm_config: LLM配置
        :param tts_config: TTS配置
        :param default_session_id: LLM默认会话ID
        :param default_tts_speaker: TTS默认音色
        :param record_duration: ASR默认录音时长（秒）
        """
        # 初始化各模块
        self.asr_config = asr_config or ASRConfig()
        self.asr_module = SenseVoiceASR(self.asr_config)
        self.record_duration = record_duration

        self.llm_config = llm_config or LLMConfig()
        self.llm_module = LlamaServerLLM(self.llm_config)
        self.default_session_id = default_session_id

        self.tts_config = tts_config or TTSConfig()
        self.tts_module = CosyVoiceTTS(self.tts_config)
        self.default_tts_speaker = default_tts_speaker

        # 模块加载状态
        self.asr_loaded = False
        self.llm_loaded = False
        self.tts_loaded = False
        self.logger = logging.getLogger("VoiceAssistant")

    def load_all_modules(self):
        """加载所有模块（防止重复加载）"""
        # 加载ASR
        if not self.asr_loaded:
            self.logger.info("===== 加载ASR模块 =====")
            self.asr_module.load_model()
            self.asr_loaded = True

        # 加载LLM
        if not self.llm_loaded:
            self.logger.info("===== 加载LLM模块 =====")
            self.llm_module._init_client()
            self.llm_loaded = True

        # 加载TTS
        if not self.tts_loaded:
            self.logger.info("===== 加载TTS模块 =====")
            self.tts_module.load_model()
            self.tts_loaded = True

        self.logger.info("所有模块加载完成！")

    def unload_all_modules(self):
        """释放所有模块资源"""
        self.logger.info("开始释放资源...")
        if self.asr_loaded:
            self.asr_module.unload_model()
        if self.llm_loaded:
            self.llm_module.close()
        if self.tts_loaded:
            self.tts_module.unload_model()
        self.logger.info("所有资源释放完成！")

    def voice_chat(self) -> str:
        """
        核心语音对话流程：录音→ASR识别→LLM回复→文字反馈（优先）→TTS语音播放
        :return: LLM回复文本
        """
        # 1. 录音+ASR识别
        self.logger.info("\n===== 准备接收语音输入 =====")
        # asr_result = self.asr_module.recognize_from_mic(duration=self.record_duration)
        asr_result = self.asr_module.stream_recognize_from_mic(language=self.asr_config.language)
        if not asr_result:
            self.logger.warning("未识别到有效语音")
            return "未识别到有效语音，请重新说话！"

        # 2. LLM生成回复（保留多轮上下文）
        self.logger.info(f"识别到的语音内容: {asr_result}")
        try:
            llm_response = self.llm_module.chat(
                message=asr_result,
                session_id=self.default_session_id
            )
        except Exception as e:
            error_msg = f"LLM回复生成失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

        # 3. 优先输出文字反馈（核心要求：文字先于语音）
        print(f"\n【文字回复】: {llm_response}")
        self.logger.info(f"LLM生成回复: {llm_response}")

        # 4. 异步触发TTS语音播放（不阻塞文字输出）
        # 注意：TTS的text_to_speech内部已用多线程播放，此处直接调用即可
        try:
            self.logger.info("开始生成并播放语音回复...")
            # 非阻塞调用TTS（内部多线程播放，不等待语音结束）
            self.tts_module.text_to_speech(
                text=llm_response,
                speaker=self.default_tts_speaker,
                play_audio=True
            )
        except Exception as e:
            self.logger.error(f"TTS语音播放失败: {str(e)}")

        return llm_response


# -------------------------- 测试主流程 --------------------------
def main():
    """ASR+LLM+TTS全链路测试入口"""
    # 1. 自定义配置（根据实际环境修改）
    asr_config = ASRConfig(
        model_path=r"E:\models\sense-voice-small",
        vad_model_path=r"E:\models\speech_fsmn_vad_zh-cn-16k-common-pytorch",
        default_record_duration=15,
        language="zh_cn"
    )

    llm_config = LLMConfig(
        base_url="http://127.0.0.1:8080/v1",
        model_alias="qwen35",
        max_history=20,
        system_prompt="你叫千语，是我的语音助手，回复简洁、友好，不超过250字。注意事项：我给你的信息是经过ASR转录的文字，因此有很多识别错误的词汇，你应该根据上下文进行自动判断进行修正，禁止回复修正的原因和过程。"
    )

    tts_config = TTSConfig(
        model_path=r'E:\models\CosyVoice-300M-SFT',
        default_speaker="中文女",
        min_segment_length=15
    )

    # 2. 初始化语音助手
    assistant = VoiceAssistant(
        asr_config=asr_config,
        llm_config=llm_config,
        tts_config=tts_config,
        record_duration=5
    )

    try:
        # 3. 加载所有模块
        assistant.load_all_modules()

        # 4. 交互提示（精简输出）
        print("\n=====================================")
        print("🎙️  ASR+LLM+TTS 全链路语音助手")
        print("=====================================")
        print("💡  每次按回车后开始录音（5秒），说出你的问题")
        print("💡  输入 'exit'/'quit' 退出程序")
        print("💡  输入 'clear' 清空聊天记录")
        print("=====================================\n")

        # 5. 循环交互
        while True:
            user_cmd = input("按回车开始录音（或输入指令）: ").strip()

            # 退出指令
            if user_cmd.lower() in ["exit", "quit"]:
                assistant.logger.info("用户退出程序")
                break

            # 清空聊天记录
            if user_cmd.lower() == "clear":
                assistant.llm_module.clear_session(assistant.default_session_id)
                print("✅ 已清空聊天记录！")
                continue

            # 空输入/回车：触发语音对话
            if not user_cmd:
                try:
                    # 核心语音对话流程
                    assistant.voice_chat()
                except Exception as e:
                    assistant.logger.error(f"语音对话失败: {str(e)}", exc_info=True)
                    print(f"❌ 处理失败: {str(e)}")
            else:
                print("❌ 无效指令，请按回车录音，或输入exit/quit/clear")

    except KeyboardInterrupt:
        assistant.logger.info("\n用户终止程序")
    except Exception as e:
        assistant.logger.error(f"程序异常: {str(e)}", exc_info=True)
    finally:
        # 释放所有资源
        assistant.unload_all_modules()
        print("\n👋 程序已退出")


if __name__ == "__main__":
    main()