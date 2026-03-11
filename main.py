import tkinter as tk
from tkinter import scrolledtext, ttk, Text, filedialog
import logging
import warnings
import threading
from typing import Optional
import os

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


# -------------------------- 全链路联动核心类（复用原有逻辑） --------------------------
class VoiceAssistant:
    """ASR+LLM+TTS全链路语音助手"""

    def __init__(
            self,
            asr_config: Optional[ASRConfig] = None,
            llm_config: Optional[LLMConfig] = None,
            tts_config: Optional[TTSConfig] = None,
            default_session_id: str = "voice_assistant_session",
            record_duration: int = 5,
            ui_callback=None  # UI回调函数，用于更新界面
    ):
        """
        初始化语音助手
        :param asr_config: ASR配置
        :param llm_config: LLM配置
        :param tts_config: TTS配置
        :param default_session_id: LLM默认会话ID
        :param record_duration: ASR默认录音时长（秒）
        :param ui_callback: UI更新回调函数 (msg_type, content)
                            msg_type: "asr", "llm", "status", "error"
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

        # 模块加载状态
        self.asr_loaded = False
        self.llm_loaded = False
        self.tts_loaded = False
        self.logger = logging.getLogger("VoiceAssistant")
        self.ui_callback = ui_callback  # 保存UI回调

    def _update_ui(self, msg_type, content):
        """更新UI界面（线程安全）"""
        if self.ui_callback:
            try:
                self.ui_callback(msg_type, content)
            except Exception as e:
                self.logger.error(f"UI更新失败: {str(e)}")

    def load_all_modules(self):
        """加载所有模块（防止重复加载）"""
        self._update_ui("status", "开始加载ASR模块...")
        # 加载ASR
        if not self.asr_loaded:
            try:
                self.asr_module.load_model()
                self.asr_loaded = True
                self._update_ui("status", "ASR模块加载完成！")
            except Exception as e:
                self._update_ui("error", f"ASR加载失败: {str(e)}")
                self.logger.error(f"ASR加载失败: {str(e)}")
                return False

        # 加载LLM
        self._update_ui("status", "开始加载LLM模块...")
        if not self.llm_loaded:
            try:
                self.llm_module._init_client()
                self.llm_loaded = True
                self._update_ui("status", "LLM模块加载完成！")
            except Exception as e:
                self._update_ui("error", f"LLM加载失败: {str(e)}")
                self.logger.error(f"LLM加载失败: {str(e)}")
                return False

        # 加载TTS
        self._update_ui("status", "开始加载TTS模块...")
        if not self.tts_loaded:
            try:
                self.tts_module.load_model()
                self.tts_loaded = True
                self._update_ui("status", "TTS模块加载完成！")
            except Exception as e:
                self._update_ui("error", f"TTS加载失败: {str(e)}")
                self.logger.error(f"TTS加载失败: {str(e)}")
                return False

        self._update_ui("status", "所有模块加载完成！可以开始语音对话~")
        self.logger.info("所有模块加载完成！")
        return True

    def unload_all_modules(self):
        """释放所有模块资源"""
        self._update_ui("status", "开始释放资源...")
        try:
            if self.asr_loaded:
                self.asr_module.unload_model()
            if self.llm_loaded:
                self.llm_module.close()
            if self.tts_loaded:
                self.tts_module.unload_model()
            self._update_ui("status", "所有资源释放完成！")
        except Exception as e:
            self._update_ui("error", f"资源释放失败: {str(e)}")
            self.logger.error(f"资源释放失败: {str(e)}")

    def voice_chat(self):
        """
        核心语音对话流程（线程执行，不阻塞UI）
        """

        def run_chat():
            # 1. 录音+ASR识别
            self._update_ui("status", "\n🎙️  准备接收语音输入...")
            self._update_ui("status",
                            f"（{self.asr_module.config.silence_timeout}秒静音/{self.asr_module.config.max_record_duration}秒最长自动结束）")
            try:
                asr_result = self.asr_module.stream_recognize_from_mic(language=self.asr_config.language)
                if not asr_result:
                    self._update_ui("status", "⚠️  未识别到有效语音")
                    self.logger.warning("未识别到有效语音")
                    return
                # 更新ASR识别结果到UI
                self._update_ui("asr", f"你说：{asr_result}")
                self.logger.info(f"识别到的语音内容: {asr_result}")
            except Exception as e:
                self._update_ui("error", f"ASR识别失败: {str(e)}")
                self.logger.error(f"ASR识别失败: {str(e)}")
                return

            # 2. LLM生成回复（保留多轮上下文）
            self._update_ui("status", "🤖  正在思考回复...")
            try:
                llm_response, llm_reasoning = self.llm_module.chat(
                    message=asr_result,
                    session_id=self.default_session_id
                )
                # 更新LLM回复到UI
                self._update_ui("llm", f"千语：{llm_response}")
                # 新增：更新思考过程到UI
                self._update_ui("llm_thinking", llm_reasoning)
                self.logger.info(f"LLM生成回复: {llm_response} | 思考过程: {llm_reasoning}")
            except Exception as e:
                error_msg = f"LLM回复生成失败: {str(e)}"
                self._update_ui("error", error_msg)
                self.logger.error(error_msg)
                return

            # 3. 异步触发TTS语音播放
            self._update_ui("status", "🔊  开始播放语音回复...")
            try:
                self.tts_module.text_to_speech(
                    text=llm_response,
                    speaker=self.tts_config.current_speaker,
                    play_audio=True
                )
                self._update_ui("status", "✅  语音回复播放完成！")
            except Exception as e:
                self._update_ui("error", f"TTS语音播放失败: {str(e)}")
                self.logger.error(f"TTS语音播放失败: {str(e)}")

        # 启动线程执行语音对话，避免阻塞UI
        chat_thread = threading.Thread(target=run_chat)
        chat_thread.daemon = True
        chat_thread.start()

    def clear_chat_history(self):
        """清空聊天记录"""
        try:
            self.llm_module.clear_session(self.default_session_id)
            self._update_ui("status", "✅  已清空聊天记录！")
        except Exception as e:
            self._update_ui("error", f"清空记录失败: {str(e)}")
            self.logger.error(f"清空记录失败: {str(e)}")

    # 更新配置方法（新增模型路径/LLM参数支持）
    def update_configs(self, system_prompt=None, silence_timeout=None, max_record_duration=None, voice_threshold=None,
                       max_history=None, asr_model_path=None, asr_vad_path=None, tts_model_path=None, tts_speaker=None,
                       llm_base_url=None, llm_model_alias=None, llm_temperature=None,
                       llm_top_p=None, llm_max_tokens=None, llm_thinking=None):
        """
        更新配置参数
        :param system_prompt: 新的系统提示词
        :param silence_timeout: 新的静音检测阈值（秒）
        :param max_record_duration: 新的录音上限（秒）
        :param voice_threshold: 语音能量阈值
        :param tts_speaker: 说话人
        # 新增模型路径参数
        :param asr_model_path: ASR模型路径
        :param asr_vad_path: ASR VAD模型路径
        :param tts_model_path: TTS模型路径
        # 新增LLM参数
        :param llm_base_url: OpenAI兼容API地址
        :param llm_model_alias: 模型名称
        :param llm_temperature: 温度系数
        :param llm_top_p: top_p
        :param llm_max_tokens: 最大生成token数
        :param llm_thinking: 是否启用思考模式
        :param max_history: 新的最大对话轮次


        """
        # 1. 基础配置更新（原有逻辑）
        if system_prompt is not None and system_prompt.strip():
            self.llm_config.system_prompt = system_prompt.strip()
            if self.llm_loaded:
                self.llm_module._init_client()
            self.logger.info(f"更新系统提示词完成: {system_prompt[:50]}...")

        if silence_timeout is not None and isinstance(silence_timeout, (int, float)) and silence_timeout > 0:
            self.asr_config.silence_timeout = silence_timeout
            self.asr_module.config.silence_timeout = silence_timeout
            self.logger.info(f"更新静音检测阈值完成: {silence_timeout}秒")

        if max_record_duration is not None and isinstance(max_record_duration, int) and max_record_duration > 0:
            self.asr_config.max_record_duration = max_record_duration
            self.asr_module.config.max_record_duration = max_record_duration
            self.logger.info(f"更新录音上限完成: {max_record_duration}秒")

        if voice_threshold is not None and isinstance(voice_threshold, float) and voice_threshold >= 0:
            self.asr_config.voice_detection_threshold = voice_threshold
            self.asr_module.config.voice_detection_threshold = voice_threshold
            self.logger.info(f"更新有效语音检测阈值完成: {voice_threshold}")

        # 更新TTS说话人
        if tts_speaker is not None and tts_speaker in self.tts_config.supported_speakers:
            self.tts_config.current_speaker = tts_speaker
            self.logger.info(f"TTS说话人已切换为: {tts_speaker}")

        if max_history is not None and isinstance(max_history, int) and max_history > 0:
            self.llm_config.max_history = max_history
            self.llm_module.max_history = max_history
            self.logger.info(f"更新最大对话轮次完成: {max_history}轮")

        # 2. 新增：模型路径更新（仅修改配置，加载时生效）
        if asr_model_path is not None and asr_model_path.strip() and os.path.isdir(asr_model_path.strip()):
            self.asr_config.model_path = asr_model_path.strip()
            self.logger.info(f"更新ASR模型路径完成: {asr_model_path}")

        if asr_vad_path is not None and asr_vad_path.strip() and os.path.isdir(asr_vad_path.strip()):
            self.asr_config.vad_model_path = asr_vad_path.strip()
            self.logger.info(f"更新ASR VAD模型路径完成: {asr_vad_path}")

        if tts_model_path is not None and tts_model_path.strip() and os.path.isdir(tts_model_path.strip()):
            self.tts_config.model_path = tts_model_path.strip()
            self.logger.info(f"更新TTS模型路径完成: {tts_model_path}")

        # 3. 新增：LLM参数更新
        if llm_base_url is not None and llm_base_url.strip():
            self.llm_config.base_url = llm_base_url.strip()
            if self.llm_loaded:
                self.llm_module._init_client()
            self.logger.info(f"更新LLM API地址完成: {llm_base_url}")

        if llm_model_alias is not None and llm_model_alias.strip():
            self.llm_config.model_alias = llm_model_alias.strip()
            if self.llm_loaded:
                self.llm_module._init_client()
            self.logger.info(f"更新LLM模型名称完成: {llm_model_alias}")

        if llm_temperature is not None and isinstance(llm_temperature, (int, float)) and 0 <= llm_temperature <= 2:
            self.llm_config.temperature = llm_temperature
            if self.llm_loaded:
                self.llm_module._init_client()
            self.logger.info(f"更新LLM温度系数完成: {llm_temperature}")

        if llm_top_p is not None and isinstance(llm_top_p, (int, float)) and 0 <= llm_temperature <= 1:
            self.llm_config.top_p = llm_top_p
            if self.llm_loaded:
                self.llm_module._init_client()
            self.logger.info(f"更新LLM的top_p完成: {llm_top_p}")

        if llm_max_tokens is not None:
            if llm_max_tokens == "" or llm_max_tokens is None:
                self.llm_config.max_tokens = None
                self.logger.info("更新LLM最大Token数为：无限制")
            elif isinstance(llm_max_tokens, int) and llm_max_tokens > 0:
                self.llm_config.max_tokens = llm_max_tokens
                self.logger.info(f"更新LLM最大Token数完成: {llm_max_tokens}")

        if llm_thinking is not None:
            self.llm_config.thinking = llm_thinking
            if self.llm_loaded:
                self.llm_module._init_client()
            self.logger.info(f"LLM思考模式已设置为: {llm_thinking}")

        self._update_ui("status", "✅ 配置参数更新完成（模型路径需重新加载生效）！")


# -------------------------- Tkinter UI界面 --------------------------
class VoiceAssistantUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ 千语 - 语音助手")
        self.root.geometry("1280x800")
        self.root.resizable(True, True)

        # 初始化变量
        self.assistant = None
        self.is_loading = False

        # 设置字体
        self.font_normal = ("Microsoft YaHei", 10)
        self.font_bold = ("Microsoft YaHei", 10, "bold")
        self.font_title = ("Microsoft YaHei", 14, "bold")

        # 创建UI组件（新增模型路径/LLM参数配置）
        self._create_widgets()
        # 初始化语音助手
        self._init_assistant()

    def _create_widgets(self):
        """创建UI组件（新增模型路径/LLM参数配置控件）"""
        # 整体布局：左右分栏（配置区+聊天区）
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -------------------------- 左侧配置面板（分三个子面板） --------------------------
        config_frame = ttk.Frame(main_panel)
        main_panel.add(config_frame, weight=1)

        # 1. 模型路径配置面板（加载后生效）
        model_frame = ttk.LabelFrame(config_frame, text="模型路径配置（需重新加载生效）", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # ASR模型路径
        asr_model_label = ttk.Label(model_frame, text="ASR模型路径：", font=self.font_normal)
        asr_model_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.asr_model_var = tk.StringVar(value=r"E:\models\sense-voice-small")
        asr_model_entry = ttk.Entry(model_frame, textvariable=self.asr_model_var, width=30)
        asr_model_entry.grid(row=0, column=1, padx=5, pady=5)
        asr_model_btn = ttk.Button(model_frame, text="浏览", command=lambda: self._browse_dir(self.asr_model_var))
        asr_model_btn.grid(row=0, column=2, padx=5, pady=5)

        # ASR VAD模型路径
        asr_vad_label = ttk.Label(model_frame, text="ASR VAD路径：", font=self.font_normal)
        asr_vad_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.asr_vad_var = tk.StringVar(value=r"E:\models\speech_fsmn_vad_zh-cn-16k-common-pytorch")
        asr_vad_entry = ttk.Entry(model_frame, textvariable=self.asr_vad_var, width=30)
        asr_vad_entry.grid(row=1, column=1, padx=5, pady=5)
        asr_vad_btn = ttk.Button(model_frame, text="浏览", command=lambda: self._browse_dir(self.asr_vad_var))
        asr_vad_btn.grid(row=1, column=2, padx=5, pady=5)

        # TTS模型路径
        tts_model_label = ttk.Label(model_frame, text="TTS模型路径：", font=self.font_normal)
        tts_model_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.tts_model_var = tk.StringVar(value=r'E:\models\CosyVoice-300M-SFT')
        tts_model_entry = ttk.Entry(model_frame, textvariable=self.tts_model_var, width=30)
        tts_model_entry.grid(row=2, column=1, padx=5, pady=5)
        tts_model_btn = ttk.Button(model_frame, text="浏览", command=lambda: self._browse_dir(self.tts_model_var))
        tts_model_btn.grid(row=2, column=2, padx=5, pady=5)

        # 2. LLM参数配置面板
        llm_frame = ttk.LabelFrame(config_frame, text="LLM参数配置", padding=10)
        llm_frame.pack(fill=tk.X, padx=5, pady=5)

        # API地址
        llm_url_label = ttk.Label(llm_frame, text="API地址：", font=self.font_normal)
        llm_url_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.llm_url_var = tk.StringVar(value="http://127.0.0.1:8080/v1")
        llm_url_entry = ttk.Entry(llm_frame, textvariable=self.llm_url_var, width=30)
        llm_url_entry.grid(row=0, column=1, padx=5, pady=5)

        # 模型名称
        llm_alias_label = ttk.Label(llm_frame, text="模型名称：", font=self.font_normal)
        llm_alias_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.llm_alias_var = tk.StringVar(value="qwen35")
        llm_alias_entry = ttk.Entry(llm_frame, textvariable=self.llm_alias_var, width=30)
        llm_alias_entry.grid(row=1, column=1, padx=5, pady=5)

        # 温度系数
        llm_temp_label = ttk.Label(llm_frame, text="温度系数：", font=self.font_normal)
        llm_temp_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.llm_temp_var = tk.StringVar(value="0.7")
        llm_temp_entry = ttk.Entry(llm_frame, textvariable=self.llm_temp_var, width=10)
        llm_temp_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        # top_p
        llm_top_p_label = ttk.Label(llm_frame, text="top_p：", font=self.font_normal)
        llm_top_p_label.grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.llm_top_p_var = tk.StringVar(value="0.95")
        llm_top_p_entry = ttk.Entry(llm_frame, textvariable=self.llm_top_p_var, width=10)
        llm_top_p_entry.grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)

        # 最大Token数
        llm_tokens_label = ttk.Label(llm_frame, text="最大Token数：", font=self.font_normal)
        llm_tokens_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.llm_tokens_var = tk.StringVar(value="")
        llm_tokens_entry = ttk.Entry(llm_frame, textvariable=self.llm_tokens_var, width=10)
        llm_tokens_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # 最大对话轮次
        history_label = ttk.Label(llm_frame, text="最大对话轮次：", font=self.font_normal)
        history_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.history_var = tk.StringVar(value="20")
        history_entry = ttk.Entry(llm_frame, textvariable=self.history_var, width=10)
        history_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

        # 思考模式开关
        self.llm_thinking_var = tk.BooleanVar(value=False)
        llm_thinking_checkbox = ttk.Checkbutton(llm_frame, text="开启思考模式", variable=self.llm_thinking_var)
        llm_thinking_checkbox.grid(row=4, column=2, padx=5, pady=5, sticky=tk.W)

        # 3. 其他参数配置面板
        other_frame = ttk.LabelFrame(config_frame, text="其他参数配置", padding=10)
        other_frame.pack(fill=tk.X, padx=5, pady=5)

        # 系统提示词
        prompt_label = ttk.Label(other_frame, text="系统提示词：", font=self.font_bold)
        prompt_label.pack(anchor=tk.W, pady=(0, 5))
        self.prompt_text = Text(other_frame, height=6, width=40, font=self.font_normal)
        self.prompt_text.pack(fill=tk.X, pady=(0, 10))
        default_prompt = "你叫千语，是我的语音助手，回复简洁、友好，不超过250字。注意事项：我给你的信息是经过ASR转录的文字，因此有很多识别错误的词汇，你应该根据上下文进行自动判断进行修正，禁止回复修正的原因和过程。"
        self.prompt_text.insert(1.0, default_prompt)

        # 语音检测参数
        vad_frame = ttk.Frame(other_frame)
        vad_frame.pack(fill=tk.X, pady=(0, 10))

        silence_label = ttk.Label(vad_frame, text="静音检测阈值-秒：", font=self.font_normal)
        silence_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.silence_var = tk.StringVar(value="2")
        silence_entry = ttk.Entry(vad_frame, textvariable=self.silence_var, width=10)
        silence_entry.grid(row=0, column=1, padx=5, pady=5)

        speaker_label=ttk.Label(vad_frame,text="音色：")
        speaker_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.speaker_var = tk.StringVar(value="中文女")
        speaker_combobox = ttk.Combobox(vad_frame, textvariable=self.speaker_var,
                                        values=TTSConfig.supported_speakers,state="readonly")
        speaker_combobox.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        max_record_label = ttk.Label(vad_frame, text="最大录音时长-秒：", font=self.font_normal)
        max_record_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_record_var = tk.StringVar(value="60")
        max_record_entry = ttk.Entry(vad_frame, textvariable=self.max_record_var, width=10)
        max_record_entry.grid(row=1, column=1, padx=5, pady=5)

        # 新增：有效语音检测阈值
        voice_threshold_label = ttk.Label(vad_frame, text="有效语音阈值：", font=self.font_normal)
        voice_threshold_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.voice_threshold_var = tk.StringVar(value="0.003")  # 默认值
        voice_threshold_entry = ttk.Entry(vad_frame, textvariable=self.voice_threshold_var, width=10)
        voice_threshold_entry.grid(row=2, column=1, padx=5, pady=5)
        # 可选：添加说明标签
        tip_label = ttk.Label(vad_frame, text="(建议0.0001-0.01，环境嘈杂时可适当调高)", font=("Microsoft YaHei", 8), foreground="gray")
        tip_label.grid(row=2, column=3, padx=5, pady=5)

        # 保存配置按钮
        save_btn = ttk.Button(config_frame, text="保存所有配置", command=self._save_config)
        save_btn.pack(fill=tk.X, padx=5, pady=10)

        # -------------------------- 右侧功能面板 -----------------------------------
        func_frame = ttk.Frame(main_panel)
        main_panel.add(func_frame, weight=2)

        # 标题栏
        title_frame = ttk.Frame(func_frame)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_label = ttk.Label(title_frame, text="千语 - ASR+LLM+TTS 全链路语音助手", font=self.font_title)
        title_label.pack(side=tk.LEFT)

        # 聊天记录区
        chat_frame = ttk.Frame(func_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        chat_label = ttk.Label(chat_frame, text="聊天记录：", font=self.font_bold)
        chat_label.pack(anchor=tk.W)
        self.chat_text = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=self.font_normal)
        self.chat_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_text.config(state=tk.DISABLED)

        # 思考过程展示区域
        thinking_frame = ttk.Frame(func_frame)
        thinking_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        thinking_label = ttk.Label(thinking_frame, text="思考过程：", font=self.font_bold)
        thinking_label.pack(anchor=tk.W)
        self.thinking_text = scrolledtext.ScrolledText(thinking_frame,height=8,state=tk.DISABLED)
        self.thinking_text.pack(fill=tk.BOTH, padx=5)

        # 状态提示区
        status_frame = ttk.Frame(func_frame)
        status_frame.pack(fill=tk.X, pady=5)
        status_label = ttk.Label(status_frame, text="状态：", font=self.font_bold)
        status_label.pack(side=tk.LEFT)
        self.status_text = ttk.Label(status_frame, text="未加载模块", font=self.font_normal, foreground="gray")
        self.status_text.pack(side=tk.LEFT, padx=5)

        # 按钮区
        btn_frame = ttk.Frame(func_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.load_btn = ttk.Button(btn_frame, text="加载所有模块", command=self._load_modules)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.record_btn = ttk.Button(btn_frame, text="🎙️ 开始录音", command=self._start_chat, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="清空记录", command=self._clear_chat)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        exit_btn = ttk.Button(btn_frame, text="退出程序", command=self._exit_app)
        exit_btn.pack(side=tk.RIGHT, padx=5)

    def _browse_dir(self, var):
        """浏览文件夹（用于模型路径选择）"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            var.set(dir_path)

    def _init_assistant(self):
        """初始化语音助手"""
        # 自定义配置（从UI读取初始值）
        asr_config = ASRConfig(
            model_path=self.asr_model_var.get(),
            vad_model_path=self.asr_vad_var.get(),
            default_record_duration=15,
            language="zh_cn",
            silence_timeout=3,
            max_record_duration=60,
            voice_detection_threshold=float(self.voice_threshold_var.get())
        )

        # 处理LLM max_tokens（空值为None）
        try:
            llm_max_tokens = int(self.llm_tokens_var.get()) if self.llm_tokens_var.get().strip() else None
        except ValueError:
            llm_max_tokens = None

        llm_config = LLMConfig(
            base_url=self.llm_url_var.get(),
            model_alias=self.llm_alias_var.get(),
            max_history=int(self.history_var.get()),
            system_prompt=self.prompt_text.get(1.0, tk.END).strip(),
            temperature=float(self.llm_temp_var.get()),
            max_tokens=llm_max_tokens
        )

        tts_config = TTSConfig(
            model_path=self.tts_model_var.get(),
            default_speaker="中文女",
            min_segment_length=15
        )

        # 创建语音助手实例
        self.assistant = VoiceAssistant(
            asr_config=asr_config,
            llm_config=llm_config,
            tts_config=tts_config,
            record_duration=5,
            ui_callback=self._update_ui
        )

    def _save_config(self):
        """保存所有配置参数"""
        try:
            # 1. 读取基础配置
            system_prompt = self.prompt_text.get(1.0, tk.END).strip()
            silence_timeout = float(self.silence_var.get())
            max_record_duration = int(self.max_record_var.get())
            current_speaker = self.speaker_var.get().strip()
            max_history = int(self.history_var.get())
            voice_threshold = float(self.voice_threshold_var.get())

            # 2. 读取模型路径（验证是否为文件夹）
            asr_model_path = self.asr_model_var.get().strip()
            asr_vad_path = self.asr_vad_var.get().strip()
            tts_model_path = self.tts_model_var.get().strip()

            # 3. 读取LLM参数
            llm_base_url = self.llm_url_var.get().strip()
            llm_model_alias = self.llm_alias_var.get().strip()
            llm_temperature = float(self.llm_temp_var.get())
            llm_top_p = float(self.llm_top_p_var.get())
            llm_max_tokens = int(self.llm_tokens_var.get()) if self.llm_tokens_var.get().strip() else None
            llm_thinking = self.llm_thinking_var.get()

            # 4. 参数验证
            error_msg = ""
            if silence_timeout <= 0:
                error_msg = "静音检测阈值必须大于0！"
            elif max_record_duration <= 0:
                error_msg = "最大录音时长必须大于0！"
            elif voice_threshold < 0:
                error_msg = "有效语音阈值必须大于等于0！"
            elif max_history <= 0:
                error_msg = "最大对话轮次必须大于0！"
            elif asr_model_path and not os.path.isdir(asr_model_path):
                error_msg = f"ASR模型路径不是有效文件夹：{asr_model_path}"
            elif asr_vad_path and not os.path.isdir(asr_vad_path):
                error_msg = f"ASR VAD路径不是有效文件夹：{asr_vad_path}"
            elif tts_model_path and not os.path.isdir(tts_model_path):
                error_msg = f"TTS模型路径不是有效文件夹：{tts_model_path}"
            elif not (0 <= llm_temperature <= 2):
                error_msg = "温度系数必须在0-2之间！"
            elif not (0 <= llm_top_p <= 1):
                error_msg = "top_p必须在0-1之间！"
            elif llm_max_tokens is not None and llm_max_tokens <= 0:
                error_msg = "最大Token数必须大于0（留空为无限制）！"

            if error_msg:
                self._update_ui("error", error_msg)
                return

            # 5. 更新配置
            self.assistant.update_configs(
                system_prompt=system_prompt,
                silence_timeout=silence_timeout,
                max_record_duration=max_record_duration,
                voice_threshold=voice_threshold,
                tts_speaker=current_speaker,
                max_history=max_history,
                asr_model_path=asr_model_path,
                asr_vad_path=asr_vad_path,
                tts_model_path=tts_model_path,
                llm_base_url=llm_base_url,
                llm_model_alias=llm_model_alias,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_max_tokens=llm_max_tokens,
                llm_thinking=llm_thinking
            )
        except ValueError as e:
            self._update_ui("error", f"参数格式错误：{str(e)}（请检查数字类型参数）")
        except Exception as e:
            self._update_ui("error", f"保存配置失败: {str(e)}")

    def _update_ui(self, msg_type, content):
        """UI更新回调函数"""
        self.root.after(0, self._safe_update_ui, msg_type, content)

    def _safe_update_ui(self, msg_type, content):
        """线程安全的UI更新"""
        if msg_type == "status":
            self.status_text.config(text=content)
            if "所有模块加载完成" in content:
                self.record_btn.config(state=tk.NORMAL)
                self.load_btn.config(state=tk.DISABLED)
        elif msg_type == "asr":
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.insert(tk.END, f"{content}\n", "asr")
            self.chat_text.see(tk.END)
            self.chat_text.config(state=tk.DISABLED)
        elif msg_type == "llm":
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.insert(tk.END, f"{content}\n\n", "llm")
            self.chat_text.see(tk.END)
            self.chat_text.config(state=tk.DISABLED)
        elif msg_type == "error":
            self.status_text.config(text=content, foreground="red")
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.insert(tk.END, f"❌ {content}\n", "error")
            self.chat_text.see(tk.END)
            self.chat_text.config(state=tk.DISABLED)
        elif msg_type == "llm_thinking":
            self.thinking_text.config(state=tk.NORMAL)
            self.thinking_text.delete(1.0, tk.END)
            self.thinking_text.insert(tk.END, f"{content}\n" if content else "无思考过程", "llm_thinking")
            self.thinking_text.config(state=tk.DISABLED)

        # 设置文本样式
        self.chat_text.tag_config("asr", foreground="#0066cc", font=self.font_normal)
        self.chat_text.tag_config("llm", foreground="#cc3300", font=self.font_normal)
        self.chat_text.tag_config("error", foreground="red", font=self.font_normal)

    def _load_modules(self):
        """加载所有模块"""
        if self.is_loading:
            return
        self.is_loading = True
        self.load_btn.config(state=tk.DISABLED)

        def load_thread():
            self.assistant.load_all_modules()
            self.is_loading = False

        load_thread = threading.Thread(target=load_thread)
        load_thread.daemon = True
        load_thread.start()

    def _start_chat(self):
        """开始语音对话"""
        self.record_btn.config(state=tk.DISABLED)
        self.assistant.voice_chat()
        self.root.after(1000, lambda: self.record_btn.config(state=tk.NORMAL))

    def _clear_chat(self):
        """清空聊天记录"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.config(state=tk.DISABLED)
        self.assistant.clear_chat_history()

    def _exit_app(self):
        """退出程序"""
        if self.assistant:
            self.assistant.unload_all_modules()
        self.root.quit()


# -------------------------- 主程序入口 --------------------------
def main():
    """Tkinter UI主程序"""
    root = tk.Tk()
    app = VoiceAssistantUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._exit_app()


if __name__ == "__main__":
    main()