<img width="1282" height="832" alt="image" src="https://github.com/user-attachments/assets/76803b49-5645-4aa0-9f68-f6bd7aff6e8f" /># 项目依赖说明

小白写的项目，conda环境配置见enviroment.yaml，torch的版本可能会不匹配，按需更改。
模型方面LLM用的接口，ASR和TTS需要自行下载模型。

## 模型说明

### 1. Sense-Voice-Small
语音识别模型，用于语音转文字核心能力：
- ModelScope 地址：[iic/SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
- HuggingFace 地址：[FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

### 2. OpenAI 兼容 API 接口
需部署/接入符合 OpenAI 接口规范的 API 服务，用于相关功能的调用。

### 3. CosyVoice-300M-SFT
语音合成模型，用于文字转语音核心能力：
- ModelScope 地址：[iic/CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)

### 4. speech_fsmn_vad_zh-cn-16k-common-pytorch
语音活动检测（VAD）模型，用于语音端点检测：
- ModelScope 地址：[iic/speech_fsmn_vad_zh-cn-16k-common-pytorch](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
