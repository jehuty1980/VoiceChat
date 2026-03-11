import openai
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import logging


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


# 初始化LLM模块logger
logger = setup_logger("LLMModule")


# -------------------------- 配置类（统一管理参数） --------------------------
@dataclass
class LLMConfig:
    """LLM问答模块配置类，集中管理所有参数"""
    # 服务端配置
    base_url: str = "http://127.0.0.1:8080/v1"  # llama-server地址
    model_alias: str = "qwen35"  # 模型别名（对应--alias参数）
    api_key: str = "sk-no-key-required"  # 本地服务占位符
    # 对话配置
    max_history: int = 20  # 最大对话历史轮数
    system_prompt: str = "你叫千语，是我的个人助手，每次回复不超过250字。"
    # 接口调用配置
    thinking: bool = False # 思考模式，默认关闭
    timeout: int = 30  # 接口超时时间（秒）
    temperature: float = 0.7  # 生成温度（可调整随机性）
    top_p: float = 0.95
    max_tokens: Optional[int] = None  # 最大生成token数（None则使用模型默认）


# -------------------------- LLM核心模块类 --------------------------
class LlamaServerLLM:
    """对接llama-server的LLM问答模块，支持多轮会话、会话隔离"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM模块
        :param config: LLM配置对象，若为None则使用默认配置
        """
        self.config = config or LLMConfig()
        self.client: Optional[openai.OpenAI] = None  # 延迟初始化客户端
        # 支持多会话管理（key: 会话ID, value: 对话历史）
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.default_session_id = "default_session"  # 默认会话ID

    def _init_client(self) -> None:
        """初始化OpenAI客户端（延迟初始化）"""
        if self.client is not None:
            logger.warning("LLM客户端已初始化，无需重复创建")
            return

        try:
            logger.info(f"初始化LLM客户端，服务地址: {self.config.base_url}")
            self.client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout  # 设置超时时间
            )
            logger.info("LLM客户端初始化成功")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {str(e)}", exc_info=True)
            raise

    def _get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取指定会话的历史记录，不存在则初始化"""
        if session_id not in self.conversations:
            # 初始化会话历史（包含系统提示）
            self.conversations[session_id] = [
                {"role": "system", "content": self.config.system_prompt}
            ]
            logger.debug(f"初始化新会话: {session_id}")
        return self.conversations[session_id]

    def _trim_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """修剪对话历史，保留系统消息 + 最近max_history轮对话"""
        # 系统消息(1条) + 轮数*2（user+assistant）
        max_len = 1 + 2 * self.config.max_history
        if len(history) > max_len:
            # 保留系统消息，截断更早的对话
            trimmed_history = [history[0]] + history[-2 * self.config.max_history:]
            logger.debug(f"修剪对话历史，原长度{len(history)} → 新长度{len(trimmed_history)}")
            return trimmed_history
        return history

    def chat(self, message: str, session_id: Optional[str] = None) -> tuple[str, str]:
        """
        核心问答接口：接收用户输入，返回模型回复（维护多轮上下文）
        :param message: 用户输入文本
        :param session_id: 会话ID，None则使用默认会话
        :return: 模型生成的回复文本
        """
        # 初始化客户端（按需）
        if self.client is None:
            self._init_client()

        # 确定会话ID
        target_session = session_id or self.default_session_id
        # 获取/初始化会话历史
        history = self._get_session_history(target_session)

        # 空输入校验
        if not message.strip():
            logger.warning(f"会话{target_session}收到空输入")
            return "请输入有效的问题或指令！",""

        # 添加用户输入到历史
        history.append({"role": "user", "content": message.strip()})
        # 修剪历史（控制长度）
        history = self._trim_history(history)
        self.conversations[target_session] = history

        try:
            logger.info(f"会话{target_session}调用LLM生成回复，输入长度: {len(message)}字")
            # 调用llama-server接口
            completion = self.client.chat.completions.create(
                model=self.config.model_alias,
                messages=history,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.config.thinking},},
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens
            )

            # 解析回复：思考过程
            response_dict = completion.model_dump()
            thinking_content = response_dict["choices"][0]["message"].get("reasoning_content", "").strip()

            # 解析回复：回复内容
            response = completion.choices[0].message.content.strip()
            # 添加模型回复到历史
            history.append({"role": "assistant", "content": response})
            self.conversations[target_session] = history

            logger.info(f"会话{target_session}LLM回复生成完成，回复长度: {len(response)}字")
            return response,thinking_content

        except Exception as e:
            error_msg = f"LLM接口调用失败: {str(e)}"
            logger.error(f"会话{target_session}{error_msg}", exc_info=True)
            # 出错时清空当前会话历史（避免错误上下文累积）
            self.clear_session(target_session)
            raise Exception(error_msg)

    def clear_session(self, session_id: Optional[str] = None) -> None:
        """
        清空指定会话的历史记录（保留系统提示）
        :param session_id: 会话ID，None则清空默认会话
        """
        target_session = session_id or self.default_session_id
        if target_session in self.conversations:
            self.conversations[target_session] = [
                {"role": "system", "content": self.config.system_prompt}
            ]
            logger.info(f"已清空会话{target_session}的对话历史")
        else:
            logger.warning(f"会话{target_session}不存在，无需清空")

    def remove_session(self, session_id: str) -> None:
        """删除指定会话（释放资源）"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"已删除会话: {session_id}")
        else:
            logger.warning(f"会话{session_id}不存在，无需删除")

    def close(self) -> None:
        """关闭客户端，释放资源"""
        if self.client is not None:
            del self.client
            self.client = None
            logger.info("LLM客户端已关闭")


# -------------------------- 模块使用示例 --------------------------
def main():
    """LLM模块测试入口"""
    # 1. 初始化配置（可自定义参数）
    custom_config = LLMConfig(
        base_url="http://127.0.0.1:8080/v1",
        model_alias="qwen35",
        max_history=20,
        system_prompt="你叫千语，是我的个人助手，每次回复不超过250字。"
    )

    # 2. 初始化LLM模块
    llm_module = LlamaServerLLM(custom_config)

    try:
        logger.info("\n=== Llama-Server LLM模块测试 ===")
        logger.info(f"服务地址: {custom_config.base_url} | 模型别名: {custom_config.model_alias}")
        logger.info("输入 'exit'/'quit' 退出 | 输入 'clear' 清空历史 | 输入 'remove' 删除当前会话")
        logger.info("-" * 60)

        while True:
            user_input = input("\n你: ").strip()

            # 退出条件
            if user_input.lower() in ["exit", "quit"]:
                print("AI: 再见！")
                break

            # 清空历史
            if user_input.lower() == "clear":
                llm_module.clear_session()
                print("AI: 已清空对话历史！")
                continue

            # 删除会话
            if user_input.lower() == "remove":
                llm_module.remove_session(llm_module.default_session_id)
                print("AI: 已删除当前会话！")
                continue

            # 空输入处理
            if not user_input:
                print("AI: 请输入有效的问题或指令！")
                continue

            # 生成回复
            try:
                print("AI: ", end="", flush=True)
                response, reasoning = llm_module.chat(user_input)
                if reasoning != "":
                    print('-'*40,'reasoning','-'*40)
                    print(reasoning)
                    print('-'*40,'response','-'*41)
                print(response)
            except Exception as e:
                print(f"\n出错了: {str(e)}")

    except KeyboardInterrupt:
        logger.info("\n用户终止程序，开始清理资源...")
    except Exception as e:
        logger.error(f"程序异常: {str(e)}", exc_info=True)
    finally:
        # 关闭客户端
        llm_module.close()
        logger.info("程序正常退出")


if __name__ == "__main__":
    main()