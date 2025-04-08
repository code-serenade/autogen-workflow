import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from docx import Document
from markdownify import markdownify as md

from utils.model_loader import get_model_client

# === llm client ===
model_client = get_model_client(provider="ollama", model="llama3.2")

DEFAULT_INPUT_PATH = "/Users/ancient/data/autogen/input.docx"
DEFAULT_PROMPT_PATH = "/Users/ancient/data/autogen/prompt.txt"
DEFAULT_OUTPUT_PATH = "/Users/ancient/data/autogen/output.md"


# === 工具函数 ===
def docx_to_markdown(docx_path):
    doc = Document(docx_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return md(text)


def load_prompt(prompt_file=None):
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return input("请输入提示词：")


def save_markdown(output_text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"✅ 扩写结果已保存至 {output_path}")


async def log_and_save_stream(stream):
    history = []
    last_response = None

    async for event in stream:
        if getattr(event, "type", None) == "TextMessage":
            print(f"🗣️ {event.source}: {event.content}")
            history.append({"sender": event.source, "content": event.content})

            if event.source == "assistant":
                last_response = event.content

    return last_response, history


async def main(docx_path, prompt_path, output_path):
    prompt = load_prompt(prompt_path)
    raw_markdown = docx_to_markdown(docx_path)

    print("\n📄 原始语料（已转 Markdown）：\n")
    print(raw_markdown[:500] + "..." if len(raw_markdown) > 500 else raw_markdown)

    task = f"""以下是我的提示词：
    {prompt}

    以下是原始语料（Markdown 格式）：
    {raw_markdown}

    请基于提示词，对语料进行内容丰富、风格一致的扩写，尽可能提升内容深度和表达质量。"""

    # === Agent + Client ===
    assistant = AssistantAgent("assistant", model_client=model_client)

    user_proxy = UserProxyAgent("user_proxy", input_func=input)  # Console输入
    termination = TextMentionTermination("APPROVE")  # 你手动输入 APPROVE 即结束

    team = RoundRobinGroupChat(
        [assistant, user_proxy], termination_condition=termination
    )

    stream = team.run_stream(task=task)

    last_response, _history = await log_and_save_stream(stream)

    # await Console(stream)  # 控制台流输出
    await model_client.close()

    if last_response:
        save_markdown(last_response, output_path)
    else:
        print("❌ 没有捕获到 Assistant 的输出内容。")


# === CLI 调用示例 ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autogen LLM 写作扩写工作流")

    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="上传的 Word 文件路径（.docx）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT_PATH,
        help="提示词文本文件路径，可选",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="输出 Markdown 文件路径",
    )

    args = parser.parse_args()
    asyncio.run(main(args.file, args.prompt, args.output))
