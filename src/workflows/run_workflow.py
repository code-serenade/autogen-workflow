import asyncio
import json
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from utils.model_loader import get_model_client

# === 默认路径 ===
DEFAULT_INPUT_PATH = "data/input.jsonl"
DEFAULT_PROMPT_PATH = "data/prompt.txt"
DEFAULT_OUTPUT_MD = "data/output/output.md"
DEFAULT_OUTPUT_JSONL = "data/output/output.jsonl"

# === 工具函数 ===


def load_prompt(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return input("请输入提示词：")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_markdown(content, output_path, item_id=None):
    with open(output_path, "a", encoding="utf-8") as f:
        if item_id is not None:
            f.write(f"\n\n## 扩写结果 {item_id}\n")
        f.write(content + "\n")
        f.write("\n---\n")
    print(f"✅ 第 {item_id} 条内容已追加保存到 {output_path}")


def append_jsonl(data: dict, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def log_and_save_stream(stream, item_id, md_path, jsonl_path):
    last_response = None
    async for event in stream:
        if getattr(event, "type", None) == "TextMessage":
            print(f"🗣️ {event.source}: {event.content}")
            if event.source == "assistant":
                last_response = event.content

    if last_response:
        append_markdown(last_response, md_path, item_id=item_id)
        append_jsonl({"id": item_id, "output": last_response}, jsonl_path)
    else:
        print(f"❌ 第 {item_id} 条数据未生成内容")


# === 主逻辑 ===
async def main(input_path, prompt_path, output_md, output_jsonl):
    model_client = get_model_client(provider="ollama", model="llama3.2")
    prompt = load_prompt(prompt_path)
    items = load_jsonl(input_path)

    for idx, item in enumerate(items):
        item_id = item.get("id", idx + 1)
        print(f"\n🚀 正在处理第 {idx + 1} 条数据（ID: {item_id}）...\n")

        task = f"""以下是我的提示词：
{prompt}

以下是原始语料：
{item['text']}

请基于提示词，对语料进行内容丰富、风格一致的扩写，尽可能提升内容深度和表达质量。"""

        assistant = AssistantAgent("assistant", model_client=model_client)
        user_proxy = UserProxyAgent("user", input_func=input)
        termination = TextMentionTermination("APPROVE")

        team = RoundRobinGroupChat(
            [assistant, user_proxy], termination_condition=termination
        )
        stream = team.run_stream(task=task)

        await log_and_save_stream(
            stream, item_id=item_id, md_path=output_md, jsonl_path=output_jsonl
        )

    await model_client.close()
    print("🎉 全部处理完毕！")


# === CLI ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autogen 多条语料扩写工作流")

    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_PATH, help="输入 .jsonl 文件路径"
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT_PATH, help="提示词文件路径"
    )
    parser.add_argument(
        "--output-md", type=str, default=DEFAULT_OUTPUT_MD, help="Markdown 输出文件路径"
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=DEFAULT_OUTPUT_JSONL,
        help="结构化输出 jsonl 路径",
    )

    args = parser.parse_args()

    asyncio.run(main(args.input, args.prompt, args.output_md, args.output_jsonl))
