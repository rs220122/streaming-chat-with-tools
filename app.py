"""
Author: your name
Date: 2024-09-07 20:20:24
"""

# third-party packages
import json
from typing import Callable

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import (
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI

# user-defined packages
from models import AIResponse, ChatParams, ToolResponse, UsageData
from tools import tool_dict, tools

load_dotenv()

system_message = SystemMessage(
    "You are a AI Assistant. 返事はなるべくフレンドリーに返事してください。"
)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


def tool_call_from_ai_message(
    tool_dict: dict[str, Callable], tool_chunks: AIMessageChunk, messages: list
):
    messages_temp = messages.copy()

    usage_data = UsageData(
        input_tokens=tool_chunks.usage_metadata["input_tokens"],
        output_tokens=tool_chunks.usage_metadata["output_tokens"],
    )
    response_list = []
    for tool_info in tool_chunks.tool_calls:
        print(f'CALL: {tool_info["name"]}, {tool_info["args"]}')
        tool_output = tool_dict[tool_info["name"].lower()].invoke(tool_info["args"])
        messages_temp.append(ToolMessage(tool_output, tool_call_id=tool_info["id"]))

        response_list.append(
            ToolResponse(
                id=tool_info["id"],
                tool_name=tool_info["name"],
                tool_args=json.dumps(tool_info["args"]),
                result=str(tool_output),
                usage_data=usage_data,
            )
        )
    return messages_temp, response_list


async def streaming(chat_params: ChatParams):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=chat_params.temperature,
        max_tokens=chat_params.max_tokens,
    ).bind_tools(tools)

    trimmer = trim_messages(
        max_tokens=10000,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
    )

    messages = chat_params.lc_messages
    messages = [system_message] + messages
    is_answer = False

    while not is_answer:
        aggregate = None
        is_finish = False
        finish_reason = None
        async for chunk in llm.astream(trimmer.invoke(messages), stream_usage=True):
            aggregate = chunk if aggregate is None else aggregate + chunk
            if is_finish:
                messages.append(aggregate)
                if finish_reason == "tool_calls":
                    print("Tool Calling Start!!")
                    messages, tool_responses = tool_call_from_ai_message(
                        tool_dict, aggregate, messages
                    )

                    for t in tool_responses:
                        yield t.format_sse()
                else:
                    print()
                    print(f"finish_reason: {finish_reason}")
                    is_answer = True
                    usage_data = UsageData(
                        input_tokens=aggregate.usage_metadata["input_tokens"],
                        output_tokens=aggregate.usage_metadata["output_tokens"],
                    )
                    yield AIResponse(
                        content="",
                        is_done=True,
                        finish_reason=finish_reason,
                        usage_data=usage_data,
                    ).format_sse()
                break

            if "finish_reason" in chunk.response_metadata:
                finish_reason = chunk.response_metadata["finish_reason"]
                is_finish = True

            if len(chunk.content) == 0 and len(chunk.tool_call_chunks) == 0:
                continue

            if len(chunk.tool_call_chunks) > 0:
                continue

            # AI回答文
            yield AIResponse(
                content=chunk.content,
                is_done=False,
                finish_reason=None,
            ).format_sse()

        if is_answer:
            break


@app.post("/chat")
async def chat(request: Request, chat_params: ChatParams):

    print(chat_params.model_dump_json())

    for c in chat_params.lc_messages:
        print(c)

    return StreamingResponse(streaming(chat_params=chat_params))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
