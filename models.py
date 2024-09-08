"""
Author: your name
Date: 2024-09-08 23:27:34
"""

import uuid
from typing import Literal, Optional

# third-party packages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, model_validator

# user-defined packages


class Message(BaseModel):
    content: str
    role: Literal["user", "assistant", "tool"]
    id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None

    @model_validator(mode="after")
    def validate_tool(self) -> "Message":
        if self.role in ["user", "assistant"]:
            return self

        # id, tool_name, tool_argsの中で、Noneとなっている変数名をいかに追加する
        missing_vals = []
        if self.id is None:
            missing_vals.append("id")
        elif self.tool_name is None:
            missing_vals.append("tool_name")
        elif self.tool_args is None:
            missing_vals.append("tool_args")

        if len(missing_vals) > 0:
            raise ValueError(
                "When role is tool, you must set id, tool_name and tool_args"
            )
        return self


class ChatParams(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(default=100, ge=100, le=1000)
    temperature: float = Field(deafult=0.7, ge=0.0, le=1.0)

    @property
    def lc_messages(self) -> list[BaseMessage]:
        m_to_cl = {"user": HumanMessage, "assistant": AIMessage}
        mm = []

        tools: list[Message] = []
        for m in self.messages:
            if m.role in ["user", "assistant"]:
                if len(tools) > 0:
                    # tools を追加する
                    # AIMessageは、AIがtools_callingを指定する履歴を格納する
                    # その後に、実際に指定した関数を呼び出した後の結果を格納する
                    tool_call_list = [
                        {
                            "id": t.id,
                            "type": "function",
                            "function": {"arguments": t.tool_args, "name": t.tool_name},
                        }
                        for t in tools
                    ]
                    tool_call_lists = [
                        {
                            "id": t.id,
                            "type": "tool_call",
                            "name": t.tool_name,
                            "args": t.tool_args,
                        }
                        for t in tools
                    ]
                    mm.append(
                        AIMessage(
                            content="",
                            # addional_kwargs={"tool_calls": tool_call_list},
                            tool_calls=tool_call_lists,
                        )
                    )
                    for t in tools:
                        mm.append(ToolMessage(content=t.content, tool_call_id=t.id))
                    tools = []
                mm.append(m_to_cl[m.role](m.content))
            elif m.role == "tool":
                tools.append(m)
        return mm


class UsageData(BaseModel):
    input_tokens: int
    output_tokens: int


class StreamModel(BaseModel):

    def format_sse(self):
        return f"data: {self.model_dump_json(indent=2)}\n\n"


class ToolResponse(StreamModel):
    id: str
    tool_name: str
    tool_args: str
    result: str
    usage_data: UsageData
    is_done: Literal[False] = False
    role: Literal["tool"] = "tool"


class AIResponse(StreamModel):
    content: str
    is_done: bool
    finish_reason: Optional[str] = None
    usage_data: Optional[UsageData] = None
    id: Optional[str] = None
    role: Literal["assistant"] = "assistant"

    @model_validator(mode="after")
    def _post_init(self) -> "AIResponse":
        if self.is_done:
            self.id = str(uuid.uuid4())

        return self


if __name__ == "__main__":
    pass
