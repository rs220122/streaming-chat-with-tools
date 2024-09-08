"""
Author: your name
Date: 2024-09-08 23:26:39
"""

# third-party packages

# user-defined packages
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]
tool_dict = {"add": add, "multiply": multiply}


if __name__ == "__main__":
    pass
