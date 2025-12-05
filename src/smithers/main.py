"""Entry point for the LangChain-based application."""

import asyncio

from .chain import build_chain


async def main() -> None:
    """Run simple invocations against the LangChain RAG chain."""
    chain = build_chain()

    print("Task 1: What is 2 + 2?")
    response1 = chain.invoke({"question": "What is 2 + 2?"})
    print(f"Response: {response1}\n")

    print("Task 2: Explain Smithers project briefly.")
    response2 = chain.invoke({"question": "Explain the Smithers project briefly."})
    print(f"Response: {response2}\n")


if __name__ == "__main__":
    asyncio.run(main())
