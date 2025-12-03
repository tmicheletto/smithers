"""Entry point for the agentic application."""

import asyncio

from .agent import process_task
from .models import TaskRequest


async def main() -> None:
    """Run the agentic application with example tasks."""
    # Example 1: Simple task request
    task1 = TaskRequest(
        task="What is 2 + 2?",
        metadata={"example": "basic_math"},
    )

    print("Task 1:", task1.task)
    response1 = await process_task(task1)
    print(f"Response: {response1.result}\n")

    # Example 2: Text processing task
    task2 = TaskRequest(
        task="Process this text: 'Hello, World! This is an example.'",
        metadata={"example": "text_processing"},
    )

    print("Task 2:", task2.task)
    response2 = await process_task(task2)
    print(f"Response: {response2.result}\n")


if __name__ == "__main__":
    asyncio.run(main())
