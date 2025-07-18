# agents.py

from typing import List, Optional, Any, AsyncGenerator, Dict
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import AsyncOpenAI


class Tool:
    name: str
    description: str
    input_type: type

    async def call(self, input: Any) -> str:
        raise NotImplementedError("Tool must implement the 'call' method")


class OpenAIChatCompletionsModel:
    def __init__(self, model: str, openai_client: AsyncOpenAI):
        self.model = model
        self.client = openai_client

    async def chat_stream(
        self, messages: List[ChatCompletionMessageParam]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            yield chunk


class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        model: OpenAIChatCompletionsModel,
        tools: Optional[List[Tool]] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = {tool.name: tool for tool in tools or []}

    def get_system_message(self) -> Dict:
        return {
            "role": "system",
            "content": self.instructions,
        }


class Runner:
    @staticmethod
    def run_streamed(starting_agent: Agent, input: str):
        class Result:
            def __init__(self, agent: Agent, input: str):
                self.agent = agent
                self.input = input

            async def stream_events(self):
                messages = [
                    self.agent.get_system_message(),
                    {"role": "user", "content": self.input},
                ]
                async for chunk in self.agent.model.chat_stream(messages):
                    delta_content = chunk.choices[0].delta.content or ""
                    yield RawResponseEvent(delta=delta_content)

        return Result(starting_agent, input)


class RawResponseEvent:
    def __init__(self, delta: str):
        self.type = "raw_response_event"
        self.data = ResponseTextDeltaEvent(delta=delta)


class ResponseTextDeltaEvent:
    def __init__(self, delta: str):
        self.delta = delta
