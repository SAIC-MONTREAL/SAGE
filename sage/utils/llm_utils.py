"""Util functions to handle LLMs"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Type
import datetime

from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from langchain import HuggingFaceTextGenInference
from langchain.schema.messages import HumanMessage
from langchain.chat_models import ChatAnthropic

from sage.base import BaseConfig


@dataclass
class LLMConfig(BaseConfig):
    """Base LLM configuration"""

    _target: Type = None

    def instantiate(self):
        kwargs = vars(self).copy()
        kwargs.pop("_target")

        return self._target(**kwargs)


@dataclass
class GPTConfig(LLMConfig):
    """Configuration of Open AI llms"""

    _target: Type = field(default_factory=lambda: ChatOpenAI)

    model_name: str = "gpt-4"
    temperature: float = 0.0
    streaming: bool = True
    request_timeout: int = 600
    max_tokens: int = 1000
    n: int = 1
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeConfig(LLMConfig):
    """Configuration of Anthropic llms"""

    _target: Type = field(default_factory=lambda: ChatAnthropic)
    model_name: str = "claude-2"
    temperature: float = 0.0
    anthropic_api_key: str = os.environ["ANTHROPIC_API_KEY"]
    default_request_timeout: int = 10200
    max_tokens_to_sample: int = 5000
    streaming: bool = True


@dataclass
class TGIConfig(LLMConfig):
    """Configuration of open source llms hosted using text generation inferance API"""

    _target: Type = field(default_factory=lambda: HuggingFaceTextGenInference)
    inference_server_url: str = "http://105.160.143.23:8080/"
    max_new_tokens: int = 500
    temperature: float = 0.01
    stop_sequences: List[str] = field(default_factory=lambda: [])


def make_chatgpt_request(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop_sequences: List[str],
    n: int,
) -> Dict[str, Any]:

    """Given a prompt, sends a request to ChatGPT"""

    model_kwargs = {
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop_sequences,
    }

    llm_config = GPTConfig(
        model_name="gpt-4",
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        model_kwargs=model_kwargs,
    )
    llm = llm_config.instantiate()

    message = HumanMessage(content=prompt)
    response = llm([message])

    return {
        "prompt": prompt,
        "response": response,
        "created_at": str(datetime.datetime.now()),
    }
