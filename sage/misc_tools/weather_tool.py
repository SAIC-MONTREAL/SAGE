"""Weather tool"""
import os
from dataclasses import dataclass, field
from typing import Any, List, Type

from langchain.utilities import OpenWeatherMapAPIWrapper
from sage.base import SAGEBaseTool, BaseToolConfig


@dataclass
class WeatherToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: WeatherTool)
    name: str = "weather_tool"
    description: str = "Use this tool to get weather information of a given place. The input to this tool should be a string in this format 'City, Country'"


class WeatherTool(SAGEBaseTool):
    weather: OpenWeatherMapAPIWrapper = OpenWeatherMapAPIWrapper()

    def _run(self, command: str) -> Any:

        try:
            return self.weather.run(command)
        except Exception:
            return "Invalid input. The input to this tool should be a string in this format 'City, Country'. Try again"
