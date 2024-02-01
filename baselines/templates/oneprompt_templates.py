from typing import Dict
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate


class OnePromptResponse(BaseModel):
    diff: Dict = Field(
        description="A JSON of all of the changed device states. If you are not asked to change device states, return an empty dict."
    )
    output: str = Field(description="A natural language response to the user query.")


oneprompt_template = """
You are an AI that controls a smart home. You receive a user command and the state
of all devices (in JSON) format, and then assign settings to devices in response.

user command: {command}
devices: {device_state}
TV guide: {tv_guide}

{format_instructions}
"""

# input template to llm
oneprompt_prompt_template = PromptTemplate(
    template=oneprompt_template,
    input_variables=["command", "device_state", "tv_guide"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=OnePromptResponse
        ).get_format_instructions()
    },
)
