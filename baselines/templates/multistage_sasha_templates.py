from langchain import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Dict


### Template strings ###


# Multistage template strings
# Define your desired data structure.
class ClarificationResponse(BaseModel):
    response: str = Field(
        description="Answer by YES or NO if the task is possible or the query can be answered"
    )
    explanation: str = Field(
        description="Return an explanation of the response to the user"
    )


class FilteringResponse(BaseModel):
    devices: list[str] = Field(
        description="A list that contains the devices that can potentially accomplish the task"
    )


class PersistentResponse(BaseModel):
    trigger: list = Field(
        description="Given the dictionary names devices, return the list of keys to extract the value of interest in the nested dictionary. Example: Input: {'device_id': {main':{'temperatureMeasurement':{'temperatureMeasurement': {'value':5}}}}  Output:  '['device_id', 'main', 'temperatureMeasurement', 'temperatureMeasurement', 'value']'"
    )
    trigger_value: str = Field(
        description="The value that the trigger must have in order for the action to occur"
    )
    output: str = Field(
        description="Explain what the agent should do once the trigger is activated."
    )
    explanation: str = Field(
        description="Provide a natural language explanation of why trigger and action values were chosen."
    )


class PrePlanningResponse(BaseModel):
    response: str = Field(
        description="Respond, IN ONE WORD, whether the instruction is associated with sensor (the retrieval of information from a sensor), control (the control of a device), or persistent (complex goals that demand the creation of automation routines)."
    )


class PlanningResponse(BaseModel):
    devices: Dict = Field(
        description="Return a diff json file that follows the structure of FILTERED DEVICES and specifies which values of the dictionary should be changed to best accomplish the USER COMMAND."
    )


class ReadingResponse(BaseModel):
    output: str = Field(
        description="Return a diff json file that follows the structure of FILTERED DEVICES and specifies which values of the dictionary should be changed to best accomplish the USER COMMAND."
    )


clarification_template = """
You are an AI that controls a smart home.
You will be provided with a set of home appliances and an instruction.
Respond, IN ONE WORD, whether it is possible to successfully carry out the instruction using the following devices.

Home appliances: {devices}

Instruction : {user_command}


{format_instructions}
"""


filtering_template = """
You are an AI that controls a smart home.
You will be provided with a set of home appliances and an instruction.
Extract the names of the devices that are relevant to execute the instruction

APPLIANCE LIST: {devices}

Instruction : {user_command}

Respond by returning a subset of the devices under APPLIANCE LIST
{format_instructions}
"""

pre_planning_template = """
You are an AI that controls a smart home.
You will be provided with a set of home appliances and an instruction.

DEVICES: {devices}
INSTRUCTION: {user_command}

{format_instructions}
"""


planning_template = """
You are an AI that controls a smart home.
You will be provided with a set of home appliances and an instruction.
Your job is to figure out how to use the set of home appliances to execute the instruction.

FILTERED DEVICES: {devices}

USER COMMAND: {user_command}

Return a json file that follows the structure of FILTERED DEVICES. Change the values of the
dictionary to best accomplish the USER COMMAND.

{format_instructions}
"""

persistent_template = """
You are an AI that controls a smart home. You receive user commands and create automation routines in response.

Devices and Sensors: {devices}

User command:{user_command}

Analyze the devices and sensors. Propose a sensor trigger, and how you would change the devices based on that trigger.

{format_instructions}
"""

reading_template = """
You are an AI that controls a smart home.
You are asked to retrieve the relevant information from a device state.


DEVICE STATE: {devices}
USER COMMAND: {user_command}
ANSWER:

{format_instructions}
"""


### Setup the prompt template ###

### multistage prompts ###
clarification_prompt_template = PromptTemplate(
    input_variables=[
        "devices",
        "user_command",
    ],
    template=clarification_template,
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=ClarificationResponse
        ).get_format_instructions()
    },
)

filtering_prompt_template = PromptTemplate(
    input_variables=["devices", "user_command"],
    template=filtering_template,
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=FilteringResponse
        ).get_format_instructions()
    },
)

pre_planning_prompt_template = PromptTemplate(
    input_variables=["devices", "user_command"],
    template=pre_planning_template,
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=PrePlanningResponse
        ).get_format_instructions()
    },
)

planning_prompt_template = PromptTemplate(
    input_variables=["devices", "user_command"],
    template=planning_template,
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=PlanningResponse
        ).get_format_instructions()
    },
)

reading_prompt_template = PromptTemplate(
    input_variables=["devices", "user_command"],
    template=reading_template,
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=ReadingResponse
        ).get_format_instructions()
    },
)

persistent_prompt_template = PromptTemplate(
    input_variables=["devices", "user_command"],
    template=persistent_template,
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=PersistentResponse
        ).get_format_instructions()
    },
)
