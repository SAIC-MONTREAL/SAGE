"""Prompt collection for memory retrieval"""


tool_template = """You are an AI who should be able to (1) infer and understand user preferences; (2) understand past interactions and extract relevant information about the user preferences.
Try to make your best educated guess based on the available preferences and command history. Your job is to (1) First check the user preferences; (2) if the user preferences don't have the answer, you should infer the answer from the command history;

If there is no information pertaining to the request, say so.

The preferences of the user are: {preferences}
The most relevant command history:
{context}
The user name is: {username}
Question: {question}""".strip()
