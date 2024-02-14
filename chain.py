import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

SYSTEM_DERIVE_FACTS = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/derive_facts.yaml'))
SYSTEM_INTROSPECTION = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/introspection.yaml'))
SYSTEM_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/response.yaml'))

def langchain_message_converter(messages: List):
    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


class LMChain:
    pass