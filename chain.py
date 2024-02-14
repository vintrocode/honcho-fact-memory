import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

SYSTEM_DERIVE_FACTS = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/core/derive_facts.yaml'))
SYSTEM_INTROSPECTION = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/core/introspection.yaml'))
SYSTEM_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/core/response.yaml'))
SYSTEM_CHECK_DUPS = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts/utils/check_dup_facts.yaml'))


def langchain_message_converter(messages: List):
    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


class LMChain:
    "Wrapper class for encapsulating the multiple different chains used"
    output_parser = NumberedListOutputParser()
    llm: ChatOpenAI = ChatOpenAI(model_name = "gpt-4")
    system_derive_facts: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_DERIVE_FACTS)
    system_introspection: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_INTROSPECTION)
    system_response: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_RESPONSE)
    system_check_dups: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_CHECK_DUPS)

    def __init__(self) -> None:
        pass

    @classmethod
    async def derive_facts(self, input: str):
        """Derive facts from the user input"""
        pass
    
    @classmethod
    async def check_dups(self, facts: List):
        """Check that we're not storing duplicate facts"""
        pass

    @classmethod
    async def introspect(self, chat_history: List, input:str):
        """Generate questions about the user to use as retrieval over the fact store"""
        pass
    
    @classmethod
    async def respond(self, chat_history: List, facts: List, input: str):
        """Take the facts and chat history and generate a personalized response"""
        pass