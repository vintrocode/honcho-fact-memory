import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, load_prompt
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from honcho import Collection

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
    async def derive_facts(cls, input: str):
        """Derive facts from the user input"""

        # format prompt
        fact_derivation = ChatPromptTemplate.from_messages([
            cls.system_derive_facts
        ])

        # LCEL
        chain = fact_derivation | cls.llm
        
        # inference
        response = await chain.ainvoke({
            "user_input": input
        })

        # parse output
        facts = cls.output_parser.parse(response.content)

        # add as metamessage
        return facts
    
    @classmethod
    async def check_dups(cls, collection: Collection, facts: List):
        """Check that we're not storing duplicate facts"""

        # format prompt
        check_duplication = ChatPromptTemplate.from_messages([
            cls.system_check_dups
        ])

        query = " ".join(facts)
        result = collection.query(query=query, top_k=10) 
        existing_facts = [document.content for document in result]

        # LCEL
        chain = check_duplication | cls.llm

        # inference
        response = await chain.ainvoke({
            "existing_facts": existing_facts,
            "facts": facts
        })

        # parse output
        new_facts = cls.output_parser.parse(response.content)

        # TODO: write to vector store
        for fact in new_facts:
            collection.create_document(content=fact)

        return new_facts
        

    @classmethod
    async def introspect(cls, chat_history, input:str):
        """Generate questions about the user to use as retrieval over the fact store"""

        # format prompt
        introspection_prompt = ChatPromptTemplate.from_messages([
            cls.system_introspection
        ])

        # LCEL
        chain = introspection_prompt | cls.llm

        # inference
        response = await chain.ainvoke({
            "chat_history": chat_history,
            "input": input
        })

        # parse output
        questions = cls.output_parser.parse(response.content)

        # write as metamessages

        return questions

    
    @classmethod
    async def respond(cls, collection: Collection, chat_history, input: str):
        """Take the facts and chat history and generate a personalized response"""

        # format prompt
        response_prompt = ChatPromptTemplate.from_messages([
            cls.system_response,
            chat_history
        ])

        # TODO: query vector store for facts
        retrieved_facts = collection.query(query=input, top_k=10)

        # LCEL
        chain = response_prompt | cls.llm

        # inference
        response = await chain.ainvoke({
            "facts": retrieved_facts,
        })

        return response.content

        
        
