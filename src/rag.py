import os
import sys

sys.path.append("./")

from dotenv import load_dotenv

from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.data.data_processing import load_and_preprocess_data

from src.retriever import HierarchicalRetriever

load_dotenv()

giga_key = os.getenv("API_KEY")


class RAG:
    def __init__(self, data, model_name="GigaChat"):
        self.retriever = HierarchicalRetriever(data)
        self.llm = GigaChat(
            credentials=giga_key,
            model=model_name,
            timeout=30,
            verify_ssl_certs=False,
            profanity_check=False,
        )
        self.rag_system_prompt = """
            Ты - опытный туристический гид с обширными знаниями о путешествиях, культуре и истории разных мест.

            При ответе:
            1. В первую очередь используй факты из предоставленного контекста - это самая актуальная и проверенная информация
            2. Дополняй ответ релевантными общими знаниями о:
            - Культурных особенностях и традициях
            - Исторических фактах
            - Практических советах по путешествиям
            - Современных тенденциях туризма
            3. Четко разграничивай информацию из контекста и общие знания
            4. Структурируй информацию в удобном для чтения формате
            5. Если информация из разных источников противоречит друг другу, отдавай приоритет контексту
            6. Если в контексте недостаточно информации по какому-то аспекту вопроса - используй свои знания, но укажи это.

            Стремись дать максимально полезный, информативный и практичный ответ, комбинируя все доступные знания.
            Ответ:"""
        self.general_system_prompt = """
            Ты - опытный туристический гид с обширными знаниями о путешествиях, культуре и истории разных мест.
            Стремись дать максимально полезный, информативный и практичный ответ, комбинируя все доступные знания.
            """
        self.rag_user_prompt_template = """Контекст из надежного источника: {context}

Вопрос пользователя: {question}"""


    def retrieve(self, query):
        return self.retriever.retrieve(query)
    
    def create_prompt(self, query, context):
        if len(context) > 0:
            return [
                SystemMessage(content=self.rag_system_prompt),
                HumanMessage(self.rag_user_prompt_template.format(question=query, context=context))
                ]
        return [
                SystemMessage(content=self.general_system_prompt),
                HumanMessage(query)
                ]

    def run(self, query):
        context = self.retrieve(query)
        prompt = self.create_prompt(query, context)
 
        response = self.llm.invoke(prompt)
        if response.response_metadata['finish_reason'] == 'blacklist':
            prompt = self.create_prompt(query, [])
            response = self.llm.invoke(prompt)
        result = response.content
        return {"response": result, "retrieved_chunks": context}


if __name__ == "__main__":
    chunked_data = load_and_preprocess_data(datadir="./data")

    rag = RAG(chunked_data)
    print(rag.run("Что посмотреть в Москве?"))
    # print(rag.run("Что посмотреть в Шанхае?"))
