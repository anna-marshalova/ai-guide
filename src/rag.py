import os
import sys

sys.path.append("./")

from dotenv import load_dotenv

from langchain_community.chat_models import GigaChat
from langchain.chains import ConversationChain
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
        self.rag_prompt_template = """
            Ты - опытный туристический гид с обширными знаниями о путешествиях, культуре и истории разных мест.
            Контекст из надежного источника: {context}

            Вопрос пользователя: {question}

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
        self.general_prompt_template = """
            Ты - опытный туристический гид с обширными знаниями о путешествиях, культуре и истории разных мест.

            Вопрос пользователя: {question}
            
            Стремись дать максимально полезный, информативный и практичный ответ, комбинируя все доступные знания.
            
            Ответ:"""

        self.chain = ConversationChain(llm=self.llm)

    def retrieve(self, query):
        return self.retriever.retrieve(query)

    def run(self, query):
        context = self.retrieve(query)
        # result = self.rag_chain.invoke({"question": query, "context": context}).content
        if len(context) > 0:
            prompt = self.rag_prompt_template.format(question=query, context=context)
        else:
            prompt = self.general_prompt_template.format(question=query)
        result = self.chain.predict(input=prompt)
        return {"response": result, "retrieved_chunks": context}


if __name__ == "__main__":
    chunked_data = load_and_preprocess_data(datadir="./data")

    rag = RAG(chunked_data)
    print(rag.run("привет"))
    # print(rag.run("Что посмотреть в Шанхае?"))
