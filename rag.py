import json
import os

from dotenv import load_dotenv

from langchain_community.chat_models import GigaChat

from data_processing import flatten_data, make_chunks

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from hierarchical_retrieval import HierarchicalRetrieval

load_dotenv()

giga_key = os.getenv("API_KEY")


class RAG:
    def __init__(self, data):
        self.retriever = HierarchicalRetrieval(data)
        self.llm = GigaChat(
            credentials=giga_key,
            model="GigaChat-Pro",
            timeout=30,
            verify_ssl_certs=False,
            profanity_check=False,
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            """
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
        )
        self.rag_chain = (
            {
                "context": RunnableLambda(self.retrieve),
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
        )

    def retrieve(self, query):
        return self.retriever.retrieve(query["question"])

    def run(self, query):
        return self.rag_chain.invoke({"question": query}).content


if __name__ == "__main__":
    paths = os.listdir("./data")
    data = []
    for path in paths:
        with open(f"./data/{path}") as f:
            data.append(json.load(f))

    flat_data = flatten_data(data)
    chunked_data = make_chunks(flat_data)
    assert all(
        len(chunk) < 2000 for chunks in chunked_data.values() for chunk in chunks
    )

    rag = RAG(chunked_data)
    print(rag.run("Что посмотреть в Шанхае?"))
