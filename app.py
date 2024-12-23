import gradio as gr
from src.rag import RAG
from src.data.data_processing import load_and_preprocess_data
from src.interface import create_interface

DATA_DIR = "data"
LLM_NAME = "GigaChat"


chunked_data = load_and_preprocess_data(datadir=DATA_DIR)

rag = RAG(chunked_data, model_name=LLM_NAME)
interface = create_interface(lambda query: rag.run(query)["response"])
interface.launch(share=True)
