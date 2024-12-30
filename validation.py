import os
import json

from dotenv import load_dotenv

from datasets import Dataset
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, NonLLMContextRecall
from langchain_openai import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

file = open("data/validation_data.json")
eval_data = json.load(file)
eval_hf_dataset = Dataset.from_dict(eval_data)
eval_dataset = EvaluationDataset.from_hf_dataset(eval_hf_dataset)


os.environ["RAGAS_DEBUG"] = "true"

run_config = RunConfig(timeout=6000, log_tenacity=True)

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="qwen2.5-7b-instruct",
        base_url="http://localhost:1234/v1",
        api_key=openai_api_key,
        timeout=6000,
    )
)

metrics = [
    LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    NonLLMContextRecall(),
]
results = evaluate(
    dataset=eval_dataset, metrics=metrics, run_config=run_config, llm=evaluator_llm
)
print(results)

results_df = results.to_pandas()
results_df.to_csv("data/validation_results.csv", index=False)
