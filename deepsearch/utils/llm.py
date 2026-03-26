
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

def create_llm_from_env(temperature: float=0.5):
    return init_chat_model(model="deepseek:deepseek-chat", temperature=temperature)
