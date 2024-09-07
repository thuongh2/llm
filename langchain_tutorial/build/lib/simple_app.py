import dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

dotenv.load_dotenv()

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    SystemMessage(content="Translate the following from English into VietNames"),
    HumanMessage(content="hi!"),
]

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "vietnames", "text": "hi"})

print(result)

chain = prompt_template | model | parser

response = chain.invoke({"language": "vietnames", "text": "hi"})
print(response)