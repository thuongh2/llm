from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv

dotenv.load_dotenv()

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

message = [
    SystemMessage(
        content="You're an assistant knowledgeable about healthcare. Only answer healthcare-related questions."
    ),
    HumanMessage(content = "What is Medicaid managed care?"),
]

print(chat_model.invoke(message))