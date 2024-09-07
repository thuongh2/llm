__import__('pysqlite3')

import dotenv
import sys
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

REVIEWS_CHROMA_PATH = "chroma_data/"

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embeddings,
)

question = """Has anyone complained about
           communication with the hospital staff?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)

for relevant_doc in relevant_docs:
    print("[RELEVANT]")
    print(relevant_doc.page_content)
    print("=============")