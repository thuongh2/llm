__import__('pysqlite3')

import dotenv
import sys
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

dotenv.load_dotenv()

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

review_vector_db = Chroma.from_documents(
    reviews, embeddings, persist_directory=REVIEWS_CHROMA_PATH
)