import os
import dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

dotenv.load_dotenv()

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print("ok1")


neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)
    
review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)

messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

reviews_vector_chain = RetrievalQA.from_llm(
            llm=chat_model, 
            retriever=neo4j_vector_index.as_retriever(k=12),
        )


reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

query = """Which state had the largest percent increase
           in Medicaid visits from 2022 to 2023?"""

response = reviews_vector_chain.invoke(query)

print(response.get("result"))