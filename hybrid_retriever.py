from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.retrievers import BM25Retriever , EnsembleRetriever
from dotenv import load_dotenv
import os
load_dotenv()

loader = PyPDFLoader('data/d2l-en.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-small-en-v1.5'
)

persist_directory = './db'

if not os.path.exists(persist_directory):
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory
    )

else:
    vector_store = Chroma(
        persist_directory = persist_directory,
        embedding_function = embeddings
    )


chunk_retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        "k" : 3,
        "fetch_k" : 10
    }
)

bm25_retreiver = BM25Retriever.from_documents(
    documents = chunks,
    k = 3
)

hybrid_retriever = EnsembleRetriever(
    retrievers = [chunk_retriever, bm25_retreiver],
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

