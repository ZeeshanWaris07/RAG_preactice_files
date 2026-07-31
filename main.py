from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()


def load_documents(folder_name = "data"):

    documents = []

    for file in os.listdir(folder_name):

        if file.endswith(".pdf"):

            print(f"Loading {file}")

            loader = PyPDFLoader(os.path.join(folder_name,file))

            documents.extend(loader.load())

    return documents


doc_pages = load_documents()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 40
)

chunks = splitter.split_documents(doc_pages)

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-small-en-v1.5'
)

persist_directory = "./chroma_db"

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

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 3,
        "fetch_k" : 10
    }
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

