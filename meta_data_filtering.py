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
from templates import prompt , rewrite_prompt, topic_prompt
import os
load_dotenv()

def format_docs(docs):

    context = []

    for doc in docs:

        context.append(
            f"""
Source: {doc.metadata['source']}
Page: {doc.metadata['page']}

{doc.page_content}
"""
        )

    return "\n\n".join(context)

def load_documents(folder_name = "data"):

    documents = []

    for file in os.listdir(folder_name):
        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(folder_name,file))

            documents.extend(loader.load())

    return documents

def add_metadata_to_docs(chunks):

    for chunk in chunks:

        filename = os.path.basename(chunk.metadata["source"])

        if filename == "Introduction-to-AI-and-Basic-Concepts.pdf":
            chunk.metadata['topic'] = "artificial intelligence"

        elif filename == "ML_do.pdf":
            chunk.metadata['topic'] = "machine learning"

        elif filename == "d2l-en.pdf":
            chunk.metadata['topic'] = "deep learning"

documents = load_documents()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 40
)

chunks = splitter.split_documents(documents)

add_metadata_to_docs(chunks)

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-small-en-v1.5'
)

persist_directory = "./chroma_db_2"

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

llm = ChatGoogleGenerativeAI(
    model = "gemini-flash-3.6"
)

topic_chain = (
    topic_prompt
    | llm 
    | StrOutputParser()
)

rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)


chat_history = []

while(True):

    choice = input("chat or exit : ")

    if choice == "exit":
        break
    else:

        question = input("What is your Question? : ")

        rewrited_question = rewrite_chain.invoke({
            "chat_history" : chat_history,
            "question" : question
        })

        topic = topic_chain.invoke(rewrited_question)

        print(f"Question : {question}")
        print(f"Rewrited Question : {rewrited_question}")
        print(f"Topic : {topic}")

        retriever = vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k" : 3,
                "fetch_k" : 10,
            },
            filer = {
                "metadata" : {
                    "topic" : topic
                }
            }
        )

        retrieved_docs = retriever.invoke(rewrited_question)
        formatted_docs = format_docs(retrieved_docs)

        response = (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({
            "context" : formatted_docs,
            "question" : rewrited_question
        })

        print(f"Response : {response}")
