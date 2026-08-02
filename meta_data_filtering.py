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
from langchain_ollama import ChatOllama
from templates import prompt , rewrite_prompt, topic_prompt,metadata_prompt,retrieval_prompt
from pydantic import BaseModel
import os
load_dotenv()

class ChunkMetadata(BaseModel):
    topic : str
    subtopic : str
    difficulty : str

class SearchRequest(BaseModel):
    query : str
    topic : str | None = None
    subtopic : str | None = None
    difficulty : str | None = None

def format_docs(docs):

    context = []

    for doc in docs:

        context.append(
            f"""
Source: {doc.metadata['source']}
Page: {doc.metadata['page']}
Topic: {doc.metadata['topic']}
Subtopic: {doc.metadata['subtopic']}
Difficulty: {doc.metadata['difficulty']}

{doc.page_content}
"""
        )

    return "\n\n".join(context)

def load_documents(folder_name = "data"):

    documents = []

    for file in os.listdir(folder_name):
        if file.endswith("Introduction-to-AI-and-Basic-Concepts.pdf"):

            loader = PyPDFLoader(os.path.join(folder_name,file))

            documents.extend(loader.load())

    return documents

def add_metadata_to_docs(chunks,metadata_chain):

    for chunk in chunks:

        metadata = metadata_chain.invoke({
            "text" : chunk.page_content
        })

        chunk.metadata['topic'] = metadata.topic
        chunk.metadata['subtopic'] = metadata.subtopic
        chunk.metadata['difficulty'] = metadata.difficulty

def extract_filters_from_search_request(search_request):

    conditions = []

    if search_request.topic:
        conditions.append({"topic": search_request.topic})

    if search_request.subtopic:
        conditions.append({"subtopic": search_request.subtopic})

    if search_request.difficulty:
        conditions.append({"difficulty": search_request.difficulty})

    if len(conditions) == 0:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }

documents = load_documents()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 40
)


meta_data_llm = ChatOllama(model = "llama3.2")

structured_llm = meta_data_llm.with_structured_output(ChunkMetadata)

metadata_chain = (
    metadata_prompt
    | structured_llm
)

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-small-en-v1.5'
)

persist_directory = "./chroma_db_2"

if not os.path.exists(persist_directory):

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks : {len(chunks)}")
    add_metadata_to_docs(chunks, metadata_chain)

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
    model = "gemini-3.6-flash"
)

search_request_llm = llm.with_structured_output(SearchRequest)

filter_chain = (
    retrieval_prompt
    | search_request_llm
)

rewrite_chain = (
    rewrite_prompt
    | meta_data_llm
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
        
        search = filter_chain.invoke({"question": rewrited_question})

        filters = extract_filters_from_search_request(search)
    
        print(f"Question : {question}")
        print(f"Rewrited Question : {rewrited_question}")
        print(f"Query : {search.query}")
        print(f"Filters : {filters}")

        retriever = vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k" : 3,
                "fetch_k" : 10,
                "filter" : filters
            },
        )

        retrieved_docs = retriever.invoke(search.query)
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
