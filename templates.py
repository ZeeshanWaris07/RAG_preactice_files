from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question in a simple way in your own words.
Also Explain the answer and cite the source.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
Given the following conversation:

{chat_history}

Rewrite the user's latest question into a standalone question.

Only rewrite it.
Do not answer it.

Question:
{question}
""")

topic_prompt = ChatPromptTemplate.from_template("""
You are a classifier.

Possible topics:

- artificial intelligence
- machine learning
- deep learning

Return ONLY one topic.

Question:
{question}
""")

metadata_prompt = ChatPromptTemplate.from_template("""
You are an expert at classifying document chunks.

Analyze the following text and extract:

- topic
- subtopic
- difficulty (Beginner, Intermediate, Advanced)

Text:
{text}
""")

retrieval_prompt = ChatPromptTemplate.from_template("""
    You are a search assistant.

Given the user's question, extract:

- the semantic search query
- any metadata filters

Return:

{{
  "query": "...",
  "topic": "...",
  "subtopic": "...",
  "difficulty": "..."
}}

If a field is not mentioned, leave it null.

Question:
{question}
""")