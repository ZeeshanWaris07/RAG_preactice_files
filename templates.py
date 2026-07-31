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