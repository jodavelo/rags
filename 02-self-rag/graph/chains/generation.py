from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# Inline equivalent of hub.pull("rlm/rag-prompt") — the hub module was removed
# from the langchain package, and pulling public prompts now requires a
# LangSmith API key.

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

def format_docs(docs) -> str:
    """Render retrieved documents as plain text.

    Interpolating Document objects directly would dump their full metadata
    (Unstructured's `orig_elements` alone is ~30k chars per doc) into the prompt.
    """
    return "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs
    )


generation_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)