from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'.
'yes' means every claim in the generation is explicitly stated in or directly inferable from the facts.
'no' means the generation contains claims that are not present in the facts, are off-topic relative to the facts, or contradict the facts.
If the facts do not discuss the subject of the generation at all, the generation is NOT grounded: answer 'no'."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
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


hallucination_grader: RunnableSequence = (
    RunnablePassthrough.assign(documents=lambda x: format_docs(x["documents"]))
    | hallucination_prompt
    | structured_llm_grader
)