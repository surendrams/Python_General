import argparse
import os
import sys
from typing import List
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a careful assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I don't know based on the provided PDF."

<context>
{context}
</context>

Question: {question}

Return a concise, direct answer. If helpful, cite page numbers when available in metadata."""
)


def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # one Document per page (metadata includes "page")
    # Split pages into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # FAISS.from_documents handles embedding + index build
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        src = d.metadata.get("source")
        header = []
        if src:
            header.append(f"source={src}")
        if page is not None:
            header.append(f"page={page}")
        meta = f"[{', '.join(header)}]" if header else ""
        parts.append(f"{meta}\n{d.page_content}".strip())
    return "\n\n---\n\n".join(parts)


def answer_question(llm: ChatOpenAI, retriever, question: str) -> str:
    # retriever is a Runnable in modern LangChain; invoke(question) returns docs
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    msg = RAG_PROMPT.format_messages(context=context, question=question)
    resp = llm.invoke(msg)
    return resp.content


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A over a single PDF (LangChain + FAISS)")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Chat model name")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading + chunking: {args.pdf}")
    chunks = load_and_chunk_pdf(args.pdf)
    print(f"Chunks: {len(chunks)}")

    print("Building FAISS index...")
    vs = build_vectorstore(chunks)
    retriever = vs.as_retriever(search_kwargs={"k": args.k})

    llm = ChatOpenAI(model=args.model, temperature=0)

    print("\nReady. Ask questions (type 'exit' to quit).\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        a = answer_question(llm, retriever, q)
        print(f"\nA> {a}\n")


if __name__ == "__main__":
    load_dotenv(find_dotenv())  # Load .env if it exists
    main()