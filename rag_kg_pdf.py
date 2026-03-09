import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.graphs.networkx_graph import KnowledgeTriple, NetworkxEntityGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Prompts
# -----------------------------

TRIPLE_PROMPT = ChatPromptTemplate.from_template(
    """Extract factual knowledge triples from the text.

Return STRICT JSON only with this schema:
{{
  "triples": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ]
}}

Rules:
- Use short, canonical entity names (e.g., "LangChain", "FAISS", "OpenAIEmbeddings").
- Predicates should be simple verbs/relations (e.g., "uses", "defines", "requires", "causes", "located_in").
- Only include triples clearly supported by the text.
- Up to {max_triples} triples.

Text:
{text}
"""
)

ENTITY_PROMPT = ChatPromptTemplate.from_template(
    """Extract key entities from the user question.

Return STRICT JSON only:
{{"entities": ["...", "..."]}}

Rules:
- Entities should be short canonical names.
- 3 to 10 entities max.
- If none, return {{"entities": []}}.

Question: {question}
"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a careful assistant. Answer the user's question using ONLY:
1) Retrieved PDF context, and
2) Knowledge graph facts (triples) derived from the PDF.

If the answer is not supported, say: "I don't know based on the provided PDF."

<retrieved_context>
{retrieved_context}
</retrieved_context>

<knowledge_graph_facts>
{kg_facts}
</knowledge_graph_facts>

Question: {question}

Return a concise, direct answer. If helpful, cite page numbers from retrieved context."""
)


# -----------------------------
# Utilities
# -----------------------------

def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # one doc per page

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.from_documents(chunks, embeddings)


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


def _safe_json_load(s: str) -> Dict:
    # Attempt strict parse first; if model wrapped extra text, extract first JSON object.
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(m.group(0))


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


def extract_triples(llm: ChatOpenAI, text: str, max_triples: int = 12) -> List[Triple]:
    msg = TRIPLE_PROMPT.format_messages(text=text, max_triples=max_triples)
    resp = llm.invoke(msg).content
    data = _safe_json_load(resp)

    triples = []
    for t in data.get("triples", []):
        s = (t.get("subject") or "").strip()
        p = (t.get("predicate") or "").strip()
        o = (t.get("object") or "").strip()
        if s and p and o:
            triples.append(Triple(s, p, o))
    return triples


def build_kg_from_chunks(llm: ChatOpenAI, chunks: List[Document], max_chunks: int = 60) -> NetworkxEntityGraph:
    """
    Build a lightweight KG by extracting triples from up to max_chunks chunks.
    Increase max_chunks for better coverage (more cost/time).
    """
    graph = NetworkxEntityGraph()

    use_chunks = chunks[:max_chunks]
    for i, d in enumerate(use_chunks, start=1):
        text = d.page_content
        triples = extract_triples(llm, text, max_triples=10)

        for tr in triples:
            # NetworkxEntityGraph expects a KnowledgeTriple
            graph.add_triple(KnowledgeTriple(tr.subject, tr.predicate, tr.object))

        if i % 10 == 0:
            print(f"  KG progress: processed {i}/{len(use_chunks)} chunks...")

    return graph


def extract_entities(llm: ChatOpenAI, question: str) -> List[str]:
    msg = ENTITY_PROMPT.format_messages(question=question)
    resp = llm.invoke(msg).content
    data = _safe_json_load(resp)
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []
    cleaned = []
    for e in entities:
        if isinstance(e, str):
            e2 = e.strip()
            if e2:
                cleaned.append(e2)
    # de-dupe while preserving order
    seen = set()
    out = []
    for e in cleaned:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def kg_context_for_question(llm: ChatOpenAI, graph: NetworkxEntityGraph, question: str, depth: int = 2, max_lines: int = 40) -> str:
    entities = extract_entities(llm, question)
    lines: List[str] = []
    for e in entities:
        facts = graph.get_entity_knowledge(e, depth=depth)  # returns ["A rel B", ...]
        lines.extend(facts)

    # de-dupe
    uniq = []
    seen = set()
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)

    return "\n".join(uniq[:max_lines])


def answer_question(llm: ChatOpenAI, retriever, graph: NetworkxEntityGraph, question: str, kg_depth: int = 2) -> str:
    retrieved_docs = retriever.invoke(question)
    retrieved_context = format_context(retrieved_docs)

    kg_facts = kg_context_for_question(llm, graph, question, depth=kg_depth)

    msg = ANSWER_PROMPT.format_messages(
        retrieved_context=retrieved_context,
        kg_facts=kg_facts if kg_facts else "(no relevant KG facts found)",
        question=question,
    )
    resp = llm.invoke(msg)
    return resp.content


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG + Knowledge Graph Q&A over a single PDF (LangChain + FAISS + NetworkX KG)")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Chat model name")
    parser.add_argument("--kg-chunks", type=int, default=60, help="Max chunks to use for KG construction (cost/time knob)")
    parser.add_argument("--kg-depth", type=int, default=2, help="Graph traversal depth when collecting KG facts for a query")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    llm = ChatOpenAI(model=args.model, temperature=0)

    print(f"Loading + chunking: {args.pdf}")
    chunks = load_and_chunk_pdf(args.pdf)
    print(f"Chunks: {len(chunks)}")

    print("Building FAISS index...")
    vs = build_vectorstore(chunks)
    retriever = vs.as_retriever(search_kwargs={"k": args.k})

    print(f"Building KG from up to {args.kg_chunks} chunks (this uses the LLM)...")
    graph = build_kg_from_chunks(llm, chunks, max_chunks=args.kg_chunks)
    print(f"KG built. Triples: {len(graph.get_triples())}")

    print("\nReady. Ask questions (type 'exit' to quit).\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        a = answer_question(llm, retriever, graph, q, kg_depth=args.kg_depth)
        print(f"\nA> {a}\n")


if __name__ == "__main__":
    load_dotenv(find_dotenv()) 
    main()
