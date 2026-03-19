import json
import os
import re
import time
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from dotenv import find_dotenv, load_dotenv

# ── LangChain / Anthropic ────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Tavily web search ────────────────────────────────────────────────────────
try:
    from tavily import TavilyClient
    TAVILY_CLIENT_AVAILABLE = True
except ImportError:
    TAVILY_CLIENT_AVAILABLE = False
    logging.warning(
        "tavily-python not installed; falling back to direct Tavily API requests."
    )

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
PDF_PATH        = Path(
    "ClaudeCertifiedArchitectFoundationsCertificationExamGuide.pdf"
)
SCHEMA_PATH     = BASE_DIR / "question_bank_schema.json"
OUTPUT_DIR      = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE          = 5          # questions per task-statement batch (raise to 10-15 for prod)
DIFFICULTY_SPLIT    = {          # must sum to 1.0
    "conceptual_knowledge":      0.30,
    "applied_judgment":          0.50,
    "anti_pattern_recognition":  0.20,
}
SLEEP_BETWEEN_BATCHES = 2        # seconds – avoid rate-limit spikes
MODEL_NAME            = "claude-sonnet-4-20250514"
MAX_TOKENS            = 8192

# ── Tavily search queries per domain ─────────────────────────────────────────
DOMAIN_SEARCH_QUERIES: dict[str, list[str]] = {
    "D1": [
        "Anthropic Claude Agent SDK agentic loop stop_reason tool_use 2024 2025",
        "Anthropic multi-agent orchestration coordinator subagent Task tool allowedTools",
    ],
    "D2": [
        "Anthropic MCP Model Context Protocol tool descriptions isError structured errors",
        "Claude tool_choice auto any forced selection MCP server .mcp.json configuration",
    ],
    "D3": [
        "Claude Code CLAUDE.md configuration hierarchy rules skills slash commands 2024 2025",
        "Claude Code plan mode CI CD pipeline -p flag --output-format json",
    ],
    "D4": [
        "Anthropic Claude few-shot prompting structured output JSON schema tool_use 2024 2025",
        "Anthropic Message Batches API cost savings custom_id batch processing",
    ],
    "D5": [
        "Claude context window management lost in middle summarization multi-agent 2024 2025",
        "Claude human review confidence calibration stratified sampling extraction accuracy",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & index the exam guide PDF
# ══════════════════════════════════════════════════════════════════════════════

def load_exam_guide(pdf_path: Path) -> str:
    """Return the full exam guide text, chunked-then-joined for the prompt."""
    log.info("Loading exam guide PDF: %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(pages)
    full_text = "\n\n".join(c.page_content for c in chunks)
    log.info("Exam guide loaded – %d chars across %d chunks", len(full_text), len(chunks))
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# 2. Web search for latest Anthropic content
# ══════════════════════════════════════════════════════════════════════════════

def fetch_web_context(domain_id: str, max_results: int = 3) -> str:
    """Run Tavily searches for the domain and return combined snippets."""
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        log.warning("TAVILY_API_KEY not set – skipping web search.")
        return ""

    queries  = DOMAIN_SEARCH_QUERIES.get(domain_id, [])
    snippets = []

    for query in queries:
        try:
            log.info("  Web search: %s", query)
            for result in _tavily_search(query, tavily_key, max_results):
                content = result.get("content", "")
                url = result.get("url", "")
                if content:
                    snippets.append(f"[SOURCE: {url}]\n{content[:800]}")
        except Exception as exc:
            log.warning("  Web search failed for '%s': %s", query, exc)

    combined = "\n\n---\n\n".join(snippets)
    log.info("  Web context: %d chars from %d queries", len(combined), len(queries))
    return combined


def _tavily_search(query: str, api_key: str, max_results: int) -> list[dict[str, str]]:
    """Return Tavily search results via the Python client or direct HTTP."""
    if TAVILY_CLIENT_AVAILABLE:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            topic="general",
            search_depth="advanced",
            include_raw_content=False,
        )
        return _normalize_tavily_results(response)

    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "topic": "general",
            "search_depth": "advanced",
            "include_raw_content": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _normalize_tavily_results(body)


def _normalize_tavily_results(response: Any) -> list[dict[str, str]]:
    """Extract url/content pairs from a Tavily response payload."""
    if isinstance(response, dict):
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        results = []

    normalized = []
    for item in results:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "url": str(item.get("url", "")),
                "content": str(
                    item.get("content")
                    or item.get("raw_content")
                    or item.get("snippet")
                    or ""
                ),
            }
        )
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# 3. Load the question bank schema
# ══════════════════════════════════════════════════════════════════════════════

def load_schema(schema_path: Path) -> dict:
    log.info("Loading schema: %s", schema_path)
    with open(schema_path, "r") as f:
        return json.load(f)


def difficulty_counts(batch_size: int) -> dict[str, int]:
    """Return per-difficulty question counts that sum to batch_size."""
    counts = {}
    remaining = batch_size
    tiers = list(DIFFICULTY_SPLIT.items())
    for i, (tier, pct) in enumerate(tiers):
        if i == len(tiers) - 1:
            counts[tier] = remaining
        else:
            n = round(batch_size * pct)
            counts[tier] = n
            remaining -= n
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# 4. Build the LangChain chain
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_TEMPLATE = """\
You are an expert exam question author for the "Claude Certified Architect – Foundations" \
certification exam produced by Anthropic.

Your task is to generate high-quality multiple-choice questions strictly following the \
rules below.

════════════════════════════════
EXAM CONTEXT (from official guide)
════════════════════════════════
{exam_guide_excerpt}

════════════════════════════════
LATEST ANTHROPIC DOCUMENTATION (web)
════════════════════════════════
{web_context}

════════════════════════════════
QUESTION BANK SCHEMA RULES
════════════════════════════════
Difficulty tiers:
  • conceptual_knowledge   – recall / understand (Bloom L1-L2)
  • applied_judgment       – apply / analyze in a production scenario (Bloom L3-L4)
  • anti_pattern_recognition – evaluate / critique a flawed approach (Bloom L5-L6)

Distractor rules (MANDATORY for every question):
  1. At least ONE distractor must be a "prompt-only / probabilistic compliance" anti-pattern
     (sounds correct but relies on LLM following instructions rather than deterministic enforcement).
  2. At least ONE distractor must be "over-engineered" (would work but adds unnecessary infrastructure).
  3. All distractors must reflect genuine misconceptions held by candidates with 3-5 months experience.
  4. No distractor should be obviously wrong or trivially excluded.

Stem variation rules:
  • Mix: "What should you do?", "What is the root cause?", "Why is this insufficient?",
         "Which option introduces risk?", "A colleague proposes X — what is the flaw?"
  • Include production metrics where relevant (e.g. "12% failure rate", "40% latency increase").
  • Reference specific API constructs (stop_reason, tool_choice, isError) for applied/anti-pattern tiers.

Output format:
  Return ONLY a valid JSON array. No markdown fences, no commentary outside the array.
  Each element must match this schema exactly:

  {{
    "id":                  "<domain_id>.<task_id>.<scenario_id>.<difficulty_abbrev>.<3-digit-seq>",
    "domain_id":           "<string>",
    "task_statement_id":   "<string>",
    "scenario_ids":        ["<string>", ...],
    "difficulty":          "conceptual_knowledge | applied_judgment | anti_pattern_recognition",
    "bloom_level":         "remember | understand | apply | analyze | evaluate",
    "key_concept_tested":  "<specific concept from key_concepts list>",
    "stem":                "<question text>",
    "options": [
      {{"label": "A", "text": "<option text>", "is_correct": false}},
      {{"label": "B", "text": "<option text>", "is_correct": false}},
      {{"label": "C", "text": "<option text>", "is_correct": false}},
      {{"label": "D", "text": "<option text>", "is_correct": false}}
    ],
    "correct_label": "<A|B|C|D>",
    "explanation":   "<why correct is right AND why each distractor is wrong>",
    "distractor_strategy": {{
      "plausible_but_prompt_only": "<label of that distractor>",
      "over_engineered":           "<label of that distractor>",
      "wrong_layer":               "<label or N/A>",
      "valid_but_not_first_step":  "<label or N/A>"
    }},
    "tags": ["<tag1>", "<tag2>", ...]
  }}
"""

HUMAN_TEMPLATE = """\
Generate exactly {batch_size} questions for:

  Domain:         {domain_id} – {domain_name}
  Task Statement: {task_id}   – {task_title}
  Key concepts to cover: {key_concepts}

  Primary scenarios to use: {scenario_ids}
  Scenario context:
{scenario_context}

  Difficulty distribution required:
{difficulty_distribution}

  Vary the correct answer label across the batch:
  suggested rotation: {label_rotation}

Remember:
  - Base questions on the exam guide excerpt and latest Anthropic documentation above.
  - Every question must be grounded in a realistic production scenario.
  - Do NOT reuse the same scenario framing within this batch.
  - Return ONLY a valid JSON array — no extra text.
"""


def build_chain(llm: ChatAnthropic):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Parse LLM output safely
# ══════════════════════════════════════════════════════════════════════════════

def parse_questions(raw: str, domain_id: str, task_id: str) -> list[dict]:
    """Extract JSON array from LLM output, with graceful fallback."""
    # Strip markdown fences if model added them despite instructions
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find the outermost [ ... ]
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        log.error("No JSON array found in response for %s / %s", domain_id, task_id)
        log.debug("Raw output: %s", raw[:500])
        return []

    try:
        questions = json.loads(raw[start:end + 1])
        log.info("  Parsed %d questions", len(questions))
        return questions
    except json.JSONDecodeError as exc:
        log.error("JSON parse error for %s / %s: %s", domain_id, task_id, exc)
        log.debug("Problematic JSON: %s", raw[start:start + 300])
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main generation loop
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario_context(schema: dict, scenario_ids: list[str]) -> str:
    scenarios = {s["id"]: s for s in schema["scenarios"]}
    lines = []
    for sid in scenario_ids:
        s = scenarios.get(sid)
        if s:
            lines.append(f"  [{s['id']}] {s['title']}: {s['context']}")
    return "\n".join(lines)


def label_rotation(batch_size: int, offset: int = 0) -> str:
    labels = ["A", "B", "C", "D"]
    rotation = [labels[(i + offset) % 4] for i in range(batch_size)]
    return ", ".join(rotation)


def generate_all_questions(
    schema:          dict,
    exam_guide_text: str,
    llm:             ChatAnthropic,
) -> list[dict]:
    """
    Iterate: domain → task_statement → generate batch.
    Returns all questions combined.
    """
    chain       = build_chain(llm)
    all_questions: list[dict] = []
    global_seq  = 0
    web_cache: dict[str, str] = {}   # cache web context per domain

    domains = schema["domains"]

    for domain in domains:
        domain_id   = domain["id"]
        domain_name = domain["name"]
        log.info("━━━ Domain %s: %s ━━━", domain_id, domain_name)

        # Fetch web context once per domain
        if domain_id not in web_cache:
            web_cache[domain_id] = fetch_web_context(domain_id)
        web_ctx = web_cache[domain_id]

        # Primary scenarios for this domain
        primary_scenario_ids = [
            s["id"] for s in schema["scenarios"]
            if domain_id in s["primary_domains"]
        ]

        for task in domain["task_statements"]:
            task_id    = task["id"]
            task_title = task["title"]
            key_concepts = task["key_concepts"]
            log.info("  ── Task %s: %s", task_id, task_title)

            # Build difficulty distribution string for prompt
            diff_counts = difficulty_counts(BATCH_SIZE)
            diff_str = "\n".join(
                f"    {tier}: {count} question(s)"
                for tier, count in diff_counts.items()
            )

            scenario_ctx = build_scenario_context(schema, primary_scenario_ids[:2])

            try:
                raw = chain.invoke({
                    "exam_guide_excerpt": exam_guide_text[:12000],  # token budget
                    "web_context":        web_ctx[:4000],
                    "batch_size":         BATCH_SIZE,
                    "domain_id":          domain_id,
                    "domain_name":        domain_name,
                    "task_id":            task_id,
                    "task_title":         task_title,
                    "key_concepts":       json.dumps(key_concepts, indent=2),
                    "scenario_ids":       primary_scenario_ids[:2],
                    "scenario_context":   scenario_ctx,
                    "difficulty_distribution": diff_str,
                    "label_rotation":     label_rotation(BATCH_SIZE, global_seq),
                })

                questions = parse_questions(raw, domain_id, task_id)

                # Assign sequential IDs if model didn't
                for i, q in enumerate(questions):
                    global_seq += 1
                    if not q.get("id") or q["id"].startswith("<"):
                        abbrev = {
                            "conceptual_knowledge":     "CK",
                            "applied_judgment":         "AJ",
                            "anti_pattern_recognition": "AP",
                        }.get(q.get("difficulty", ""), "XX")
                        q["id"] = f"{task_id}.{primary_scenario_ids[0] if primary_scenario_ids else 'SX'}.{abbrev}.{global_seq:04d}"

                all_questions.extend(questions)

                # Save per-task file immediately (safe checkpointing)
                task_file = OUTPUT_DIR / f"questions_{task_id.replace('.', '_')}.json"
                with open(task_file, "w") as f:
                    json.dump(questions, f, indent=2)
                log.info("  Saved %d questions → %s", len(questions), task_file.name)

            except Exception as exc:
                log.error("  FAILED for %s / %s: %s", domain_id, task_id, exc)

            time.sleep(SLEEP_BETWEEN_BATCHES)

    return all_questions


# ══════════════════════════════════════════════════════════════════════════════
# 7. Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv(find_dotenv())

    # ── API keys ──────────────────────────────────────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        log.warning(
            "TAVILY_API_KEY not set. Questions will be generated from the exam guide only.\n"
            "Get a free key at https://app.tavily.com and export TAVILY_API_KEY='tvly-...'"
        )

    # ── Load inputs ───────────────────────────────────────────────────────────
    schema          = load_schema(SCHEMA_PATH)
    exam_guide_text = load_exam_guide(PDF_PATH)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        anthropic_api_key=anthropic_key,
        temperature=0.7,          # some creativity for distractor variety
    )
    log.info("LLM initialised: %s", MODEL_NAME)

    # ── Generate ──────────────────────────────────────────────────────────────
    log.info("Starting question generation — %d questions per task statement", BATCH_SIZE)
    log.info("Domains: %d | Total task statements: %d",
             len(schema["domains"]),
             sum(len(d["task_statements"]) for d in schema["domains"]))

    all_questions = generate_all_questions(schema, exam_guide_text, llm)

    # ── Save merged bank ──────────────────────────────────────────────────────
    merged_file = OUTPUT_DIR / "questions_ALL.json"
    with open(merged_file, "w") as f:
        json.dump(all_questions, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("═══════════════════════════════════════")
    log.info("Generation complete!")
    log.info("  Total questions generated : %d", len(all_questions))
    log.info("  Merged output file        : %s", merged_file)
    log.info("  Per-task files            : %s/*.json", OUTPUT_DIR)

    # Difficulty breakdown
    from collections import Counter
    diff_counts = Counter(q.get("difficulty", "unknown") for q in all_questions)
    log.info("  Difficulty breakdown:")
    for tier, count in sorted(diff_counts.items()):
        log.info("    %-32s %d", tier, count)

    # Domain breakdown
    domain_counts = Counter(q.get("domain_id", "?") for q in all_questions)
    log.info("  Domain breakdown:")
    for did, count in sorted(domain_counts.items()):
        log.info("    %-6s %d", did, count)
    log.info("═══════════════════════════════════════")


if __name__ == "__main__":
    main()
