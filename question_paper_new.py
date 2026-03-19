"""
Claude Certified Architect – Foundations
Token-Optimised Question Bank Generator
==================================================
Target: 1000 questions across 30 task statements (≈34 per task)

Token Optimisation Strategy (vs original question_paper.py)
------------------------------------------------------------
1. PROMPT CACHING   – exam guide PDF is sent as a cache_control block.
                      First call per domain pays full price; every subsequent
                      call in that domain pays only the cache-read rate (~10x cheaper).

2. LARGE BATCHES    – batch_size raised from 5 → 34.
                      200 LLM calls → 30 calls (85% fewer), input tokens shared
                      across more questions.

3. LEAN SYSTEM PROMPT – exam_guide sent once via cache; web_context trimmed to
                        2 000 chars; distractor_strategy label removed from
                        per-question JSON (added deterministically post-generation).

4. LEAN OUTPUT JSON – only essential fields requested from LLM; heavy metadata
                      (distractor_strategy, bloom_level) injected post-hoc in Python
                      so we don't pay output tokens for boilerplate.

5. CHECKPOINT RESUME – if output/questions_D1_T1.json already exists and is valid,
                       that task is skipped entirely (free on reruns).

6. WEB CACHE         – Tavily results cached per domain to a local file
                       (output/web_cache.json); reused across reruns.

Usage
-----
  pip install langchain langchain-anthropic langchain-community \
              tavily-python pypdf python-dotenv

  export ANTHROPIC_API_KEY="sk-ant-..."
  export TAVILY_API_KEY="tvly-..."        # free at app.tavily.com

  python question_paper_optimised.py

Outputs
-------
  output/questions_D1_T1.json  …  (per task, checkpointed)
  output/questions_ALL.json        (merged bank)
  output/web_cache.json            (Tavily cache)
  output/run_stats.json            (token + cost summary)
"""

import json
import os
import re
import time
import logging
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from tavily import TavilyClient
    TAVILY_CLIENT_AVAILABLE = True
except ImportError:
    TAVILY_CLIENT_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PDF_PATH    = BASE_DIR / "ClaudeCertifiedArchitectFoundationsCertificationExamGuide.pdf"
SCHEMA_PATH = BASE_DIR / "question_bank_schema.json"
OUTPUT_DIR  = BASE_DIR / "output"
WEB_CACHE_FILE = OUTPUT_DIR / "web_cache.json"
STATS_FILE     = OUTPUT_DIR / "run_stats.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
# 30 task statements × 34 = 1 020 questions
BATCH_SIZE            = 34
TARGET_QUESTIONS      = 1000
DIFFICULTY_SPLIT      = {
    "conceptual_knowledge":     0.30,   # ~10 per batch
    "applied_judgment":         0.50,   # ~17 per batch
    "anti_pattern_recognition": 0.20,   # ~7  per batch
}
SLEEP_BETWEEN_BATCHES = 3        # seconds
MODEL_NAME            = "claude-sonnet-4-20250514"
MAX_TOKENS            = 16000    # large batches need more output budget
WEB_CONTEXT_CHARS     = 2000     # trimmed from 4 000 in original
EXAM_GUIDE_CHARS      = 12000    # same as original, but sent via cache block

# Sonnet 4 pricing per million tokens (as of 2025)
PRICE_INPUT_PER_M        = 3.00
PRICE_OUTPUT_PER_M       = 15.00
PRICE_CACHE_WRITE_PER_M  = 3.75   # slightly more than input on first write
PRICE_CACHE_READ_PER_M   = 0.30   # ~10x cheaper than standard input

# ── Tavily queries per domain ─────────────────────────────────────────────────
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
        "Claude Code CLAUDE.md configuration hierarchy rules skills slash commands 2025",
        "Claude Code plan mode CI CD pipeline -p flag --output-format json",
    ],
    "D4": [
        "Anthropic Claude few-shot prompting structured output JSON schema tool_use 2025",
        "Anthropic Message Batches API cost savings custom_id batch processing",
    ],
    "D5": [
        "Claude context window management lost in middle summarization multi-agent 2025",
        "Claude human review confidence calibration stratified sampling extraction accuracy",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION 1 — Prompt Caching helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_cached_system_message(exam_guide_text: str, web_context: str) -> list[dict]:
    """
    Build the system message with cache_control on the expensive exam guide block.
    This is passed as raw message dicts directly to the Anthropic client so we
    can use the cache_control extension that LangChain doesn't yet expose natively.

    Structure:
      [static rules block]  ← small, not cached
      [exam guide block]    ← large, CACHED (cache_control: ephemeral)
      [web context block]   ← medium, not cached (changes per domain)
    """
    static_rules = (
        "You are an expert exam question author for the "
        "\"Claude Certified Architect – Foundations\" certification exam.\n\n"
        "QUESTION BANK RULES\n"
        "════════════════════\n"
        "Difficulty tiers:\n"
        "  • conceptual_knowledge      – recall / understand (Bloom L1-L2)\n"
        "  • applied_judgment          – apply / analyze in a production scenario (Bloom L3-L4)\n"
        "  • anti_pattern_recognition  – evaluate / critique a flawed approach (Bloom L5-L6)\n\n"
        "Distractor rules (MANDATORY per question):\n"
        "  1. ONE distractor must be a prompt-only / probabilistic compliance anti-pattern.\n"
        "  2. ONE distractor must be over-engineered (unnecessary infrastructure).\n"
        "  3. All distractors must reflect genuine misconceptions (3-5 months experience).\n"
        "  4. No distractor should be trivially wrong.\n\n"
        "Stem variation rules:\n"
        "  • Mix: 'What should you do?', 'What is the root cause?',\n"
        "         'Why is this insufficient?', 'Which option introduces risk?',\n"
        "         'A colleague proposes X — what is the flaw?'\n"
        "  • Include production metrics where relevant (e.g. '12% failure rate').\n"
        "  • Reference specific constructs (stop_reason, tool_choice, isError) for "
        "applied/anti-pattern tiers.\n\n"
        "LEAN OUTPUT FORMAT — return ONLY a valid JSON array, no markdown fences.\n"
        "Each object must contain EXACTLY these keys:\n"
        "  id, domain_id, task_statement_id, scenario_ids, difficulty,\n"
        "  key_concept_tested, stem, options, correct_label, explanation, tags\n\n"
        "options format: [{\"label\":\"A\",\"text\":\"...\",\"is_correct\":false}, ...]\n"
        "correct_label: single character A | B | C | D\n"
    )

    return [
        # ── Block 1: static rules (small, no cache) ─────────────────────────
        {
            "type": "text",
            "text": static_rules,
        },
        # ── Block 2: exam guide (large, CACHED) ─────────────────────────────
        {
            "type": "text",
            "text": (
                "\nEXAM GUIDE (official source of truth)\n"
                "══════════════════════════════════════\n"
                + exam_guide_text[:EXAM_GUIDE_CHARS]
            ),
            "cache_control": {"type": "ephemeral"},   # ← prompt cache
        },
        # ── Block 3: web context (medium, changes per domain, no cache) ──────
        {
            "type": "text",
            "text": (
                "\nLATEST ANTHROPIC DOCS (web search)\n"
                "════════════════════════════════════\n"
                + (web_context[:WEB_CONTEXT_CHARS] if web_context else "(none available)")
            ),
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION 2 — Direct Anthropic client call (bypasses LangChain prompt)
# ══════════════════════════════════════════════════════════════════════════════

def call_with_cache(
    llm: ChatAnthropic,
    system_blocks: list[dict],
    human_text: str,
) -> tuple[str, dict]:
    """
    Call Claude directly using raw message dicts so cache_control is respected.
    Returns (response_text, usage_dict).
    """
    client = llm._client   # access underlying anthropic.Anthropic client

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=system_blocks,
        messages=[{"role": "user", "content": human_text}],
        temperature=0.7,
    )

    text = "".join(
        block.text for block in response.content
        if hasattr(block, "text")
    )
    usage = {
        "input_tokens":        response.usage.input_tokens,
        "output_tokens":       response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0),
    }
    return text, usage


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION 3 — Lean human prompt (per task statement)
# ══════════════════════════════════════════════════════════════════════════════

def build_human_prompt(
    batch_size:    int,
    domain_id:     str,
    domain_name:   str,
    task_id:       str,
    task_title:    str,
    key_concepts:  list[str],
    scenario_ids:  list[str],
    scenario_ctx:  str,
    diff_str:      str,
    label_hint:    str,
) -> str:
    return (
        f"Generate exactly {batch_size} exam questions.\n\n"
        f"Domain         : {domain_id} – {domain_name}\n"
        f"Task Statement : {task_id} – {task_title}\n"
        f"Key concepts   : {json.dumps(key_concepts)}\n\n"
        f"Scenarios:\n{scenario_ctx}\n\n"
        f"Difficulty distribution:\n{diff_str}\n\n"
        f"Rotate correct_label across the batch (suggested): {label_hint}\n\n"
        "Rules:\n"
        "  - Every question grounded in a realistic production scenario.\n"
        "  - Do NOT reuse same scenario framing within this batch.\n"
        "  - Return ONLY a valid JSON array — no extra text, no markdown.\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION 4 — Post-hoc metadata injection (saves output tokens)
# ══════════════════════════════════════════════════════════════════════════════

DIFFICULTY_BLOOM_MAP = {
    "conceptual_knowledge":     "understand",
    "applied_judgment":         "apply",
    "anti_pattern_recognition": "evaluate",
}

def enrich_question(q: dict, task_id: str, scenario_id: str, seq: int) -> dict:
    """
    Add bloom_level and distractor_strategy fields deterministically
    so the LLM doesn't have to generate them (saves output tokens).
    """
    diff = q.get("difficulty", "applied_judgment")
    q.setdefault("bloom_level", DIFFICULTY_BLOOM_MAP.get(diff, "apply"))

    # Infer distractor_strategy from explanation text if possible
    exp = q.get("explanation", "").lower()
    opts = {o["label"]: o["text"].lower() for o in q.get("options", [])}
    correct = q.get("correct_label", "A")
    distractors = [l for l in ["A", "B", "C", "D"] if l != correct]

    prompt_only = "N/A"
    over_eng    = "N/A"
    for label in distractors:
        txt = opts.get(label, "")
        if any(kw in txt for kw in ["system prompt", "instruct", "tell", "add a rule", "prompt"]):
            prompt_only = label
        if any(kw in txt for kw in ["deploy", "service", "infrastructure",
                                     "classifier", "separate", "registry"]):
            over_eng = label

    q.setdefault("distractor_strategy", {
        "plausible_but_prompt_only": prompt_only,
        "over_engineered":           over_eng,
        "wrong_layer":               "N/A",
        "valid_but_not_first_step":  "N/A",
    })

    # Assign ID if model left placeholder
    if not q.get("id") or q["id"].startswith("<"):
        abbrev = {"conceptual_knowledge": "CK",
                  "applied_judgment": "AJ",
                  "anti_pattern_recognition": "AP"}.get(diff, "XX")
        q["id"] = f"{task_id}.{scenario_id}.{abbrev}.{seq:04d}"

    return q


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION 5 — Checkpoint resume
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(task_id: str) -> Path:
    return OUTPUT_DIR / f"questions_{task_id.replace('.', '_')}.json"


def load_checkpoint(task_id: str) -> list[dict] | None:
    """Return saved questions if checkpoint exists and is valid JSON."""
    path = checkpoint_path(task_id)
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                log.info("  ✓ Checkpoint found – skipping %s (%d questions)", task_id, len(data))
                return data
        except Exception:
            pass
    return None


def save_checkpoint(task_id: str, questions: list[dict]):
    with open(checkpoint_path(task_id), "w") as f:
        json.dump(questions, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Web helpers (unchanged from original, with file caching added)
# ══════════════════════════════════════════════════════════════════════════════

def load_web_cache() -> dict:
    if WEB_CACHE_FILE.exists():
        try:
            with open(WEB_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_web_cache(cache: dict):
    with open(WEB_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _tavily_search(query: str, api_key: str, max_results: int = 3) -> list[dict]:
    if TAVILY_CLIENT_AVAILABLE:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            topic="general",
            search_depth="advanced",
            include_raw_content=False,
        )
    else:
        payload = json.dumps({
            "api_key": api_key, "query": query,
            "max_results": max_results, "topic": "general",
            "search_depth": "advanced", "include_raw_content": False,
        }).encode()
        req = urllib.request.Request(
            "https://api.tavily.com/search", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            response = json.loads(r.read().decode())

    results = response.get("results", []) if isinstance(response, dict) else response
    return [
        {"url": str(r.get("url", "")),
         "content": str(r.get("content") or r.get("snippet") or "")}
        for r in results if isinstance(r, dict)
    ]


def fetch_web_context(domain_id: str, cache: dict) -> str:
    """Return cached or freshly-fetched web context for a domain."""
    if domain_id in cache:
        log.info("  Web cache hit for %s", domain_id)
        return cache[domain_id]

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        log.warning("  TAVILY_API_KEY not set – skipping web search for %s", domain_id)
        return ""

    queries  = DOMAIN_SEARCH_QUERIES.get(domain_id, [])
    snippets = []
    for query in queries:
        try:
            log.info("  Web search: %s", query)
            for r in _tavily_search(query, tavily_key):
                if r["content"]:
                    snippets.append(f"[{r['url']}]\n{r['content'][:600]}")
        except Exception as exc:
            log.warning("  Web search failed '%s': %s", query, exc)

    combined = "\n\n---\n\n".join(snippets)
    cache[domain_id] = combined
    save_web_cache(cache)
    log.info("  Web context cached for %s (%d chars)", domain_id, len(combined))
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_exam_guide(pdf_path: Path) -> str:
    log.info("Loading exam guide PDF: %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=200,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(pages)
    text = "\n\n".join(c.page_content for c in chunks)
    log.info("Exam guide loaded – %d chars", len(text))
    return text


def load_schema(schema_path: Path) -> dict:
    with open(schema_path) as f:
        return json.load(f)


def difficulty_counts(batch_size: int) -> dict[str, int]:
    counts, remaining = {}, batch_size
    for i, (tier, pct) in enumerate(DIFFICULTY_SPLIT.items()):
        if i == len(DIFFICULTY_SPLIT) - 1:
            counts[tier] = remaining
        else:
            n = round(batch_size * pct)
            counts[tier] = n
            remaining  -= n
    return counts


def build_scenario_context(schema: dict, scenario_ids: list[str]) -> str:
    lookup = {s["id"]: s for s in schema["scenarios"]}
    return "\n".join(
        f"  [{sid}] {lookup[sid]['title']}: {lookup[sid]['context']}"
        for sid in scenario_ids if sid in lookup
    )


def label_rotation(batch_size: int, offset: int) -> str:
    labels = ["A", "B", "C", "D"]
    return ", ".join(labels[(i + offset) % 4] for i in range(batch_size))


def parse_questions(raw: str, domain_id: str, task_id: str) -> list[dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        log.error("No JSON array in response for %s/%s", domain_id, task_id)
        return []
    try:
        qs = json.loads(raw[start:end + 1])
        log.info("  Parsed %d questions", len(qs))
        return qs
    except json.JSONDecodeError as exc:
        log.error("JSON parse error %s/%s: %s", domain_id, task_id, exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Main generation loop
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_questions(schema: dict, exam_guide_text: str, llm: ChatAnthropic) -> list[dict]:
    all_questions: list[dict] = []
    global_seq  = 0
    web_cache   = load_web_cache()

    # Running token / cost stats
    stats = {
        "total_input_tokens":        0,
        "total_output_tokens":       0,
        "total_cache_write_tokens":  0,
        "total_cache_read_tokens":   0,
        "llm_calls":                 0,
        "skipped_checkpoints":       0,
        "estimated_cost_usd":        0.0,
    }

    domains = schema["domains"]

    for domain in domains:
        domain_id   = domain["id"]
        domain_name = domain["name"]
        log.info("━━━ Domain %s: %s ━━━", domain_id, domain_name)

        # Fetch (or load from cache) web context once per domain
        web_ctx = fetch_web_context(domain_id, web_cache)

        # Build the cached system blocks ONCE per domain
        # (the exam_guide block is identical across all tasks in this domain
        #  → Anthropic returns cache_read hits for calls 2-N)
        system_blocks = make_cached_system_message(exam_guide_text, web_ctx)

        primary_scenario_ids = [
            s["id"] for s in schema["scenarios"]
            if domain_id in s["primary_domains"]
        ]

        for task in domain["task_statements"]:
            task_id      = task["id"]
            task_title   = task["title"]
            key_concepts = task["key_concepts"]
            log.info("  ── Task %s: %s", task_id, task_title)

            # ── OPTIMISATION 5: skip if checkpoint exists ─────────────────
            cached_qs = load_checkpoint(task_id)
            if cached_qs is not None:
                all_questions.extend(cached_qs)
                stats["skipped_checkpoints"] += 1
                continue

            diff_str = "\n".join(
                f"    {tier}: {count} question(s)"
                for tier, count in difficulty_counts(BATCH_SIZE).items()
            )
            scenario_ctx = build_scenario_context(schema, primary_scenario_ids[:2])
            human_text   = build_human_prompt(
                batch_size   = BATCH_SIZE,
                domain_id    = domain_id,
                domain_name  = domain_name,
                task_id      = task_id,
                task_title   = task_title,
                key_concepts = key_concepts,
                scenario_ids = primary_scenario_ids[:2],
                scenario_ctx = scenario_ctx,
                diff_str     = diff_str,
                label_hint   = label_rotation(BATCH_SIZE, global_seq),
            )

            try:
                raw, usage = call_with_cache(llm, system_blocks, human_text)

                # Track tokens and cost
                inp  = usage["input_tokens"]
                out  = usage["output_tokens"]
                cw   = usage["cache_creation_input_tokens"]
                cr   = usage["cache_read_input_tokens"]
                cost = (
                    (inp / 1e6) * PRICE_INPUT_PER_M +
                    (out / 1e6) * PRICE_OUTPUT_PER_M +
                    (cw  / 1e6) * PRICE_CACHE_WRITE_PER_M +
                    (cr  / 1e6) * PRICE_CACHE_READ_PER_M
                )
                stats["total_input_tokens"]       += inp
                stats["total_output_tokens"]      += out
                stats["total_cache_write_tokens"] += cw
                stats["total_cache_read_tokens"]  += cr
                stats["estimated_cost_usd"]       += cost
                stats["llm_calls"]                += 1

                log.info(
                    "  Tokens → input:%d output:%d cache_write:%d cache_read:%d  cost:$%.4f",
                    inp, out, cw, cr, cost,
                )

                questions = parse_questions(raw, domain_id, task_id)

                # ── OPTIMISATION 4: enrich post-hoc ──────────────────────
                primary_sid = primary_scenario_ids[0] if primary_scenario_ids else "SX"
                for q in questions:
                    global_seq += 1
                    enrich_question(q, task_id, primary_sid, global_seq)

                all_questions.extend(questions)
                save_checkpoint(task_id, questions)
                log.info("  Saved %d questions → %s", len(questions), checkpoint_path(task_id).name)

            except Exception as exc:
                log.error("  FAILED %s/%s: %s", domain_id, task_id, exc)

            time.sleep(SLEEP_BETWEEN_BATCHES)

    # Save final stats
    stats["total_questions"] = len(all_questions)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    return all_questions


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv(find_dotenv())

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    if not os.getenv("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set – web search disabled (exam guide only).")

    schema          = load_schema(SCHEMA_PATH)
    exam_guide_text = load_exam_guide(PDF_PATH)

    llm = ChatAnthropic(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        anthropic_api_key=anthropic_key,
        temperature=0.7,
    )
    log.info("LLM: %s | Batch size: %d | Target: %d questions",
             MODEL_NAME, BATCH_SIZE, TARGET_QUESTIONS)

    total_tasks = sum(len(d["task_statements"]) for d in schema["domains"])
    log.info("Domains: %d | Task statements: %d | Est. questions: %d",
             len(schema["domains"]), total_tasks, total_tasks * BATCH_SIZE)

    all_questions = generate_all_questions(schema, exam_guide_text, llm)

    # Save merged bank
    merged_file = OUTPUT_DIR / "questions_ALL.json"
    with open(merged_file, "w") as f:
        json.dump(all_questions, f, indent=2)

    # Print summary
    with open(STATS_FILE) as f:
        stats = json.load(f)

    log.info("═══════════════════════════════════════════")
    log.info("Generation complete!")
    log.info("  Total questions     : %d", len(all_questions))
    log.info("  LLM calls made      : %d", stats["llm_calls"])
    log.info("  Checkpoints skipped : %d", stats["skipped_checkpoints"])
    log.info("  Input tokens        : %d", stats["total_input_tokens"])
    log.info("  Output tokens       : %d", stats["total_output_tokens"])
    log.info("  Cache write tokens  : %d", stats["total_cache_write_tokens"])
    log.info("  Cache read tokens   : %d", stats["total_cache_read_tokens"])
    log.info("  Estimated cost      : $%.4f", stats["estimated_cost_usd"])
    log.info("  Merged output       : %s", merged_file)

    diff_counts = Counter(q.get("difficulty", "?") for q in all_questions)
    log.info("  Difficulty breakdown:")
    for tier, count in sorted(diff_counts.items()):
        log.info("    %-35s %d", tier, count)

    domain_counts = Counter(q.get("domain_id", "?") for q in all_questions)
    log.info("  Domain breakdown:")
    for did, count in sorted(domain_counts.items()):
        log.info("    %-6s %d", did, count)
    log.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()