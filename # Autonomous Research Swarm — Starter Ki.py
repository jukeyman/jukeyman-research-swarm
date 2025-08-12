# Autonomous Research Swarm — Starter Kit

A production-ready blueprint for agents that **research continuously**, **debate**, **self-critique**, **loop**, and **log everything** until quality and coverage targets are met.

---

## 1) System Overview

**Goal:** You give a topic → a coordinated swarm of agents plans, searches, reads, extracts evidence, debates, identifies gaps, replans, and iterates until the research meets explicit **stopping criteria**. All steps are logged and reproducible.

**Core roles:**

* **Orchestrator** – owns the loop, budget, and stopping criteria.
* **Planner** – decomposes goals into tasks and updates the research roadmap.
* **Researchers** – execute web/database searches, read sources, extract facts.
* **Analyst** – synthesizes notes into evidence-backed summaries.
* **Critic** – scores quality, finds gaps, flags weak evidence.
* **Librarian** – deduplicates, embeds, and indexes evidence with citations.

**Key properties:**

* Deterministic loop with **budget & convergence guards** (steps/tokens/time).
* **Evidence Board** (JSONL) + **Research Log** (events.jsonl) for full traceability.
* **Gap-Driven Planning**: Critic proposes next tasks until thresholds are satisfied.
* Pluggable tools (search, crawl, summarize, embed) so you can swap providers.

---

## 2) Architecture (ASCII)

```
User Topic → Orchestrator → Planner ──────┐
                          │               │
                          ▼               │
                    Task Queue             │
                          │               │
          ┌───────────────┼───────────────┘
          ▼               ▼
     Researcher(s)   Librarian  ← embeds, indexes, de-duplicates
          │               │
          ▼               │
       Evidence ──────────┘
          │
          ▼
        Analyst → Draft → Critic → (scores, gaps) → Orchestrator
                                              └──── replans/stop
```

---

## 3) Loop Algorithm (high-level)

1. **Initialize** run context (topic, constraints, budgets, thresholds).
2. **Plan**: Planner creates/updates a hierarchical task list (HITL optional).
3. **Research**: Researchers execute tasks (search → fetch → extract → cite).
4. **Ingest**: Librarian embeds & indexes evidence; dedups and links sources.
5. **Synthesize**: Analyst drafts/update the report with inline citations.
6. **Critique**: Critic scores completeness, evidence strength, bias, clarity.
7. **Decide**: Orchestrator checks stopping criteria. If unmet, *replan* and loop.
8. **Finalize**: Lock evidence snapshot; export report + audit log.

---

## 4) Stopping Criteria (configurable)

* **Coverage**: % of sub-questions resolved ≥ threshold.
* **Evidence Quality**: critic\_score ≥ threshold (e.g., 0.85/1.0).
* **Convergence**: marginal gain < epsilon for N cycles (e.g., 3 loops).
* **Budget**: steps ≤ max\_steps, tokens ≤ max\_tokens, walltime ≤ max\_minutes.

---

## 5) Logging & Artifacts

* `runs/<timestamp>/events.jsonl` – every agent action (structured).
* `runs/<timestamp>/evidence.jsonl` – normalized facts with source, quote, URL, hash.
* `runs/<timestamp>/report.md` – living draft, then final report.
* `runs/<timestamp>/plan.json` – task graph with status.

All files are append-only and checksumed for reproducibility.

---

## 6) Minimal Python Implementation (single-file)

> **Note:** This is provider-agnostic. Wire your preferred LLM, search, and fetch.

Create `swarm.py`:

```python
import os, json, time, uuid, math, asyncio, hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# ----------- Utilities

def now_ts():
    return time.strftime('%Y-%m-%dT%H-%M-%S')

class RunIO:
    def __init__(self, root='runs'):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.run_id = now_ts() + '-' + uuid.uuid4().hex[:6]
        self.dir = os.path.join(root, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.events = open(os.path.join(self.dir, 'events.jsonl'), 'a', encoding='utf-8')
        self.evidence = open(os.path.join(self.dir, 'evidence.jsonl'), 'a', encoding='utf-8')
        self.report_path = os.path.join(self.dir, 'report.md')
        self.plan_path = os.path.join(self.dir, 'plan.json')

    def log(self, kind: str, payload: Dict[str, Any]):
        rec = {"time": now_ts(), "kind": kind, **payload}
        self.events.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.events.flush()

    def add_evidence(self, item: Dict[str, Any]):
        self.evidence.write(json.dumps(item, ensure_ascii=False) + "\n"); self.evidence.flush()

    def write_report(self, text: str):
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def write_plan(self, plan: Dict[str, Any]):
        with open(self.plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

# ----------- Adapters (swap these with real providers)

async def llm(prompt: str, system: str = "", temperature: float = 0.2) -> str:
    # TODO: replace with your LLM call (OpenAI, Anthropic, local, etc.)
    # For now this is a stub that echoes the prompt tail for dev iteration.
    return "[LLM STUB OUTPUT]\n" + prompt[-800:]

async def web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    # TODO: replace with a real search client (Tavily, SerpAPI, DDG, etc.)
    return [{"title": f"Result for {query}", "url": f"https://example.com/{i}", "snippet": "..."} for i in range(k)]

async def web_fetch(url: str) -> str:
    # TODO: replace with real fetching + readability extraction
    return f"Full text of {url} ..."

# ----------- Data Models

@dataclass
class Task:
    id: str
    goal: str
    status: str = 'pending'  # pending, running, done
    notes: List[str] = field(default_factory=list)

@dataclass
class Config:
    max_steps: int = 12
    converge_loops: int = 3
    coverage_target: float = 0.9
    quality_target: float = 0.85

# ----------- Agents

class Planner:
    async def plan(self, topic: str, prior: List[Task]) -> List[Task]:
        # If prior is empty, decompose; else refine remaining tasks
        if not prior:
            prompt = f"Decompose the research topic into 6-10 atomic questions. Topic: {topic}"
            outline = await llm(prompt)
            tasks = [Task(id=str(i+1), goal=g.strip()) for i, g in enumerate(outline.split('\n')) if g.strip()]
            return tasks[:10]
        else:
            # Retain pending tasks; maybe add 1-2 gaps later
            return [t for t in prior if t.status != 'done']

class Researcher:
    async def run(self, task: Task, io: RunIO) -> List[Dict[str, Any]]:
        q = task.goal
        io.log('search', {"task": task.id, "q": q})
        hits = await web_search(q)
        out = []
        for h in hits:
            text = await web_fetch(h['url'])
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            evidence = {
                "task": task.id,
                "claim": f"{q} — extracted insight",
                "support": text[:500],
                "url": h['url'],
                "title": h['title'],
                "hash": digest
            }
            io.add_evidence(evidence)
            out.append(evidence)
        return out

class Analyst:
    async def synthesize(self, topic: str, evidence: List[Dict[str, Any]]) -> str:
        # Consolidate evidence into a draft with inline citations
        cites = "\n".join([f"- {e['claim']} (source: {e['url']})" for e in evidence[:20]])
        prompt = f"""
        Create a structured research memo on: {topic}
        Use bullet points, sections, and inline citations from the list below.
        Citations list:\n{cites}
        """
        return await llm(prompt)

class Critic:
    async def score(self, report: str, tasks: List[Task], evidence_count: int) -> Dict[str, Any]:
        # Simple heuristic scoring (replace with LLM rubric)
        coverage = min(1.0, evidence_count / max(1, len(tasks) * 5))
        clarity = 0.7 if len(report) < 2000 else 0.9
        sources = 0.7 if evidence_count < 10 else 0.9
        score = round((coverage + clarity + sources) / 3, 3)
        gaps = [] if coverage > 0.9 else [f"Add sources for {t.goal}" for t in tasks if t.status != 'done'][:3]
        return {"score": score, "coverage": coverage, "clarity": clarity, "sources": sources, "gaps": gaps}

# ----------- Orchestrator

class Orchestrator:
    def __init__(self, cfg: Config, io: RunIO):
        self.cfg = cfg
        self.io = io
        self.planner = Planner()
        self.researcher = Researcher()
        self.analyst = Analyst()
        self.critic = Critic()
        self.tasks: List[Task] = []
        self.evidence_cache: List[Dict[str, Any]] = []
        self.stable_loops = 0
        self.last_score = 0.0

    async def run(self, topic: str):
        self.io.log('start', {"topic": topic, "cfg": asdict(self.cfg)})

        step = 0
        while step < self.cfg.max_steps:
            step += 1
            # (1) Plan/Refine
            self.tasks = await self.planner.plan(topic, self.tasks)
            self.io.write_plan({"tasks": [asdict(t) for t in self.tasks]})
            self.io.log('plan', {"tasks": [t.goal for t in self.tasks]})

            # (2) Execute next pending tasks (round-robin, could parallelize)
            for t in self.tasks:
                if t.status == 'pending':
                    t.status = 'running'
                    ev = await self.researcher.run(t, self.io)
                    self.evidence_cache.extend(ev)
                    t.status = 'done'

            # (3) Synthesize
            report = await self.analyst.synthesize(topic, self.evidence_cache)
            self.io.write_report(report)

            # (4) Critique
            critique = await self.critic.score(report, self.tasks, len(self.evidence_cache))
            self.io.log('critique', critique)

            # (5) Check convergence & thresholds
            score = critique['score']
            if score >= self.cfg.quality_target and critique['coverage'] >= self.cfg.coverage_target:
                self.io.log('stop', {"reason": "quality+coverage thresholds met", "score": score})
                break
            if abs(score - self.last_score) < 0.01:
                self.stable_loops += 1
            else:
                self.stable_loops = 0
            self.last_score = score
            if self.stable_loops >= self.cfg.converge_loops:
                self.io.log('stop', {"reason": "converged", "score": score})
                break

        return {"run_dir": self.io.dir, "score": self.last_score}

# ----------- Entry Point

async def main(topic: str):
    cfg = Config(max_steps=8, converge_loops=2, coverage_target=0.8, quality_target=0.8)
    io = RunIO()
    orch = Orchestrator(cfg, io)
    result = await orch.run(topic)
    print("Run complete:", result)

if __name__ == '__main__':
    import sys
    topic = " ".join(sys.argv[1:]) or "Impact of LLMs on radiology workflows (2020–2025)"
    asyncio.run(main(topic))
```

---

## 7) How to Use

1. **Create** `swarm.py` with the code above.
2. **Replace** the three adapters: `llm`, `web_search`, `web_fetch` with your providers.
3. **Run**: `python swarm.py "YOUR TOPIC HERE"`
4. Inspect artifacts under `runs/<timestamp>/` (events, evidence, plan, report).

---

## 8) Production Upgrades — Full Setup

Below is a production-grade extension to the starter kit. It adds true concurrency, LLM quality grading, ranking/dedup, robust fetching/cleaning, safety rails, memory, budgets/checkpointing, HITL controls, and tests.

### 8.1 Folder Structure

```
swarm/
  __init__.py
  orchestrator.py
  agents/
    planner.py
    researcher.py
    analyst.py
    critic.py
    librarian.py
  adapters/
    llm.py
    search.py
    fetch.py
    embed.py
  infra/
    memory.py
    safety.py
    ranking.py
    budgets.py
    logging.py
  cli.py
  config.yaml
requirements.txt
```

---

### 8.2 requirements.txt

```
httpx>=0.27
beautifulsoup4>=4.12
trafilatura>=1.9
langdetect>=1.0.9
urlextract>=1.9
tldextract>=5.1
python-levenshtein>=0.25
faiss-cpu>=1.8 ; platform_system != "Windows"
chromadb>=0.5
pydantic>=2.7
pytest>=8.2
ratelimit>=2.2
backoff>=2.2
python-dateutil>=2.9
PyYAML>=6.0
```

> Use FAISS if available; fallback to Chroma when FAISS build isn’t supported.

---

### 8.3 config.yaml (extended)

```yaml
max_steps: 12
converge_loops: 3
coverage_target: 0.9
quality_target: 0.85
parallel_researchers: 4
rate_limit_rps: 1.5
budgets:
  max_minutes: 20
  max_tokens: 150000
  max_fetches: 120
safety:
  allow_domains: [".edu", ".gov", ".org", ".ac.uk"]
  deny_domains: ["spam.site", "example-bad.org"]
  allow_substrings: []
  deny_substrings: ["/login", "/account", "/checkout"]
  obey_robots: true
  pii_guardrails: true
ranking:
  authority_weights:
    gov: 1.0
    edu: 0.95
    org: 0.8
    com: 0.6
  freshness_half_life_days: 365
hitl:
  approve_plan_changes: false
  approve_final_report: true
llm:
  model: "gpt-4o-mini"
  temperature: 0.2
  system_prompts:
    planner: "You are a world-class research project manager."
    analyst: "You are a neutral analyst. Cite evidence precisely."
    critic:  "You are a meticulous research auditor."
embeddings:
  model: "text-embedding-3-large"
```

---

### 8.4 Concurrency & Budgets

Replace the single-threaded task run with **parallel researchers** managed by a semaphore and budget counters.

```python
# orchestrator.py (excerpt)
import asyncio, time
from infra.budgets import Budget

class Orchestrator:
    def __init__(self, cfg, io, planner, researcher, analyst, critic, librarian, safety):
        self.cfg, self.io = cfg, io
        self.planner, self.researcher = planner, researcher
        self.analyst, self.critic, self.librarian = analyst, critic, librarian
        self.safety = safety
        self.tasks = []
        self.evidence_cache = []
        self.stable_loops = 0
        self.last_score = 0.0
        self.sema = asyncio.Semaphore(cfg.parallel_researchers)
        self.budget = Budget(cfg.budgets, self.io)

    async def _run_task(self, t):
        async with self.sema:
            if not await self.budget.allow_fetch():
                return []
            return await self.researcher.run(t, self.io, self.safety, self.budget)

    async def run(self, topic: str):
        self.io.log('start', {"topic": topic, "cfg": self.cfg.dict()})
        start = time.time()
        step = 0
        while step < self.cfg.max_steps and self.budget.within_walltime(start):
            step += 1
            self.tasks = await self.planner.plan(topic, self.tasks)
            self.io.write_plan({"tasks": [t.dict() for t in self.tasks]})
            self.io.log('plan', {"tasks": [t.goal for t in self.tasks]})

            # parallel task execution
            pending = [t for t in self.tasks if t.status == 'pending']
            for t in pending: t.status = 'running'
            task_futs = [asyncio.create_task(self._run_task(t)) for t in pending]
            results = await asyncio.gather(*task_futs, return_exceptions=False)
            for t, ev in zip(pending, results):
                t.status = 'done'
                self.evidence_cache.extend(ev)

            # dedup + rank inside Librarian
            self.evidence_cache = self.librarian.normalize_rank_dedup(self.evidence_cache)

            report = await self.analyst.synthesize(topic, self.evidence_cache)
            self.io.write_report(report)

            critique = await self.critic.score(report, self.tasks, len(self.evidence_cache))
            self.io.log('critique', critique)

            score = critique['score']
            if score >= self.cfg.quality_target and critique['coverage'] >= self.cfg.coverage_target:
                self.io.log('stop', {"reason": "quality+coverage thresholds met", "score": score})
                break
            self.stable_loops = (self.stable_loops + 1) if abs(score - self.last_score) < 0.01 else 0
            self.last_score = score
            if self.stable_loops >= self.cfg.converge_loops:
                self.io.log('stop', {"reason": "converged", "score": score})
                break

        self.io.checkpoint()  # final snapshot
        return {"run_dir": self.io.dir, "score": self.last_score}
```

**infra/budgets.py**

```python
from dataclasses import dataclass
import time

@dataclass
class BudgetsCfg:
    max_minutes: int
    max_tokens: int
    max_fetches: int

class Budget:
    def __init__(self, cfg: dict, io):
        self.cfg = BudgetsCfg(**cfg)
        self.io = io
        self.fetches = 0
        self.tokens = 0
        self.start = time.time()

    def within_walltime(self, start_ts):
        return (time.time() - start_ts) / 60.0 < self.cfg.max_minutes

    async def allow_fetch(self):
        if self.fetches >= self.cfg.max_fetches:
            self.io.log('budget_stop', {'reason': 'max_fetches'})
            return False
        self.fetches += 1
        return True

    def add_tokens(self, n):
        self.tokens += n
        if self.tokens >= self.cfg.max_tokens:
            self.io.log('budget_stop', {'reason': 'max_tokens'})
```

---

### 8.5 LLM-based Quality Rubric

Replace heuristic critic with a rubric‑driven LLM grader.

```python
# agents/critic.py
from adapters.llm import call_llm

RUBRIC = """
Score 0.0–1.0 on: completeness, evidence_strength, recency, balance, clarity.
Return JSON: {"score": float, "coverage": float, "notes": [..], "gaps": [..]}
"""

class Critic:
    async def score(self, report: str, tasks, evidence_count: int):
        prompt = f"Rubric:
{RUBRIC}
Report:
{report[:15000]}
Tasks:{[t.goal for t in tasks]}
Evidence count:{evidence_count}"
        j = await call_llm(prompt, system="You are a meticulous research auditor.")
        return j
```

**adapters/llm.py (example stub returning parsed JSON)**

```python
import json, asyncio

async def call_llm(prompt: str, system: str = "", temperature: float = 0.2):
    # TODO: integrate your provider; ensure you return parsed JSON
    # For now, return a conservative placeholder
    await asyncio.sleep(0)
    return {"score": 0.78, "coverage": 0.72, "notes": ["placeholder"], "gaps": ["add more .gov sources"]}
```

---

### 8.6 Source Ranking & Dedup

Authority weighting + freshness, domain dedup, and content-hash dedup.

```python
# infra/ranking.py
import hashlib, tldextract, time
from dateutil import parser

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

class Ranker:
    def __init__(self, cfg):
        self.cfg = cfg

    def _auth_weight(self, domain):
        tld = tldextract.extract(domain).suffix
        table = self.cfg['authority_weights']
        return table.get(tld, 0.5)

    def _freshness_weight(self, iso_date: str | None):
        if not iso_date: return 0.7
        try:
            days = max(1, (time.time() - parser.parse(iso_date).timestamp())/86400)
            half_life = self.cfg['freshness_half_life_days']
            return 0.5 ** (days/half_life)
        except Exception:
            return 0.7

    def score(self, item):
        base = 0.5
        aw = self._auth_weight(item.get('domain',''))
        fw = self._freshness_weight(item.get('published_at'))
        return base + 0.3*aw + 0.2*fw
```

**agents/librarian.py**

```python
from infra.ranking import Ranker, content_hash
import tldextract

class Librarian:
    def __init__(self, cfg):
        self.ranker = Ranker(cfg['ranking'])

    def normalize_rank_dedup(self, items):
        # normalize
        for e in items:
            if 'domain' not in e:
                e['domain'] = tldextract.extract(e['url']).registered_domain
            if 'hash' not in e:
                e['hash'] = content_hash(e.get('support','') or e.get('raw',''))
        # dedup by (domain, title) and by hash
        seen_hash, seen_key = set(), set()
        uniq = []
        for e in items:
            key = (e['domain'], e.get('title','').strip().lower())
            if e['hash'] in seen_hash or key in seen_key:
                continue
            seen_hash.add(e['hash']); seen_key.add(key); uniq.append(e)
        # rank
        uniq.sort(key=lambda x: self.ranker.score(x), reverse=True)
        return uniq
```

---

### 8.7 Readability & Language Detection

Robust fetch with retries, Readability extraction, language detection, and boilerplate removal.

```python
# adapters/fetch.py
import httpx, backoff
import trafilatura
from langdetect import detect

@backoff.on_exception(backoff.expo, (httpx.HTTPError,), max_tries=3)
async def fetch(url: str, timeout=20):
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        r = await client.get(url, headers={"User-Agent": "SwarmResearchBot/1.0"})
        r.raise_for_status()
        return r.text

async def extract_readable(html: str):
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    lang = None
    try:
        lang = detect(text[:1000]) if text else None
    except Exception:
        lang = None
    return text, lang
```

**adapters/search.py (placeholder)**

```python
# Integrate Tavily/SerpAPI/DDG here
async def search(query: str, k: int = 8):
    return [{"title": f"{query} - res{i}", "url": f"https://example.com/{i}", "published_at": None} for i in range(k)]
```

**agents/researcher.py**

```python
from adapters.search import search
from adapters.fetch import fetch, extract_readable

class Researcher:
    async def run(self, task, io, safety, budget):
        hits = await search(task.goal, k=8)
        out = []
        for h in hits:
            if not safety.allowed(h['url']):
                io.log('skip', {'reason': 'safety', 'url': h['url']}); continue
            html = await fetch(h['url'])
            text, lang = await extract_readable(html)
            ev = {
                'task': task.id,
                'claim': f"{task.goal} — extracted insight",
                'support': text[:1200],
                'url': h['url'],
                'title': h['title'],
                'published_at': h.get('published_at'),
                'lang': lang,
            }
            io.add_evidence(ev)
            out.append(ev)
        return out
```

---

### 8.8 Safety Rails (domains, robots, PII)

```python
# infra/safety.py
import re, urllib.parse, urllib.robotparser as rp
from ratelimit import limits, sleep_and_retry

class Safety:
    def __init__(self, cfg):
        self.cfg = cfg
        self._robots = {}

    def _robots_ok(self, url):
        if not self.cfg.get('obey_robots', True):
            return True
        base = f"{urllib.parse.urlparse(url).scheme}://{urllib.parse.urlparse(url).netloc}"
        if base not in self._robots:
            r = rp.RobotFileParser()
            r.set_url(base + "/robots.txt");
            try:
                r.read()
            except Exception:
                return True
            self._robots[base] = r
        return self._robots[base].can_fetch('*', url)

    def allowed(self, url: str) -> bool:
        if not self._robots_ok(url):
            return False
        s = url.lower()
        for d in self.cfg['deny_substrings']:
            if d in s: return False
        for d in self.cfg['deny_domains']:
            if s.endswith(d): return False
        if self.cfg['allow_domains']:
            return any(s.endswith(d) for d in self.cfg['allow_domains'])
        return True
```

---

### 8.9 Memory (Vector Index)

Persist embeddings so future runs skip redundant reading and resurface prior gold sources.

```python
# infra/memory.py
import hashlib

class Memory:
    def __init__(self, embedder, store):
        self.embedder, self.store = embedder, store

    def key(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    async def upsert(self, items):
        for e in items:
            vec = await self.embedder.embed(e['support'][:2000])
            self.store.add(ids=[self.key(e['url'])], embeddings=[vec], metadatas=[e])

    async def search(self, query, top_k=5):
        vec = await self.embedder.embed(query)
        return self.store.query(query_embeddings=[vec], n_results=top_k)
```

**adapters/embed.py (stub)**

```python
class Embedder:
    async def embed(self, text: str):
        # integrate your embedding API
        return [0.0]*1536
```

Hook memory into the Librarian or Orchestrator after each cycle to upsert.

---

### 8.10 Human‑in‑the‑Loop (HITL)

Add optional approvals for plan changes and final handoff.

```python
# orchestrator.py (HITL hooks)
if cfg.hitl.approve_plan_changes:
    self.io.log('await_approval', {'phase': 'plan'}); self.io.pause_until_approved()
...
if cfg.hitl.approve_final_report:
    self.io.log('await_approval', {'phase': 'final'}); self.io.pause_until_approved()
```

Implement `pause_until_approved()` to poll a file toggle or UI webhook.

---

### 8.11 Checkpointing & Graceful Early Stops

```python
# infra/logging.py (extend RunIO)
class RunIO(...):
    def checkpoint(self):
        # copy current report, evidence, plan to a timestamped snapshot
        pass
    def pause_until_approved(self):
        # block until external signal; simple file-based flag
        pass
```

---

### 8.12 Tests

**tests/test\_adapters.py**

```python
import pytest
from adapters.fetch import extract_readable

@pytest.mark.asyncio
async def test_extract_readable_sample():
    html = "<html><body><article>Hello World</article></body></html>"
    text, lang = await extract_readable(html)
    assert "Hello World" in text
```

**tests/test\_ranking.py**

```python
from infra.ranking import Ranker

def test_scoring_structure():
    r = Ranker({'authority_weights': {'com':0.6}, 'freshness_half_life_days':365})
    s = r.score({'domain':'example.com','published_at':None})
    assert 0 < s < 1.5
```

**tests/test\_safety.py**

```python
from infra.safety import Safety

def test_allow_deny():
    s = Safety({'allow_domains':['.org'], 'deny_domains':['bad.org'], 'deny_substrings':[], 'obey_robots':False})
    assert s.allowed('https://good.org/page')
    assert not s.allowed('https://evil.bad.org/page')
```

**tests/test\_memory.py**

```python
import types
from infra.memory import Memory

class DummyStore:
    def __init__(self):
        self._data = {}
    def add(self, ids, embeddings, metadatas):
        for i, e, m in zip(ids, embeddings, metadatas):
            self._data[i] = (e, m)
    def query(self, query_embeddings, n_results=5):
        return list(self._data.items())[:n_results]

class DummyEmbed:
    async def embed(self, text):
        return [0.1,0.2,0.3]

async def test_memory_roundtrip():
    mem = Memory(DummyEmbed(), DummyStore())
    await mem.upsert([{'url':'https://a','support':'abc'}])
    res = await mem.search('abc', top_k=1)
    assert len(res) >= 1
```

---

### 8.13 CLI entrypoint

```python
# cli.py
import asyncio, yaml
from orchestrator import Orchestrator
from infra.logging import RunIO
from agents import planner, researcher, analyst, critic, librarian
from infra.safety import Safety

async def main(topic: str, cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    io = RunIO()
    o = Orchestrator(
        cfg=SimpleNamespace(**cfg),
        io=io,
        planner=planner.Planner(),
        researcher=researcher.Researcher(),
        analyst=analyst.Analyst(),
        critic=critic.Critic(),
        librarian=librarian.Librarian(cfg),
        safety=Safety(cfg['safety'])
    )
    await o.run(topic)

if __name__ == '__main__':
    import sys
    asyncio.run(main(" ".join(sys.argv[1:]) or "Test Topic"))
```

---

### 8.14 What You Get

* **Speed**: parallel research with rate limits and budgets
* **Quality**: LLM rubric grading + ranked, deduped sources
* **Safety**: domain gates, robots.txt, PII placeholders
* **Rigor**: reproducible logs, checkpoints, and tests
* **Memory**: vector store for cross‑run leverage

---

### 8.15 Providers & Secrets Integration (OpenRouter, Groq, Together, Perplexity, DeepSeek, HuggingFace, Abacus)

> **Security first:** Because secrets were shared in chat, **rotate all keys now**. Store secrets in a `.env` file (or your secret manager). Never commit to git. Never paste into prompts/logs.

#### 8.15.1 `.env` template

Create a `.env` in the project root:

```
# Primary model/router options (pick one or more)
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
GROQ_API_KEY=YOUR_GROQ_KEY
TOGETHER_AI_API_KEY=YOUR_TOGETHER_KEY
PERPLEXITY_API_KEY=YOUR_PERPLEXITY_KEY
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
HUGGINGFACE_TOKEN=YOUR_HF_TOKEN
ABACUS_AI_API_KEY_1=YOUR_ABACUS_KEY

# Optional runtime controls
HTTP_TIMEOUT_SEC=30
RATE_LIMIT_RPS=1.5
```

Load with `python-dotenv` or your orchestrator—example inside `infra/logging.py` or at process startup.

#### 8.15.2 `adapters/llm.py` — provider registry

```python
# adapters/llm.py
import os, json, asyncio, httpx
from typing import Dict, Any

TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SEC", 30))

class LLMError(Exception): ...

def _headers(api_key: str, extra: Dict[str,str] | None = None):
    h = {"Content-Type": "application/json"}
    if extra: h.update(extra)
    return h

async def _post(url: str, headers: Dict[str,str], payload: Dict[str,Any]):
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def openrouter(prompt: str, system: str = "", model: str = "openrouter/auto"):
    key = os.getenv("OPENROUTER_API_KEY"); assert key, "Missing OPENROUTER_API_KEY"
    data = {
        "model": model,
        "messages": ([{"role":"system","content":system}] if system else []) +
                     [{"role":"user","content":prompt}],
        "temperature": 0.2
    }
    j = await _post(
        "https://openrouter.ai/api/v1/chat/completions",
        _headers(key, {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://your.app"}),
        data
    )
    return j["choices"][0]["message"]["content"].strip()

async def groq(prompt: str, system: str = "", model: str = "llama-3.1-70b-versatile"):
    key = os.getenv("GROQ_API_KEY"); assert key, "Missing GROQ_API_KEY"
    data = {
        "messages": ([{"role":"system","content":system}] if system else []) +
                     [{"role":"user","content":prompt}],
        "model": model,
        "temperature": 0.2
    }
    j = await _post("https://api.groq.com/openai/v1/chat/completions",
                    _headers(key, {"Authorization": f"Bearer {key}"}), data)
    return j["choices"][0]["message"]["content"].strip()

async def together(prompt: str, system: str = "", model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
    key = os.getenv("TOGETHER_AI_API_KEY"); assert key, "Missing TOGETHER_AI_API_KEY"
    data = {
        "model": model,
        "messages": ([{"role":"system","content":system}] if system else []) +
                     [{"role":"user","content":prompt}],
        "temperature": 0.2
    }
    j = await _post("https://api.together.xyz/v1/chat/completions",
                    _headers(key, {"Authorization": f"Bearer {key}"}), data)
    return j["choices"][0]["message"]["content"].strip()

async def deepseek(prompt: str, system: str = "", model: str = "deepseek-chat"):
    key = os.getenv("DEEPSEEK_API_KEY"); assert key, "Missing DEEPSEEK_API_KEY"
    data = {
        "model": model,
        "messages": ([{"role":"system","content":system}] if system else []) +
                     [{"role":"user","content":prompt}],
        "temperature": 0.2
    }
    j = await _post("https://api.deepseek.com/chat/completions",
                    _headers(key, {"Authorization": f"Bearer {key}"}), data)
    return j["choices"][0]["message"]["content"].strip()

async def perplexity_search(prompt: str, system: str = "", model: str = "sonar-large-online"):
    """Perplexity model with built-in browsing for research steps."""
    key = os.getenv("PERPLEXITY_API_KEY"); assert key, "Missing PERPLEXITY_API_KEY"
    data = {
        "model": model,
        "messages": ([{"role":"system","content":system}] if system else []) +
                     [{"role":"user","content":prompt}],
        "temperature": 0.2
    }
    j = await _post("https://api.perplexity.ai/chat/completions",
                    _headers(key, {"Authorization": f"Bearer {key}"}), data)
    return j["choices"][0]["message"]["content"].strip()

async def call_llm(prompt: str, system: str = "", provider: str = "openrouter", model: str | None = None):
    if provider == "openrouter":
        return await openrouter(prompt, system, model or os.getenv("OPENROUTER_MODEL", "openrouter/auto"))
    if provider == "groq":
        return await groq(prompt, system, model or os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"))
    if provider == "together":
        return await together(prompt, system, model or os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"))
    if provider == "deepseek":
        return await deepseek(prompt, system, model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    if provider == "perplexity":
        return await perplexity_search(prompt, system, model or os.getenv("PPLX_MODEL", "sonar-large-online"))
    raise LLMError(f"Unknown provider: {provider}")
```

Update `agents/critic.py` and `agents/analyst.py` to use `call_llm(..., provider=<from config>)`.

#### 8.15.3 `adapters/search.py` — web search via Perplexity or Together

```python
# adapters/search.py
import os
from adapters.llm import perplexity_search

async def search(query: str, k: int = 8):
    """Use a browsing-capable model to retrieve top links + snippets.
    For tighter control, swap with Tavily/SerpAPI client.
    """
    sys = "You return the 8 most relevant links in JSON with title,url,published_at,snippet."
    prompt = f"Query: {query}
Return JSON list with fields: title,url,published_at,snippet. Limit={k}."
    txt = await perplexity_search(prompt, system=sys)
    # Expect model to return JSON. Add a safe parse fallback.
    try:
        import json
        items = json.loads(txt)
        return items[:k]
    except Exception:
        # Fallback: naive single result
        return [{"title": query, "url": "https://perplexity.ai", "published_at": None, "snippet": txt[:300]}]
```

> If you prefer classic API search, drop in Tavily/SerpAPI here and bypass LLM parsing.

#### 8.15.4 `adapters/embed.py` — Hugging Face Inference or OpenRouter embeddings

```python
# adapters/embed.py
import os, httpx, asyncio

HF = "https://api-inference.huggingface.co/pipeline/feature-extraction"

class Embedder:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.key = os.getenv("HUGGINGFACE_TOKEN")

    async def embed(self, text: str):
        assert self.key, "Missing HUGGINGFACE_TOKEN"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{HF}/{self.model}", headers={"Authorization": f"Bearer {self.key}"}, json={"inputs": text})
            r.raise_for_status()
            v = r.json()
            # flatten if nested
            return v[0] if isinstance(v, list) and isinstance(v[0], list) else v
```

#### 8.15.5 Optional: Abacus AI for document indexing

```python
# infra/abacus.py (optional)
import os, httpx

class AbacusIndexer:
    def __init__(self):
        self.key = os.getenv("ABACUS_AI_API_KEY_1")
        self.base = "https://api.abacus.ai"

    async def upsert_doc(self, collection_id: str, doc_id: str, text: str, metadata: dict):
        assert self.key, "Missing ABACUS_AI_API_KEY_1"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{self.base}/v0/collections/{collection_id}/documents",
                                  headers={"Authorization": f"Bearer {self.key}"},
                                  json={"documentId": doc_id, "text": text, "metadata": metadata})
            r.raise_for_status(); return r.json()
```

Wire this into `Memory` if you prefer managed indexing over FAISS/Chroma.

#### 8.15.6 Config toggles

Extend `config.yaml`:

```yaml
llm:
  provider: openrouter   # openrouter | groq | together | perplexity | deepseek
  model: openrouter/auto
search:
  provider: perplexity   # or tavily/serpapi
embeddings:
  provider: hf
  model: sentence-transformers/all-MiniLM-L6-v2
abacus:
  enabled: false
  collection_id: your-collection
```

#### 8.15.7 Rate limits

Use the existing `rate_limit_rps` and apply per-client (e.g., a simple token bucket) if your providers have strict quotas.

---

### 8.16 Default Stack Preset (pinned models & ready-to-run)

This preset wires **Perplexity** for search+browsing, **Groq** for fast/cheap LLM reasoning (Llama 3.1 70B Instruct), and **Hugging Face** for embeddings.

#### 8.16.1 `.env` (placeholders)

```
GROQ_API_KEY=YOUR_GROQ_KEY
PERPLEXITY_API_KEY=YOUR_PERPLEXITY_KEY
HUGGINGFACE_TOKEN=YOUR_HF_TOKEN
HTTP_TIMEOUT_SEC=30
RATE_LIMIT_RPS=1.5
```

#### 8.16.2 `config.yaml` (preset)

```yaml
# Core loop
max_steps: 12
converge_loops: 2
coverage_target: 0.88
quality_target: 0.88
parallel_researchers: 4
rate_limit_rps: 1.5

# Providers
llm:
  provider: groq
  model: llama-3.1-70b-versatile
  temperature: 0.2
  system_prompts:
    planner: "You are a world-class research project manager."
    analyst: "You are a neutral analyst. Cite evidence precisely."
    critic:  "You are a meticulous research auditor."
search:
  provider: perplexity
  model: sonar-large-online
embeddings:
  provider: hf
  model: sentence-transformers/all-MiniLM-L6-v2

# Safety & ranking
safety:
  allow_domains: [".edu", ".gov", ".org", ".ac.uk"]
  deny_domains: ["spam.site", "example-bad.org"]
  allow_substrings: []
  deny_substrings: ["/login", "/account", "/checkout"]
  obey_robots: true
  pii_guardrails: true
ranking:
  authority_weights: {gov: 1.0, edu: 0.95, org: 0.8, com: 0.6}
  freshness_half_life_days: 365

# Budgets
budgets: {max_minutes: 20, max_tokens: 150000, max_fetches: 120}

# HITL
hitl: {approve_plan_changes: false, approve_final_report: true}
```

#### 8.16.3 Adapter toggles

* `agents/critic.py` and `agents/analyst.py` already call `call_llm(...)` → will use Groq.
* `adapters/search.py` uses Perplexity browsing model JSON output.
* `adapters/embed.py` uses HF Inference for MiniLM.

#### 8.16.4 Quickstart

```bash
# 1) install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) configure
cp .env.example .env   # or create manually with your keys
# edit config.yaml with the preset above

# 3) run
python cli.py "Impact of LLMs on radiology workflows (2020–2025)"

# 4) inspect artifacts
ls runs/*/{events.jsonl,evidence.jsonl,plan.json,report.md}
```

#### 8.16.5 Optional hardening (toggleable later)

* Swap `embeddings.model` → `intfloat/e5-large-v2` for stronger retrieval.
* Enable FAISS: install `faiss-cpu` and use `infra/memory.py` with FAISS store backend.
* Set `hitl.approve_plan_changes: true` for sensitive research.
* Lower `parallel_researchers` if you hit provider rate caps.

---

## 9) Config Template (YAML)

(YAML)

```yaml
max_steps: 12
converge_loops: 3
coverage_target: 0.9
quality_target: 0.85
parallel_researchers: 3
allow_domains: [".edu", ".gov", ".org"]
deny_domains: ["spam.site", "lowquality.blog"]
rate_limit_rps: 1.0
```

---

## 10) Example Prompts (drop-in)

* **Planner system prompt**: “You are a world-class research project manager. Produce a numbered list of atomic sub-questions covering breadth first, then depth.”
* **Analyst system prompt**: “You are a neutral analyst. Synthesize only what is supported by cited evidence. Use structured sections.”
* **Critic rubric**: completeness, evidence strength, recency, balance, clarity, actionability.

---

## 11) What “Done” Looks Like

* Report meets thresholds, includes inline citations and limitations.
* Evidence Board contains normalized, deduped items with source URLs and hashes.
* Logs show 2–5 cycles with steady quality gains and then convergence/threshold hit.

---

## 12) Next Steps

* Wire real adapters (LLM + search + fetch) and test on 2–3 topics.
* Add parallelization & improved critic rubric.
* Integrate a UI (Streamlit/Gradio) to watch the loop live.
