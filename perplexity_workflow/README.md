# 🔍 Perplexity Workflow — Build Your Own AI Search Engine

> Search. Scrape. Chunk. Rerank. Answer. — The exact pipeline behind Perplexity, built in 100 lines of Python.

<img width="1456" height="418" alt="image" src="https://github.com/user-attachments/assets/c902a70a-6e9f-462e-9167-7c75bde61735" />

---

## What This Is

Most people treat Perplexity like magic. It isn't.

At its core, Perplexity runs a 5-step pipeline — and this project replicates that pipeline from scratch using open tools and production-grade patterns.

**Real numbers from a test run:**
```
Total chunks after splitting:   212
Unique chunks after dedup:      199
Top chunks after reranking:      15
Total tokens used in context:  2014
```

---

## Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Question Generator    │  Gemini Flash → 5 diverse sub-questions
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    Async Search         │  DuckDuckGo (DDGS) — all 5 questions in parallel
│    + Async Scrape       │  Trafilatura — all URLs scraped concurrently
└─────────────────────────┘
    │  ~212 raw chunks
    ▼
┌─────────────────────────┐
│   Chunk + Deduplicate   │  400-token chunks, 50-token overlap, MD5 dedup
└─────────────────────────┘
    │  199 unique chunks
    ▼
┌─────────────────────────┐
│       Rerank            │  Cohere rerank-v3.5 → Top 15 most relevant
└─────────────────────────┘
    │  ~2014 tokens
    ▼
┌─────────────────────────┐
│   Answer + Citations    │  Gemini Flash — 2-pass: answer → markdown + citations
└─────────────────────────┘
```

---

## Key Engineering Decisions

### 1. Multi-Query Generation
Instead of searching the raw user query once, we generate 5 semantically diverse sub-questions. This covers the topic from multiple angles — definition, relationship, methodology, examples, edge cases — giving the LLM far richer context to work with.

### 2. Async with Blocking Libraries
`DDGS` and `Trafilatura` are both synchronous/blocking libraries. Running them directly in an async function blocks the entire event loop.

**The fix:** `loop.run_in_executor()` pushes blocking calls into a `ThreadPoolExecutor`, making them awaitable. All 5 searches and all 15 scrapes run concurrently:

```python
executor = ThreadPoolExecutor(max_workers=10)

async def search_question_async(question: str):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, _ddgs_search, question)
    scrape_tasks = [
        loop.run_in_executor(executor, _trafilatura_fetch, r["href"])
        for r in results if r.get("href")
    ]
    scraped_texts = await asyncio.gather(*scrape_tasks)
```

This pattern works for **any blocking library** in an async codebase.

### 3. Deduplication Strategy
Two-layer dedup: URL+chunk index (same position on same page) and MD5 hash of first 200 characters (same content on different pages). Prevents the LLM from seeing the same information multiple times and overweighting it.

### 4. Reranking as Quality Gate
Keyword search finds chunks that *contain* matching words. Reranking finds chunks that are *semantically relevant* to the query. Cohere's reranker scores all 199 chunks and returns the top 15 — that's what goes into the LLM's context window.

### 5. Two-Pass Answer Generation
Single-pass "answer + cite sources" prompts produce hallucinated citations. We split it:
- **Pass 1:** Generate a factual answer grounded only in search results
- **Pass 2:** Rewrite with proper inline markdown citations per sentence

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM | Gemini Flash (`gemini-3-flash-preview`) | Question gen + answer gen |
| Search | DDGS (DuckDuckGo) | Web search, no API key needed |
| Scraping | Trafilatura | Clean text extraction from URLs |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | 400-token chunks, 50 overlap |
| Reranking | Cohere `rerank-v3.5` | Semantic relevance scoring |
| Token counting | tiktoken (`cl100k_base`) | Context window management |

---

## Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key
- Cohere API key

### Installation

```bash
git clone https://github.com/vishwajeetvishwakarma/AgentShortProjects.git
cd AgentShortProjects/perplexity_workflow
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

### Run

```bash
python app.py
```

You'll be prompted to enter a query. Example:
```
Enter query you want to search: how does transformer attention work
```

---

## Example Output

**Query:** `give me in full detail how ml and ai is different from each other`

```
Total chunks after splitting: 212
Unique chunks after dedup: 199
Top chunks after reranking: 15
Total tokens used in search results: 2014

### Definition and Scope
**Artificial Intelligence (AI):** AI is a broad, overarching concept that refers 
to technologies building machines capable of simulating human intelligence...
[AI vs Machine Learning - AWS, GeeksforGeeks]

**Machine Learning (ML):** ML is a specific branch and subset of AI. It focuses 
on developing algorithms that allow systems to learn from historical data...
[AWS, GeeksforGeeks, Sanfoundry]
...
```

---

## What I'd Add for Production

- **Redis caching** — cache query → top chunks (1-hour TTL), avoid repeat API calls
- **Rate limiting** — exponential backoff around DDGS to handle throttling
- **Domain-level dedup** — limit to 2 chunks per domain, prevent one site dominating context
- **Streaming output** — pipe Gemini response as a stream, don't make users wait 8 seconds
- **Hallucination detection** — second LLM call to verify answer is grounded in retrieved chunks

---

## Related

- 📝 **Full blog post:** [Build Your Own Perplexity — Substack](https://open.substack.com/pub/vishwajeetv2003/p/build-your-own-perplexity-in-100?r=2qbw99&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
- 📺 **YouTube walkthrough:** *(coming soon)*
- 🗂️ **More agent projects:** [AgentShortProjects](https://github.com/vishwajeetvishwakarma/AgentShortProjects)

---

## Author

**Vishwajeet Vishwakarma** — ML & GenAI Engineer
- 📝 [Substack](https://open.substack.com/pub/vishwajeetv2003/) | [Medium](https://medium.com/@vishwajeetv2003)
- 🐦 [Twitter/X](#)
- 💼 [LinkedIn](#)

---

*Part of the [AgentShortProjects](https://github.com/vishwajeetvishwakarma/AgentShortProjects) series — short, focused, production-minded AI agent implementations.*
