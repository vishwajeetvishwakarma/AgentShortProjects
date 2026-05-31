import os
import asyncio
import hashlib
import json
import cohere
import tiktoken

from dotenv import load_dotenv
load_dotenv()

from google.genai import types
from google import genai
from pydantic import BaseModel, Field
from ddgs import DDGS
import trafilatura
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, Markdown

client = genai.Client()
model = "gemini-3.5-flash"
co = cohere.Client(os.getenv("COHERE_API_KEY"))

query = input("Enter query you want to search : ")

question_generater_prompt = """
You are a helpful assistant for generating questions based on the asked query. You will be given a query and you have to generate 5 questions based on the query which can be used to search the web for getting more information about the query. The questions should be related to the query and should be specific enough to get relevant search results.
For example, if the query is "What is blockchain?", then the generated questions can be:
1. What are the key features of blockchain technology?
2. How does blockchain work?
3. What are the advantages of using blockchain?
4. What are the use cases of blockchain?
5. What are the challenges of blockchain technology?

Now, generate 5 questions based on the following query: 
<query>
{query}
</query>
"""

class Question(BaseModel):
    query: str = Field(..., description="The query for which questions need to be generated")

class QuestionGeneratorOutput(BaseModel):
    questions: list[Question] = Field(..., description="A list of 5 questions generated based on the query")

def generate_questions(input : str) -> QuestionGeneratorOutput:
    prompt = question_generater_prompt.format(query=input)
    res = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QuestionGeneratorOutput
        )
    )
    questions = json.loads(res.text).get("questions", [])
    return questions

questions = generate_questions(query)


executor = ThreadPoolExecutor(max_workers=10)

def _ddgs_search(question: str) -> list[dict]:
    """Your original DDGS logic, just wrapped for thread pool."""
    try:
        return list(DDGS().text(question, max_results=3))
    except Exception:
        return []

def _trafilatura_fetch(url: str) -> str | None:
    """Your original trafilatura logic, just wrapped for thread pool."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
    except Exception:
        pass
    return None

async def search_question_async(question: str) -> list[dict]:
    """Async version of your original search_question function."""
    loop = asyncio.get_event_loop()

    results = await loop.run_in_executor(executor, _ddgs_search, question)

    scrape_tasks = [
        loop.run_in_executor(executor, _trafilatura_fetch, r["href"])
        for r in results if r.get("href")
    ]
    scraped_texts = await asyncio.gather(*scrape_tasks)

    search_results = []
    for result, text in zip(results, scraped_texts):
        if text:
            search_results.append({
                "title": result["title"],
                "href": result["href"],
                "text": text,
            })
    return search_results

async def get_search_results_async(questions: list[dict]) -> list[dict]:
    """Async version of your original get_search_results function."""
    tasks = [search_question_async(q["query"]) for q in questions]
    results_per_question = await asyncio.gather(*tasks)

    search_results = []
    for question, results in zip(questions, results_per_question):
        search_results.append({
            "question": question,
            "results": results,
        })
    return search_results

res = asyncio.run(get_search_results_async(questions))


splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def chunk_results(search_results: list[dict]) -> list[dict]:
    """Flatten and chunk all results. Adds 'url' key for dedup later."""
    chunks = []
    for item in search_results:
        for result in item["results"]:
            for i, chunk_text in enumerate(splitter.split_text(result["text"])):
                chunks.append({
                    "title": result["title"],
                    "href": result["href"],
                    "text": chunk_text,
                    "chunk_index": i,
                })
    return chunks

chunks = chunk_results(res)
print(f"Total chunks after splitting: {len(chunks)}")


def deduplicate(chunks: list[dict]) -> list[dict]:
    seen_urls = set()
    seen_content = set()
    unique = []
    for chunk in chunks:
        url_key = f"{chunk['href']}:::{chunk['chunk_index']}"
        content_hash = hashlib.md5(chunk["text"][:200].encode()).hexdigest()
        if url_key in seen_urls or content_hash in seen_content:
            continue
        seen_urls.add(url_key)
        seen_content.add(content_hash)
        unique.append(chunk)
    return unique

unique_chunks = deduplicate(chunks)
print(f"Unique chunks after dedup: {len(unique_chunks)}")


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 15) -> list[dict]:
    if not chunks:
        return []
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[c["text"] for c in chunks],
        top_n=min(top_n, len(chunks)),
    )
    reranked = []
    for hit in response.results:
        chunk = chunks[hit.index].copy()
        chunk["relevance_score"] = hit.relevance_score
        reranked.append(chunk)
    return reranked

top_chunks = rerank_chunks(query, unique_chunks, top_n=15)
print(f"Top chunks after reranking: {len(top_chunks)}")

tokenizer = tiktoken.get_encoding("cl100k_base")
total_tokens = sum(len(tokenizer.encode(c["text"])) for c in top_chunks)
print(f"Total tokens used in search results: {total_tokens}")


create_answer_prompt = """You are a helpful assistant for answering questions based on the search results. You will be given a question and a list of search results. You have to read through the search results and then generate a concise and accurate answer to the question based on the information available in the search results. The answer should be based on the information available in the search results and should not include any information that is not present in the search results. The answer should be concise and to the point, and should not include any unnecessary information. The ans should be easy to understand 
Here is the question and the search results:
<question>
{question}
</question>

<search_results>
{search_results}
</search_results>
"""

class AnswerGeneratorOutput(BaseModel):
    answer: str = Field(..., description="The answer generated based on the search results")

def generate_answer(question, search_results) -> AnswerGeneratorOutput:
    prompt = create_answer_prompt.format(question=question, search_results=search_results)
    res = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AnswerGeneratorOutput
        )
    )
    answer = json.loads(res.text).get("answer", "")
    return answer

create_markdown_answer_prompt = """You will be given a question, an answer, and a set of original documents.

Your task is to rewrite the answer in markdown and add source citations for every important line, section, or factual statement so the reader can clearly see which document each part came from.

Rules:
- Cite sources inline after each sentence or bullet using the document title or a document number like [Doc 1].
- If a section uses multiple documents, cite all relevant documents.
- If information is only supported by one document, cite only that document.
- Do not add any information that is not supported by the documents.
- If a part of the answer cannot be traced to any document, omit it.
- Make the result concise, readable, and well-structured with headings and bullets where helpful.

Here is the question, the answer, and the original documents:

<question>
{question}
</question>

<answer>
{answer}
</answer>

<documents>
{documents}
</documents>

Return the final answer in markdown with inline citations for each section or sentence."""

class MarkdownAnswerOutput(BaseModel):
    markdown_answer: str = Field(..., description="The answer rewritten in markdown format with source citations")

def generate_markdown_answer(question, answer, documents) -> MarkdownAnswerOutput:
    prompt = create_markdown_answer_prompt.format(question=question, answer=answer, documents=documents)
    res = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MarkdownAnswerOutput
        )
    )
    markdown_answer = json.loads(res.text).get("markdown_answer", "")
    return markdown_answer

def generate_final_answer(question, search_results):
    answer = generate_answer(question, search_results)
    markdown_answer = generate_markdown_answer(question, answer, search_results)
    return markdown_answer

final_ans = generate_final_answer(query, top_chunks)

print(final_ans)
