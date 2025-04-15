import json
import re
import os
import random
import requests
from openai import OpenAI
from dotenv import load_dotenv
import tqdm
import pandas as pd

import json
import random
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



def medrxiv_fetch_articles(n=32768, date_range="2018-01-01/2022-12-20"):
    """
    Pages through the bioRxiv API from date_range start→end,
    extracts title→sentence1 and abstract→sentence2,
    and writes a CSV to output_dir/biorxiv_sentences.csv.
    """
    articles=[]
    cursor = 0

    while True:
        url = f"https://api.medrxiv.org/details/medrxiv/{date_range}/{cursor}"
        print(f"Fetching medrxiv: {url}")
        resp = requests.get(url); resp.raise_for_status()
        data = resp.json().get("collection", [])
        if not data:
            break

        for item in data:
            title   = item.get("title", "").strip()
            abstract= item.get("abstract", "").strip()
            if title and abstract:
                articles.append({'title':title,'abstract':abstract})
        cursor += len(data)
        if cursor>n:
            break
    return articles

def biomed_generate_anonymized_question_answer(content):
    """
    Use OpenAI GPT-4 to generate anonymized question-answer pairs for a Wikipedia page.
    """
    prompt = (
        f"Using the following content from a medical or biology paper, generate:\n"
        "A question that tests knowledge of the content without explicitly using the title or key phrases.\n"
        "The question should imitate a real questions for RAG"
        f"Content:\n{content}\n\n"
        "Output as JSON with key 'question' without any code formatting, backticks, or markdown."
    )
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use "gpt-4" or "gpt-3.5-turbo"
             messages=[
                {"role": "system", "content": "You are a helpful assistant that anonymizes content."},
                {"role": "user", "content": prompt}],
            max_tokens=1000,  # Adjust the max tokens as needed
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        qa_pair = json.loads(result)
        return qa_pair
    except Exception as e:
        print(f"Error generating question-answer pair: {e}")
        return {"question": None, "answer": None}


def _process_doc(doc):
    """
    Worker function: given one doc, return a retrieval entry or None.
    """
    if not isinstance(doc, dict):
        return None

    full_text = json.dumps(doc)
    qa_pair = biomed_generate_anonymized_question_answer(full_text)

    if qa_pair.get("question") and doc.get("abstract"):
        return {
            "query": qa_pair["question"],
            "corpus": doc["abstract"],
        }
    return None

def medrxiv_create_retrieval_dataset(output_file="../data/medrxiv_dataset.csv"):
    """
    Fetch Biorxiv data and generate a retrieval dataset using OpenAI GPT-4,
    in parallel across all CPU cores, with columns ['query','corpus'].
    """
    # 1) fetch & sample
    biomed_docs = medrxiv_fetch_articles(n=32768)
    biomed_docs = random.sample(biomed_docs, 16384)

    # 2) parallel map + progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_doc, biomed_docs),
                total=len(biomed_docs),
                desc="Generating QA pairs"
            )
        )

    # 3) filter out failed ones
    retrieval_data = [r for r in results if r is not None]

    # 4) save to CSV (with explicit column names)
    if output_file:
        # Create DataFrame with explicit column ordering/names
        df = pd.DataFrame(retrieval_data, columns=["query", "corpus"]).dropna()
        if len(df) > 16384:
            df = df.sample(16384)
        df.to_csv(output_file, index=False)

    return retrieval_data