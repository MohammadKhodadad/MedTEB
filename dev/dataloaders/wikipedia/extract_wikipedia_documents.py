import json
import wikipediaapi
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer, util
import random

def sample_hard_negatives(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 32,
    seed: int = 42
) -> list[int]:
    """
    Given a list of strings, returns for each index i a single index j!=i,
    randomly sampled from the top_k most semantically similar texts to texts[i].
    
    Uses a BERT‐based SentenceTransformer under the hood.
    """
    # 1. Load model & embed all texts
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    # 2. Semantic search: for each embedding, find top_k+1 hits (including itself)
    hits = util.semantic_search(
        query_embeddings=embeddings,
        corpus_embeddings=embeddings,
        top_k=top_k + 1  # +1 so we can drop the self‐match
    )
    
    random.seed(seed)
    hard_negatives = []
    
    # 3. For each list of hits, drop self (where corpus_id == query_id) then sample one
    for i, hit_list in enumerate(hits):
        # hit_list is a list of dicts: { "corpus_id": int, "score": float }
        # filter out self-match
        candidates = [h["corpus_id"] for h in hit_list if h["corpus_id"] != i]
        if not candidates:
            # fallback: pick any other index
            candidates = [j for j in range(len(texts)) if j != i]
        hard_negatives.append(random.choice(candidates))
    
    return hard_negatives


# Initialize the Wikipedia API for English
wiki_wiki = wikipediaapi.Wikipedia('Anonymous Name')



def wiki_clean_text(text):

    # Remove non-alphanumeric characters (keep letters, numbers, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]:', '', text)
    
    # Replace multiple spaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces and convert to lowercase
    text = text.strip().lower()
    
    return text

def save_data_as_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Dataset saved as {filename}")

def wiki_fetch_wikipedia_page_sections(title):
    page = wiki_wiki.page(title)
    if not page.exists():
        return None

    # Extract sections recursively
    def wiki_extract_sections(sections, level=0):
        content = {}
        for section in sections:
            content[section.title] = section.text
            subsections = wiki_extract_sections(section.sections, level + 1)
            if subsections:
                content.update(subsections)
        return content

    return wiki_extract_sections(page.sections)


def wiki_fetch_pages_in_category(category_name, max_pages=100):
    category = wiki_wiki.page(f"Category:{category_name}")
    pages = category.categorymembers
    documents = {}

    for i, (title, page) in enumerate(pages.items()):
        if page.ns == 0:  # Only consider main namespace pages (articles)
            documents[title] = wiki_fetch_wikipedia_page_sections(title)
            if len(documents) >= max_pages:
                break
    return documents

def wiki_fetch_pages_in_category_recursive(category_name, max_pages=100, max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return {}
    
    category = wiki_wiki.page(f"Category:{category_name}")
    if not category.exists():
        return {}

    documents = {}
    pages = category.categorymembers

    for title, page in pages.items():
        if page.ns == 0:  # Main namespace pages (articles)
            documents[title] = wiki_fetch_wikipedia_page_sections(title)
            documents[title]['c_depth']=current_depth
            if len(documents) >= max_pages:
                break
        elif page.ns == 14:  # Subcategories
            subcategory_documents = wiki_fetch_pages_in_category_recursive(
                page.title.replace("Category:", ""),
                max_pages=max_pages - len(documents),
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            documents.update(subcategory_documents)
            if len(documents) >= max_pages:
                break
    
    return documents

def wiki_fetch_data_from_categories(categories, max_pages_per_category=50,max_depth=1,output_dir="../data/wikipedia_dataset.csv",thr=0.7):
    all_documents = {'text':[],"label":[]}

    for category_name in categories:
        print(f"Fetching pages from category: {category_name}")
        category_documents = wiki_fetch_pages_in_category_recursive(category_name, max_pages=max_pages_per_category, max_depth=max_depth)
        # all_documents[category_name]=category_documents
        for sub_category_name, sub_documents in category_documents.items():
            
            texts=list(sub_documents.values())
            
            texts=[wiki_clean_text(text) for text in texts if isinstance(text,str) ]
            texts=[text for text in texts if len(text)>50 ]
            labels=[category_name for _ in range(len(texts))]
            # print(texts)
            all_documents['text'].extend(texts)
            all_documents['label'].extend(labels)
    # print(all_documents)
    all_documents=pd.DataFrame(all_documents)
    result=pd.DataFrame(all_documents)
    result=result.dropna()
    if len(result)>8192:
        result=result.sample(8192)
    result.to_csv(output_dir)
    
    return all_documents




def wiki_generate_anonymized_question_answer(content, page_title):
    """
    Use OpenAI GPT-4 to generate anonymized question-answer pairs for a Wikipedia page.
    """
    prompt = (
        f"Using the following content from a Wikipedia page titled '{page_title}', generate:\n"
        "1. A question that tests knowledge of the content without explicitly using the title or key phrases.\n"
        "2. An answer to the question that accurately reflects the content without reusing exact keywords or phrases.\n\n"
        f"Content:\n{content}\n\n"
        "Output as JSON with keys 'question' and 'answer' without any code formatting, backticks, or markdown."
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

def wiki_create_retrieval_dataset(categories, max_pages_per_category=50, max_depth=1, output_file="../data/wikipedia_retrieval_dataset.json"):
    """
    Fetch Wikipedia data and generate a retrieval dataset using OpenAI GPT-4.
    """
    wiki_documents = wiki_fetch_pages_in_category_recursive(categories[0], max_pages_per_category, max_depth)
    retrieval_data = []

    for page_title, content in wiki_documents.items():
        if isinstance(content, dict):
            # Combine all section content into a single text for generation
            full_text = json.dumps(content)
            qa_pair = wiki_generate_anonymized_question_answer(full_text, page_title)

            if qa_pair["question"] and qa_pair["answer"]:
                retrieval_data.append({
                    "query": qa_pair["question"],
                    "corpus": qa_pair["answer"],
                    "source_title": page_title
                })

    # Save retrieval dataset
    # save_data_as_json(retrieval_data, output_file)
    result=pd.DataFrame(retrieval_data)
    result=result.dropna()
    if len(result)>16384:
        result=result.sample(16384)
    result.to_csv(output_file)




def wiki_generate_sentence_pair(content, page_title):
    """
    Use OpenAI GPT-4 to generate two sentences from a document.
    """
    prompt = (
        f"Using the following content from a Wikipedia page titled '{page_title}', "
        "generate two related sentences based on the content. Ensure the sentences convey meaningful information. "
        "and are closely related to each other. Ensure that the sentences do not have the same keywords or the same phrasing.\n\n"
        f"Content:\n{content}\n\n"
        "Output as JSON with keys 'sentence1' and 'sentence2' without any code formatting, backticks, or markdown."
    )
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use "gpt-4" or "gpt-3.5-turbo"
             messages=[
                {"role": "system", "content": "You are a helpful assistant that generates meaningful sentences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  # Adjust the max tokens as needed
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        sentence_pair = json.loads(result)
        return sentence_pair
    except Exception as e:
        print(f"Error generating sentence pair: {e}")
        return {"sentence1": None, "sentence2": None}

def wiki_create_pair_classification_data(categories, max_pages_per_category=50, max_depth=1, output_file="../data/wikipedia_pair_classification.json"):
    """
    Create pair classification data using sentences from Wikipedia pages.
    """
    wiki_documents = wiki_fetch_pages_in_category_recursive(categories[0], max_pages_per_category, max_depth)
    pairs = []

    for page_title, content in wiki_documents.items():
        try:
            if isinstance(content, dict):
                # Combine all section content into a single text
                full_text = json.dumps(content)

                # Generate positive pair
                sentence_pair = wiki_generate_sentence_pair(full_text, page_title)
                if sentence_pair["sentence1"] and sentence_pair["sentence2"]:
                    pairs.append({
                        "sentence1": sentence_pair["sentence1"],
                        "sentence2": sentence_pair["sentence2"],
                        "label": 1  # Positive pair
                    })
        except Exception as e:
            print(e)
    neg_pairs=[]
    for idx, pair in enumerate(pairs):
        if pair["label"] == 1:  # Only process positive pairs
            # Get a random sentence2 from other pairs
            unrelated_sentence2 = random.choice(
                [p["sentence2"] for i, p in enumerate(pairs) if p["label"] == 1 and i != idx]
            )
            # Append as a negative pair
    all_corpus = [p["sentence2"] for p in pairs]

    # 3. Get one hard-negative index per example
    neg_idx = sample_hard_negatives(all_corpus, top_k=32)

    # 4. Build your negative pairs
    neg_pairs = []
    for i, p in enumerate(pairs):
        neg_pairs.append({
            "sentence1": p["sentence1"],
            "sentence2": all_corpus[neg_idx[i]],
            "label": 0
        })
    pairs.extend(neg_pairs)
    # Save pair classification data
    # save_data_as_json(pairs, output_file)
    result=pd.DataFrame(pairs)
    result=result.dropna()
    if len(result)>16384:
        result=result.sample(16384)
    result.to_csv(output_file)

    return pairs

if __name__ =='__main__':
    categories = ["Machine learning"]#"Artificial intelligence", "Data science", "Computer vision"]
    # dataset = wiki_fetch_data_from_categories(categories, max_pages_per_category=200, max_depth=2)
    # dataset = wiki_create_retrieval_dataset(categories, max_pages_per_category=10, max_depth=1)
    dataset = wiki_create_pair_classification_data(categories, max_pages_per_category=10, max_depth=1)