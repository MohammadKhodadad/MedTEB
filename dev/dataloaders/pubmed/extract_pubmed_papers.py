import json
import re
import os
from Bio import Entrez
from Bio.Entrez import efetch, read
import random
from openai import OpenAI
from dotenv import load_dotenv
import tqdm
import pandas as pd

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



# Register your email
Entrez.email = "your@email.com"

def clean_html(raw_html):
    clean_text = re.sub('<.*?>', '', raw_html)
    return clean_text

def pubmed_clean_text(text):

    # Remove non-alphanumeric characters (keep letters, numbers, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]:', '', text)
    
    # Replace multiple spaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces and convert to lowercase
    text = text.strip().lower()
    
    return text

def pubmed_get_article_details(pmid):
    handle = efetch(db='pubmed', id=str(pmid), retmode='xml')
    xml_data = read(handle)
    article_data = xml_data['PubmedArticle'][0]

    title = article_data['MedlineCitation']['Article'].get('ArticleTitle', '')
    abstract_data = article_data['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
    abstract = clean_html(abstract_data[0]) if abstract_data else ''

    mesh_terms = []
    if 'MeshHeadingList' in article_data['MedlineCitation']:
        mesh_terms = [mesh['DescriptorName'].title() for mesh in article_data['MedlineCitation']['MeshHeadingList']]

    keywords = []
    if 'KeywordList' in article_data['MedlineCitation']:
        for keyword_list in article_data['MedlineCitation']['KeywordList']:
            keywords.extend([str(keyword) for keyword in keyword_list])

    return {
        "title": title,
        "abstract": abstract,
        "mesh_terms": mesh_terms,
        "keywords": keywords
    }

def pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=10, output_file="data/pubmed_data_by_category.json"):
    all_articles = {}

    for category_name, query in categories.items():
        print(f"Fetching articles for category: {category_name}")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles_per_category)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        category_articles = []

        for pmid in id_list:
            try:
                article_details = pubmed_get_article_details(pmid)
                category_articles.append(article_details)
            except Exception as e:
                print(f"Error fetching data for PMID {pmid}: {e}")

        all_articles[category_name] = category_articles
    if output_file:
        result={'text':[],'label':[]}
        for category_name in all_articles.keys():
            # print(all_articles[category_name])
            for sub_documents in all_articles[category_name]:
                
                text=pubmed_clean_text('title: '+sub_documents['title']+' abstract: '+sub_documents['abstract'])
                # print(text)
                result['text'].append(text)
                result['label'].append(category_name)
        result=pd.DataFrame(result)
        result=result.dropna()
        if len(result)>8192:
            result=result.sample(8192)
        result.to_csv(output_file)
        # with open(output_file, mode='w', encoding='utf-8') as file:
        #     json.dump(all_articles, file, indent=4)

    print(f"Data saved to {output_file}")
    return all_articles

def pubmed_generate_anonymized_question_answer(content):
    """
    Use OpenAI GPT-4 to generate anonymized question-answer pairs for a Wikipedia page.
    """
    prompt = (
        f"Using the following content from a pubmed paper, generate:\n"
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

import json
import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ✅ Move this to top-level so it can be used in multiprocessing
def process_pubmed_doc(doc):
    if not isinstance(doc, dict):
        # print('doc not dict')
        return None
    full_text = json.dumps(doc)
    qa_pair = pubmed_generate_anonymized_question_answer(full_text)

    q = qa_pair.get("question")
    a = doc.get("abstract")  # NOTE: assumes doc has "answer" key
    if q and a:
        return {
            "query":        q,
            "corpus":       a,
            "source_title": doc.get("title", "")
        }
    # print(f'Question: {q}, Answer: {a}')
    return None


def pubmed_create_retrieval_dataset(
    categories,
    max_doc_per_category=10,
    output_file="../data/wikipedia_retrieval_dataset.csv"
):
    """
    Fetch PubMed data and generate a retrieval dataset using OpenAI GPT-4,
    parallelized across CPU cores.
    """
    # 1) Fetch documents
    docs = pubmed_fetch_and_save_articles_by_category(
        categories,
        max_articles_per_category=max_doc_per_category,
        output_file=None
    )
    pubmed_docs = []
    for key in docs:
        pubmed_docs.extend(docs[key])
    # 2) Process documents in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_pubmed_doc, pubmed_docs, chunksize=16),
            total=len(pubmed_docs),
            desc="Generating QA pairs"
        ))

    # 3) Filter out invalid results
    retrieval_data = [r for r in results if r is not None]
    
    # 4) Save & return
    df = pd.DataFrame(retrieval_data).dropna()
    if output_file:
        if len(df) > 16384:
            df = df.sample(16384)
        df.to_csv(output_file, index=False)

    return df










def pubmed_generate_sentence_pair(content):
    """
    Use OpenAI GPT-4 to generate two sentences from a document.
    """
    prompt = (
        f"Using the following content from a pubmed article,"
        "generate two related sentences based on the content. Ensure the sentences convey meaningful information "
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



import json
import random
import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count

def _make_positive_pair(doc):
    """
    Worker function: given one `doc`, returns a {"sentence1", "sentence2", "label":1} dict
    or None if no valid pair could be generated.
    """
    if not isinstance(doc, dict):
        return None

    full_text = json.dumps(doc)
    sentence_pair = pubmed_generate_sentence_pair(full_text)
    if sentence_pair.get("sentence1") and sentence_pair.get("sentence2"):
        return {
            "sentence1": sentence_pair["sentence1"],
            "sentence2": sentence_pair["sentence2"],
            "label": 1
        }
    return None

def pubmed_create_pair_classification_data(categories,
                                           max_doc_per_category=10,
                                           output_file="../data/wikipedia_pair_classification.json"):
    """
    Create pair classification data using sentences from Wikipedia pages,
    but parallelize the positive-pair generation across multiple processes.
    """
    # 1) fetch your docs as before
    docs = pubmed_fetch_and_save_articles_by_category(
        categories,
        max_articles_per_category=max_doc_per_category,
        output_file=None
    )
    pubmed_docs = []
    for key in docs:
        pubmed_docs.extend(docs[key])

    # 2) spawn a pool and map docs → positive pairs
    with Pool(processes=cpu_count()) as pool:
        # imap gives you an iterator, so you can wrap it in tqdm
        results = list(tqdm.tqdm(
            pool.imap(_make_positive_pair, pubmed_docs, chunksize=16),
            total=len(pubmed_docs),
            desc="Generating positive pairs"
        ))

    # 3) filter out the Nones
    positive_pairs = [r for r in results if r is not None]

    # 4) build your hard negatives exactly as before
    all_corpus = [p["sentence2"] for p in positive_pairs]
    neg_idx = sample_hard_negatives(all_corpus, top_k=64)

    negative_pairs = []
    for i, p in enumerate(positive_pairs):
        negative_pairs.append({
            "sentence1": p["sentence1"],
            "sentence2": all_corpus[neg_idx[i]],
            "label": 0
        })

    pairs = positive_pairs + negative_pairs

    # 5) save out
    if output_file:
        df = pd.DataFrame(pairs).dropna()
        if len(df) > 16384:
            df = df.sample(16384)
        df.to_csv(output_file, index=False)

    return pairs

if __name__ == "__main__":
    # Define categories with corresponding queries
    categories = {
        "Treatment": "treatment OR therapy OR intervention",
        "Risk Assessment": "risk assessment OR risk management OR evaluation",
        "Clinical Care": "hospital OR intensive care OR emergency room OR clinical care OR healthcare"
    }
    # print(pubmed_generate_anonymized_question_answer("Polycystic ovary syndrome (PCOS) poses a significant threat to women's fertility and quality of life. Studies have found a close association between the environmental contaminant tributyltin (TBT) and the occurrence of PCOS. The main objective of this study was to investigate the specific mechanisms by which TBT adversely affects the growth of ovarian granulosa cells. Cell viability, cycle, proliferation, and apoptosis were measured by 3-(4, 5-dimethyl-2-thiazolyl)-2, 5-diphenyl-2-H-tetrazolium bromide (MTT), 5-ethynyl-2'-deoxyuridine (EdU), and flow cytometry. Simultaneously, lactate dehydrogenase (LDH) leakage and Caspase-3 activity were measured by the corresponding kits. Besides, western blot was used to analyze the protein levels of cyclin-dependent kinase inhibitor 1\u202fC (CDKN1C) and the transcription factor Yin Yang 1 (YY1). TBT severely impaired the viability, cell cycle, and proliferation capacity of granulosa cells, and induced their apoptosis. Silencing CDKN1C and YY1 alleviated the damage caused by TBT to the cells, but these repair effects were weakened by CDKN1C overexpressed. By inhibiting the phosphatidylinositol 3-kinase/protein kinase B (PI3K/AKT) signaling pathway, TBT upregulated the YY1-mediated CDNK1C, and further exacerbated the damage to granulosa cells. This study revealed the mechanism that TBT induced the loss of ovarian granulosa cells in PCOS patients by upregulating YY1-mediated CDKN1C expression, which provided new ideas and targets for the pathogenesis and treatment of PCOS."))
    # # pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file='../data/pubmed_data_by_category.json')
    # pubmed_create_retrieval_dataset(categories, max_doc_per_category=1, output_file="../data/pubmed_retrieval_data.csv")
    # pubmed_create_pair_classification_data(categories, max_doc_per_category=1, output_file="../data/pubmed_pair_classification.json")