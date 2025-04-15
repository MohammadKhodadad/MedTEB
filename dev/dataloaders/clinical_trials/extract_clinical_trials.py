import requests
import tqdm
import pandas as pd
import time
import os
import random
from openai import OpenAI
from dotenv import load_dotenv
import json
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

def extract_simplified_dict(data):
    def get_value(d, keys, default=""):
        """Safely get a nested value from a dictionary."""
        for key in keys:
            d = d.get(key, {})
        return d if isinstance(d, (str, int, float, list)) else default

    simplified_dict = {
        "nctId": get_value(data, ["identificationModule", "nctId"]),
        "orgStudyId": get_value(data, ["identificationModule", "orgStudyIdInfo", "id"]),
        "organization": get_value(data, ["identificationModule", "organization", "fullName"]),
        "briefTitle": get_value(data, ["identificationModule", "briefTitle"]),
        "briefSummary": get_value(data, ["descriptionModule", "briefSummary"]),
        "detailedDescription": get_value(data, ["descriptionModule", "detailedDescription"]),
        "officialTitle": get_value(data, ["identificationModule", "officialTitle"]),
        "overallStatus": get_value(data, ["statusModule", "overallStatus"]),
        "startDate": get_value(data, ["statusModule", "startDateStruct", "date"]),
        "primaryCompletionDate": get_value(data, ["statusModule", "primaryCompletionDateStruct", "date"]),
        "completionDate": get_value(data, ["statusModule", "completionDateStruct", "date"]),
        "leadSponsor": get_value(data, ["sponsorCollaboratorsModule", "leadSponsor", "name"]),
        "collaborators": [collab.get("name", "") for collab in get_value(data, ["sponsorCollaboratorsModule", "collaborators"], [])],
        "conditions": get_value(data, ["conditionsModule", "conditions"], []),
        "keywords": get_value(data, ["conditionsModule", "keywords"], []),
        "studyType": get_value(data, ["designModule", "studyType"]),
        "phases": get_value(data, ["designModule", "phases"], []),
        "primaryPurpose": get_value(data, ["designModule", "designInfo", "primaryPurpose"]),
        "masking": get_value(data, ["designModule", "designInfo", "maskingInfo", "masking"]),
        "enrollmentCount": get_value(data, ["designModule", "enrollmentInfo", "count"]),
        "armGroups": [
            {
                "label": arm.get("label", ""),
                "type": arm.get("type", ""),
                "description": arm.get("description", "")
            }
            for arm in get_value(data, ["armsInterventionsModule", "armGroups"], [])
        ],
        "interventions": [
            {
                "type": intervention.get("type", ""),
                "name": intervention.get("name", ""),
                "description": intervention.get("description", "")
            }
            for intervention in get_value(data, ["armsInterventionsModule", "interventions"], [])
        ],
        "primaryOutcomes": [
            {
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "timeFrame": outcome.get("timeFrame", "")
            }
            for outcome in get_value(data, ["outcomesModule", "primaryOutcomes"], [])
        ],
        "secondaryOutcomes": [
            {
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "timeFrame": outcome.get("timeFrame", "")
            }
            for outcome in get_value(data, ["outcomesModule", "secondaryOutcomes"], [])
        ],
        "eligibilityCriteria": get_value(data, ["eligibilityModule", "eligibilityCriteria"]),
        "minimumAge": get_value(data, ["eligibilityModule", "minimumAge"]),
        "maximumAge": get_value(data, ["eligibilityModule", "maximumAge"]),
        "sex": get_value(data, ["eligibilityModule", "sex"]),

    }

    return simplified_dict




def fetch_all_studies_with_pagination(base_url, page_size=100, format_type="json", max_pages=2):
    """
    Fetch all study data from ClinicalTrials.gov using pagination.

    Parameters:
    - base_url: The base API URL.
    - page_size: Number of records per page.
    - format_type: Data format ('json' or 'csv').
    - max_pages: Maximum number of pages to fetch.

    Returns:
    - A Pandas DataFrame containing all study data (if format is 'json').
    """
    all_data = []
    next_page_token = None

    for _ in tqdm.tqdm(range(max_pages), desc="Fetching Pages"):
        # Construct the request URL
        params = {
            "pageSize": page_size,
            "format": format_type,
            "pageToken": next_page_token  # None for the first request
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        if format_type == "json":
            data = response.json()

            # Collect studies from the current page
            if "studies" in data and data["studies"]:
                all_data.extend([ extract_simplified_dict(study.get('protocolSection',{})) for study in data["studies"]])   
            else:
                print("No more studies found.")
                break

            # Check for the nextPageToken
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                print("No more pages to fetch.")
                break
        else:
            print("CSV format not supported for multi-page requests.")
            break

        # Optional: Add a delay to avoid overloading the server
        time.sleep(0.5)

    if format_type == "json":
        # Convert the collected data into a DataFrame
        return pd.DataFrame(all_data)
    else:
        return None



def clinical_trials_anonymize_corpus(title,corpus):
    """
    Use OpenAI GPT-4 to generate anonymized question-answer pairs for a Wikipedia page.
    """
    prompt = (
        f"Using the following content from a clinical trial, rephrase the corpus so that does not have anything directly from the title"
        f"Title:\n{title}\n"
        f"Corpus:\n{corpus}\n\n"
        "Output as JSON with key 'corpus' without any code formatting, backticks, or markdown."
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
        result = json.loads(result)
        return result
    except Exception as e:
        print(f"Error generating question-answer pair: {e}")
        return {"corpus": None}


import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count

def clinical_trials_create_retrieval_dataset(
    base_api_url="https://www.clinicaltrials.gov/api/v2/studies",
    col1='officialTitle',
    col2='detailedDescription',
    page_size=5,
    max_pages=1000,
    output_file="../data/clinical_trials_retrieval_dataset.json"
):
    """
    Fetch clinical trials data and generate a retrieval dataset using OpenAI GPT-4,
    but parallelized across CPU cores.
    """
    # 1) fetch everything
    clinical_trials_df = fetch_all_studies_with_pagination(
        base_api_url,
        page_size=page_size,
        max_pages=max_pages
    )
    # 2) turn into lightweight records
    records = clinical_trials_df.to_dict(orient='records')

    # 3) define worker inside so it captures col1/col2
    def _process_record(record):
        title = record.get(col1, "")
        corpus = record.get(col2, "")

        # special handling for primaryOutcomes
        if col2 == 'primaryOutcomes':
            if isinstance(corpus, list):
                # flatten list of dicts into text
                corpus = '\n'.join(
                    ' '.join(f"{k}: {item[k]}"
                             for k in item if k != 'timeFrame')
                    for item in corpus
                )
            else:
                return None

        # only keep non-empty title & corpus
        if not title or not corpus:
            return None

        # optionally anonymize here:
        # new_corpus = clinical_trials_anonymize_corpus(title, corpus).get('corpus','')
        new_corpus = corpus

        if not new_corpus:
            return None

        return {
            "query":        title,
            "corpus":       new_corpus,
            "source_title": title
        }

    # 4) parallel map with progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm.tqdm(
            pool.imap(_process_record, records, chunksize=32),
            total=len(records),
            desc="Processing clinical trials"
        ))

    # 5) filter out the Nones
    retrieval_data = [r for r in results if r is not None]

    # 6) save & return
    df = pd.DataFrame(retrieval_data).dropna()
    if output_file:
        if len(df) > 16384:
            df = df.sample(16384)
        df.to_csv(output_file, index=False)
    return df

import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count

def clinical_trials_pair_classification_dataset(
    base_api_url="https://www.clinicaltrials.gov/api/v2/studies",
    col1='officialTitle',
    col2='detailedDescription',
    page_size=5,
    max_pages=1000,
    output_file="../data/clinical_trials_pair_classification_dataset.json"
):
    """
    Fetch clinical trials data and generate a pair-classification dataset,
    parallelizing the positive-pair extraction.
    """
    # 1) Fetch all trials into a DataFrame
    df = fetch_all_studies_with_pagination(
        base_api_url,
        page_size=page_size,
        max_pages=max_pages
    )
    records = df.to_dict(orient='records')

    # 2) Worker: turn one record into a positive pair dict or None
    def _make_positive_pair(record):
        title = record.get(col1, "")
        corpus = record.get(col2, "")

        if col2 == 'primaryOutcomes':
            if isinstance(corpus, list):
                corpus = '\n'.join(
                    ' '.join(f"{k}: {item[k]}"
                             for k in item if k != 'timeFrame')
                    for item in corpus
                )
            else:
                return None

        if not title or not corpus:
            return None

        # optional anonymization step could go here
        new_corpus = corpus

        if not new_corpus:
            return None

        return {"sentence1": title, "sentence2": new_corpus, "label": 1}

    # 3) Parallel map with progress bar
    with Pool(processes=cpu_count()) as pool:
        pos_results = list(tqdm.tqdm(
            pool.imap(_make_positive_pair, records, chunksize=32),
            total=len(records),
            desc="Generating positive pairs"
        ))

    # 4) Filter out failures
    positive_pairs = [p for p in pos_results if p is not None]

    # 5) Build hard negatives
    all_corpus = [p["sentence2"] for p in positive_pairs]
    neg_idx   = sample_hard_negatives(all_corpus, top_k=64)

    negative_pairs = []
    for i, p in enumerate(positive_pairs):
        negative_pairs.append({
            "sentence1": p["sentence1"],
            "sentence2": all_corpus[neg_idx[i]],
            "label": 0
        })

    # 6) Save & return
    all_pairs = positive_pairs + negative_pairs
    result_df = pd.DataFrame(all_pairs).dropna()
    if output_file:
        if len(result_df) > 16384:
            result_df = result_df.sample(16384)
        result_df.to_csv(output_file, index=False)

    return result_df

# Example usage
if __name__ == "__main__":
    # Fetch all study data
    clinical_trials_df = clinical_trials_create_retrieval_dataset(col2='primaryOutcomes',page_size=2,max_pages=2)
    clinical_trials_df = clinical_trials_pair_classification_dataset(col2='primaryOutcomes',page_size=2,max_pages=2)