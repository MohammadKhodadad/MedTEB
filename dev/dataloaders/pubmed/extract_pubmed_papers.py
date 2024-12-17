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


def pubmed_create_retrieval_dataset(categories, max_doc_per_category=10, output_file="../data/wikipedia_retrieval_dataset.json"):
    """
    Fetch Wikipedia data and generate a retrieval dataset using OpenAI GPT-4.
    """
    docs = pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=max_doc_per_category, output_file=None)
    pubmed_docs= []
    for key in docs.keys():
        pubmed_docs.extend(docs[key])
    retrieval_data = []

    for doc in  tqdm.tqdm(pubmed_docs):
        if isinstance(doc, dict):
            # Combine all section content into a single text for generation
            full_text = json.dumps(doc)
            qa_pair = pubmed_generate_anonymized_question_answer(full_text)

            if qa_pair["question"] and qa_pair["answer"]:
                retrieval_data.append({
                    "query": qa_pair["question"],
                    "corpus": qa_pair["answer"],
                    "source_title": doc['title']
                })

    # Save retrieval dataset
    if output_file:
        pd.DataFrame(retrieval_data).to_csv(output_file)










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



def pubmed_create_pair_classification_data(categories, max_doc_per_category=10, output_file="../data/wikipedia_pair_classification.json"):
    """
    Create pair classification data using sentences from Wikipedia pages.
    """
    docs = pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=max_doc_per_category, output_file=None)
    pubmed_docs= []
    for key in docs.keys():
        pubmed_docs.extend(docs[key])
    pairs = []

    for doc in tqdm.tqdm(pubmed_docs):
        if isinstance(doc, dict):
            # Combine all section content into a single text
            full_text = json.dumps(doc)

            # Generate positive pair
            sentence_pair = pubmed_generate_sentence_pair(full_text)
            if sentence_pair["sentence1"] and sentence_pair["sentence2"]:
                pairs.append({
                    "sentence1": sentence_pair["sentence1"],
                    "sentence2": sentence_pair["sentence2"],
                    "label": 1  # Positive pair
                })
    neg_pairs=[]
    for idx, pair in enumerate(pairs):
        if pair["label"] == 1:  # Only process positive pairs
            # Get a random sentence2 from other pairs
            unrelated_sentence2 = random.choice(
                [p["sentence2"] for i, p in enumerate(pairs) if p["label"] == 1 and i != idx]
            )
            # Append as a negative pair
            neg_pairs.append({
                "sentence1": pair["sentence1"],
                "sentence2": unrelated_sentence2,
                "label": 0  # Negative pair
            })
    pairs.extend(neg_pairs)
    # Save pair classification data
    if output_file:
        pd.DataFrame(pairs).to_csv(output_file)
        # with open(output_file, mode='w', encoding='utf-8') as file:
        #     json.dump(pairs, file, indent=4)
    return pairs

if __name__ == "__main__":
    # Define categories with corresponding queries
    categories = {
        "Treatment": "treatment OR therapy OR intervention",
        "Risk Assessment": "risk assessment OR risk management OR evaluation",
        "Clinical Care": "hospital OR intensive care OR emergency room OR clinical care OR healthcare"
    }

    # pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file='../data/pubmed_data_by_category.json')
    pubmed_create_retrieval_dataset(categories, max_doc_per_category=10, output_file="../data/pubmed_retrieval_data.json")
    pubmed_create_pair_classification_data(categories, max_doc_per_category=10, output_file="../data/pubmed_pair_classification.json")