import json
import re
from Bio import Entrez
from Bio.Entrez import efetch, read

# Register your email
Entrez.email = "your@email.com"

def clean_html(raw_html):
    clean_text = re.sub('<.*?>', '', raw_html)
    return clean_text

def get_article_details(pmid):
    handle = efetch(db='pmc', id=str(pmid), retmode='xml')
    xml_data = read(handle)
    
    article_data = xml_data[0]['front']
    
    title = article_data['article-meta']['title-group']['article-title']
    abstract_data = article_data['article-meta']['abstract'][0]['p']
    abstract = clean_html(abstract_data[0]) if abstract_data else ''
    try:
        keywords = article_data['article-meta']['kwd-group'][0]['kwd']
    except:
        print('no keywords')
        keywords= []         
    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords
    }

def fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file="data/pubmed_data_by_category.json"):
    all_articles = {}

    for category_name, query in categories.items():
        print(f"Fetching articles for category: {category_name}")
        handle = Entrez.esearch(db="pmc", term=query, retmax=max_articles_per_category)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        category_articles = []

        for pmid in id_list:
            try:
                article_details = get_article_details(pmid)
                category_articles.append(article_details)
            except Exception as e:
                print(f"Error fetching data for PMID {pmid}: {e}")

        all_articles[category_name] = category_articles

    with open(output_file, mode='w', encoding='utf-8') as file:
        json.dump(all_articles, file, indent=4)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Define categories with corresponding queries
    categories = {
        "Treatment": "treatment OR therapy OR intervention",
        "Risk Assessment": "risk assessment OR risk management OR evaluation",
        "Clinical Care": "hospital OR intensive care OR emergency room OR clinical care OR healthcare"
    }

    fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file='../data/pmc_data_by_category.json')
