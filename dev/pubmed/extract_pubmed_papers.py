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

def fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file="data/pubmed_data_by_category.json"):
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

    fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file='../data/pubmed_data_by_category.json')
