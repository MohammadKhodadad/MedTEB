import json
import wikipediaapi

# Initialize the Wikipedia API for English
wiki_wiki = wikipediaapi.Wikipedia('Anonymous Name')


def fetch_wikipedia_page_sections(title):
    page = wiki_wiki.page(title)
    if not page.exists():
        return None

    # Extract sections recursively
    def extract_sections(sections, level=0):
        content = {}
        for section in sections:
            content[section.title] = section.text
            subsections = extract_sections(section.sections, level + 1)
            if subsections:
                content.update(subsections)
        return content

    return extract_sections(page.sections)


def fetch_pages_in_category(category_name, max_pages=100):
    category = wiki_wiki.page(f"Category:{category_name}")
    pages = category.categorymembers
    documents = {}

    for i, (title, page) in enumerate(pages.items()):
        if page.ns == 0:  # Only consider main namespace pages (articles)
            documents[title] = fetch_wikipedia_page_sections(title)
            if len(documents) >= max_pages:
                break
    return documents

def fetch_data_from_categories(categories, max_pages_per_category=50):
    all_documents = {}

    for category_name in categories:
        print(f"Fetching pages from category: {category_name}")
        category_documents = fetch_pages_in_category(category_name, max_pages=max_pages_per_category)
        all_documents.update(category_documents)

    return all_documents


def save_data_as_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ =='__main__':
    categories = ["Machine learning", "Artificial intelligence", "Data science", "Computer vision"]
    dataset = fetch_data_from_categories(categories, max_pages_per_category=50)
    save_data_as_json(dataset, "../data/wikipedia_dataset.json")
    print("Dataset saved as wikipedia_dataset.json")