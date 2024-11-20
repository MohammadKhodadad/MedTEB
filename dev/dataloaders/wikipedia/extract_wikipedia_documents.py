import json
import wikipediaapi

# Initialize the Wikipedia API for English
wiki_wiki = wikipediaapi.Wikipedia('Anonymous Name')

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

def wiki_fetch_data_from_categories(categories, max_pages_per_category=50,max_depth=1,output_dir="../data/wikipedia_dataset.json"):
    all_documents = {}

    for category_name in categories:
        print(f"Fetching pages from category: {category_name}")
        category_documents = wiki_fetch_pages_in_category_recursive(category_name, max_pages=max_pages_per_category, max_depth=max_depth)
        all_documents[category_name]=category_documents
    save_data_as_json(all_documents, output_dir)
    
    return all_documents

if __name__ =='__main__':
    categories = ["Machine learning", ]#"Artificial intelligence", "Data science", "Computer vision"]
    dataset = wiki_fetch_data_from_categories(categories, max_pages_per_category=200, max_depth=2)