import requests
import json
import os

def fetch_biorxiv_papers(start_date, end_date, max_results=100, output_file="../data/biorxiv_extracted_papers.json"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Base URL for the bioRxiv API
    base_url = "https://api.biorxiv.org/details/biorxiv/{}/{}/{}"

    # Initialize variables
    all_papers_by_category = {}
    batch_size = 100  # Number of records to fetch per request
    for i in range(max_results//batch_size):
        cursor= i * batch_size
        api_url = base_url.format(start_date, end_date, cursor)

        # Fetch data from the API
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")

        # Parse the JSON response
        data = response.json()
        papers = data.get('collection', [])
        if not papers:
            print("No more papers found.")

        # Process each paper and organize by category
        for paper in papers:
            category = paper.get("category", "Uncategorized")
            paper_details = {
                "doi": paper.get("doi", "No DOI"),
                "title": paper.get("title", "No Title"),
                "authors": paper.get("authors", "No Authors"),
                "author_corresponding": paper.get("author_corresponding", "No Corresponding Author"),
                "author_corresponding_institution": paper.get("author_corresponding_institution", "No Institution"),
                "date": paper.get("date", "No Date"),
                "version": paper.get("version", "No Version"),
                "category": category,
                "jats_xml_path": paper.get("jatsxml", "No JATS XML Path"),
                "abstract": paper.get("abstract", "No Abstract"),
                "published": paper.get("published", "No Published Date")
            }

            # Add the paper to the appropriate category list
            if category not in all_papers_by_category:
                all_papers_by_category[category] = []
            all_papers_by_category[category].append(paper_details)

        # Update cursor for the next batch

    # Save the papers grouped by category to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers_by_category, f, ensure_ascii=False, indent=4)

    print(f"{sum([len(all_papers_by_category[cat_]) for cat_ in all_papers_by_category.keys()])} papers saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    fetch_biorxiv_papers(start_date="2024-01-01", end_date="2024-01-31", max_results=500)
