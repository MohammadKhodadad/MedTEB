from dataloaders.wikipedia.extract_wikipedia_documents import wiki_fetch_data_from_categories


clustering_tasks = {
    "Disease Clustering by System": [
        "Cardiovascular diseases",
        "Digestive diseases",
        "Eye diseases",
        "Genitourinary system diseases",
        "Respiratory diseases",
        "Neurological disorders",
        "Musculoskeletal disorders"
    ],
    "Disorder Type Clustering": [
        "Genetic disorders by system",
        "Immune system disorders",
        "Orthopedic problems",
        "Musculoskeletal disorders",
        "Cutaneous conditions"
    ],
    "Symptom-Based Clustering": [
        "Foot diseases",
        "Rectal diseases",
        "Breast diseases",
        "Hair diseases",
        "Voice disorders"
    ],
    "Pregnancy and Women's Health": [
        "Pathology of pregnancy, childbirth and the puerperium",
        "Vaginal diseases",
        "Breast diseases"
    ],
    "Sensory Organ Disease Clustering": [
        "Eye diseases",
        "Diseases of the ear and mastoid process",
        "Cutaneous conditions"
    ],
    "Inflammatory and Immune-Related": [
        "Immune system disorders",
        "Connective tissue diseases",
        "Respiratory diseases"
    ],
    "Orthopedic and Musculoskeletal": [
        "Orthopedic problems",
        "Musculoskeletal disorders",
        "Connective tissue diseases"
    ]
}

# Fetch and save data for each clustering task
for task_name, categories in clustering_tasks.items():
    output_file = f"../data/clustering/{task_name.replace(' ', '_').lower()}_dataset.json"
    print(f"\nProcessing task: {task_name}")
    wiki_fetch_data_from_categories(categories, max_pages_per_category=500, output_dir=output_file, max_depth=1)