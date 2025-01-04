import requests
import tqdm
import pandas as pd
import time
import os
import random
from openai import OpenAI
from dotenv import load_dotenv
import json

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


def clinical_trials_create_retrieval_dataset(col1='officialTitle',col2='detailedDescription',page_size=2,max_pages=2, output_file="../data/clinical_trials_retrieval_dataset.json"):
    """
    Fetch Wikipedia data and generate a retrieval dataset using OpenAI GPT-4.
    """
    retrieval_data=[]
    clinical_trials_df = fetch_all_studies_with_pagination(base_api_url, page_size=page_size,max_pages=max_pages)

    for index, row in tqdm.tqdm(clinical_trials_df.iterrows()):
        title=row[col1]
        corpus=row[col2]
        if len(title) & len(corpus):
            new_corpus = clinical_trials_anonymize_corpus( title, corpus).get('corpus','')

            if new_corpus:
                retrieval_data.append({
                    "query": title,
                    "corpus": new_corpus,
                    "source_title": title
                })

    # Save retrieval dataset
    if output_file:
        result=pd.DataFrame(retrieval_data)
        result=result.dropna()
        if len(result)>8192:
            result=result.sample(8192)
        result.to_csv(output_file)
    return result
# Example usage
if __name__ == "__main__":
    base_api_url = "https://www.clinicaltrials.gov/api/v2/studies"
    # Fetch all study data
    clinical_trials_df = clinical_trials_create_retrieval_dataset()