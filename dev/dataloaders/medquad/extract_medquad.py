import os
import random
import multiprocessing as mp
from openai import OpenAI
from dotenv import load_dotenv
import tqdm
import pandas as pd
from datasets import load_dataset

def init_openai_client():
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def medquad_generate_anonymized_question_answer(args):
    """
    Use OpenAI GPT-4 to generate anonymized question-answer pairs for a Wikipedia page.
    """
    question, answer = args
    client = init_openai_client()
    
    prompt = (
        f"You will receive a question and an answer in medicine. Your job is to modify the answer:\n"
        "1. Which is the answer rephrased.\n"
        "2. Does not have any keywords from the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Output the text of the answer without any code formatting, backticks, or markdown."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that anonymizes content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating question-answer pair: {e}")
        return None

def medquad_retrieval_extract_and_save(save_path='../data/medquad_retrieval.csv', num_workers=8):
    """
    Loads the MedQuAD dataset from Hugging Face, extracts the 'question' and 'answer' columns,
    anonymizes the answers in parallel, and saves the processed data as a CSV file.
    """
    dataset_name = 'lavita/MedQuAD'
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])  # Assuming we use the train split
    df = df.sample(8500)
    
    questions_answers = list(zip(df['question'].tolist(), df['answer'].tolist()))
    
    print("Starting multi-processing for anonymization...")
    with mp.Pool(processes=min(os.cpu_count(), num_workers)) as pool:
        new_answers = list(tqdm.tqdm(pool.imap(medquad_generate_anonymized_question_answer, questions_answers), total=len(questions_answers)))
    
    df['new_answers'] = new_answers
    df = df[['question', 'new_answers']].rename(columns={'question': 'query', 'new_answers': 'corpus'})
    df = df.dropna()
    
    if len(df) > 8192:
        df = df.sample(8192)
    
    df.to_csv(save_path, index=False)
    print(f"Dataset saved successfully at {save_path}")

# Example usage:
# medquad_retrieval_extract_and_save("./medquad_queries.csv", num_workers=8)
