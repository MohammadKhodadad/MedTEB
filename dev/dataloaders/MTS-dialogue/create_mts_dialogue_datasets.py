import os
import tqdm
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_question(dialogue):
    """Generate a question from a given dialogue using the OpenAI Chat API."""
    try:
        # Construct the messages for the chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Given the following conversation, generate a general medical question that requires \
                      the content of this conversation to be retrieved in order to find the answer in \
                          a Retrieval-Augmented Generation (RAG) system. \
                            I age exists in the conversation, and you wanted to include it the question, \
                                  replace it with a range of age:\n\n{dialogue}\n\nQ:"
            }
        ]

        # Call the OpenAI Chat API
        response = client.chat.completions.create(
            model="gpt-4o",  # Use "gpt-4" or "gpt-3.5-turbo"
            messages=messages,
            max_tokens=500,  # Adjust the max tokens as needed
            temperature=0.7
        )

        # Extract the generated question
        question = response.choices[0].message.content.strip()
        print(question)
        return question
    except Exception as e:
        print(f"Error generating question: {e}")
        return ""


def process_dataset(file_path, output_file="../data/mts_dialogue_question_answer_pairs.csv"):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Initialize lists to store questions and answers
    questions = []
    # Iterate over each dialogue in the dataset
    for dialogue in tqdm.tqdm(df['dialogue']):
        question = generate_question(dialogue)
        questions.append(question)

    # Create a new DataFrame with the generated questions and answers
    qa_df = pd.DataFrame({
        "question": questions,
        "dialogue": df['dialogue']
    })

    # Save the DataFrame to a CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    qa_df=qa_df.dropna()
    if len(qa_df)>4096:
        qa_df=qa_df.sample(4096)
    qa_df.to_csv(output_file, index=False)
    
    print(f"Question and answer pairs saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    file_path = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv" 
    process_dataset(file_path)
