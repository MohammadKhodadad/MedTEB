import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


def find_unique_similar_patient_pair(drg_code_df, min_sim_src=0.95):
    """Finds unique patient pairs with similar DRG profiles based on cosine similarity.

        Processes APR-DRG coded patient data to identify pairs of patients with highly similar
        clinical profiles based on their DRG codes, severity levels, and mortality risk scores.
        Pairs are returned in a consistent order (lower subject ID first) with no duplicates.

        Args:
            drg_code_df (pd.DataFrame): Input DataFrame containing patient DRG records with columns:
                - subject_id: Patient identifier
                - drg_type: DRG system type (e.g., 'APR')
                - drg_code: DRG code
                - drg_severity: Numeric severity level
                - drg_mortality: Numeric mortality risk level
            min_sim_src (float, optional): Minimum cosine similarity threshold for including pairs.
                Defaults to 0.95 (very high similarity).

        Returns:
            pd.DataFrame: DataFrame containing similar patient pairs with columns:
                - subject_id_1: First patient ID (always the smaller ID)
                - subject_id_2: Second patient ID (always the larger ID)
                - similarity_score: Cosine similarity score (0-1)
                Sorted by similarity_score in descending order.
        """

    df_apr_drg_codes = drg_code_df[drg_code_df['drg_type'] == 'APR']

    # Create a DRG profile for each patient
    patient_profiles = defaultdict(dict)
    for _, row in df_apr_drg_codes.iterrows():
        key = (row['drg_code'], row['drg_severity'], row['drg_mortality'])
        patient_profiles[row['subject_id']][key] = patient_profiles[row['subject_id']].get(key, 0) + 1

    # Convert to DataFrame
    profile_df = pd.DataFrame.from_dict(patient_profiles, orient='index')
    profile_df = profile_df.fillna(0)

    # Calculate cosine similarity between patients
    similarity_matrix = cosine_similarity(profile_df)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=profile_df.index,
        columns=profile_df.index
    )

    # Get top pairs
    top_pairs = []
    for i in range(len(similarity_df)):
        for j in range(i + 1, len(similarity_df)):
            score = similarity_df.iloc[i, j]
            if score > min_sim_src:
                top_pairs.append({
                    'subject_id_1': min(similarity_df.index[i], similarity_df.index[j]),
                    'subject_id_2': max(similarity_df.index[i], similarity_df.index[j]),
                    'similarity_score': score
                })

    top_pairs_df = pd.DataFrame(top_pairs).sort_values('similarity_score', ascending=False)
    return top_pairs_df


if __name__ == '__main__':
    route_to_mimic_processed_data = '~/SO_Data/MIMIC-Processed-Data/discharge'
    data_path = Path(route_to_mimic_processed_data).expanduser().absolute()
    path_to_processed_discharges = data_path / '100_discharge_processed.csv'
    path_to_drg_codes = data_path / 'drgcodes.csv'

    df_processed_discharges = pd.read_csv(path_to_processed_discharges)
    df_drg_codes = pd.read_csv(path_to_drg_codes, nrows=10000)

    top_pairs_on_drg_codes = find_unique_similar_patient_pair(df_drg_codes, min_sim_src=0.95)
    print(top_pairs_on_drg_codes.head)
