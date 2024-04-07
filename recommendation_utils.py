import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Vectorize + Cosine Similarity Matrix
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data.str.lower())
    # Get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    print(cosine_sim_mat)
    return cosine_sim_mat


def get_recommendation(title, cosine_sim_mat, combined_df, df, num_of_rec=10):
    title_lower = title.lower()

    # indices of the course
    course_indices = pd.Series(combined_df.index, index=combined_df['Title'].str.lower())
    course_indices.drop_duplicates(keep='first', inplace=True)

    # Index of course
    idx = course_indices[title_lower]

    # Look into the cosine matr for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = combined_df.iloc[selected_course_indices].copy()
    # result_df['similarity_score'] = selected_course_scores
    result_df.loc[:, 'similarity_score'] = selected_course_scores

    if title_lower in df['Title'].str.lower().values:
        original_index = df[df['Title'].str.lower() == title_lower].index[0]
        original_course_data = df.loc[[original_index]].copy()
        original_course_data['similarity_score'] = 1.0
        result_df = pd.concat([original_course_data, result_df])

    result_df['Payment_Status'] = result_df['is_paid'].apply(get_payment_status)
    result_df['course_Rating'] = result_df['Rating'].apply(get_rating)
    result_df['course_Duration'] = result_df['Duration'].apply(get_duration)

    final_recommended_courses = result_df[['Title', 'similarity_score', 'URL', 'Site', 'course_Duration', 'course_Rating', 'Payment_Status']]
    return final_recommended_courses.head(num_of_rec)


def get_payment_status(is_paid):
    if pd.isnull(is_paid):
        return "Nothing"
    elif is_paid:
        return "Paid"
    else:
        return "Free"

def get_rating(Rating):
    if pd.isnull(Rating):
        return "Nothing"
    elif Rating:
        return Rating


def get_duration(Duration):
    if pd.isnull(Duration):
        return "Nothing"
    elif Duration:
        return Duration


def concat_input_original(title, data):
    # Convert the title to lowercase for case-insensitive comparison
    title_lower = title.lower()

    # Check if the title already exists in the DataFrame
    if title_lower in data['Title'].str.lower().values:
        print(f"The title '{title}' already exists in the dataset. Course not added.")
        return data

    # Create a DataFrame with the input title
    input_df = pd.DataFrame({'Title': [title]})

    # Concatenate the input DataFrame with the original DataFrame
    combined_df = pd.concat([data, input_df], ignore_index=True)
    print(f"The title '{title}' doesn t exists in the dataset. added Successfully.")
    return combined_df
