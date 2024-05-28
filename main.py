from flask import Flask, jsonify, request
from recommendation_utils import vectorize_text_to_cosine_mat, get_recommendation, concat_input_original
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend_courses():
    data = request.get_json()
    search_term = data.get('course')
    num_of_rec = int(data.get('recNum'))

    df = pd.read_csv("data/AllOnlineCourses_Clean.csv")
    combined_df = concat_input_original(search_term, df)
    cosine_sim_mat = vectorize_text_to_cosine_mat(combined_df['Title'])

    results = get_recommendation(search_term, cosine_sim_mat, combined_df, df, num_of_rec)
    results_json = results.to_dict('index')

    return jsonify(results_json)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
