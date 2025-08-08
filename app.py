from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load books dataset
books = pd.read_csv('data/bookdata.csv')

# Normalize genre column
books['genre'] = books['genre'].str.strip().str.title()

# Combine genre and description to form text representation
books['combined_text'] = books['genre'] + ' ' + books['description']

# Load sentence transformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for all books (based on combined text)
book_embeddings = model.encode(books['combined_text'].tolist(), convert_to_tensor=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_text = request.form.get('user_input')
    if not user_text or user_text.strip() == '':
        return "Please enter a genre or description."

    # Embed user input text
    user_embedding = model.encode(user_text, convert_to_tensor=True)

    # Compute cosine similarity with all book embeddings
    cosine_scores = util.pytorch_cos_sim(user_embedding, book_embeddings)[0]

    # Get top 5 recommendations
    top_results = cosine_scores.topk(k=5)

    recommended_books = []
    for score, idx in zip(top_results.values, top_results.indices):
        book = books.iloc[idx.item()]
        recommended_books.append({
            'title': book['title'],
            'author': book['author'],
            'genre': book['genre'],
            'description': book['description'],
            'image': book['image'],
            'score': score.item()
        })

    return render_template('result.html', query=user_text, recommendations=recommended_books)


if __name__ == '__main__':
    app.run(debug=True)
