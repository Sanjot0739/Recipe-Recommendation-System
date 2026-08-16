Recipe Recommendation System

A Flask-based recipe recommendation system using TF-IDF and cosine similarity.

Dataset

Place the dataset at:

dataset/recipes.csv

Required columns:

Title

Ingredients

Instructions

Image_Name

Cleaned_Ingredients

Images

Place recipe images inside:

static/images/

The application automatically supports .jpg, .jpeg, .png, and .webp.

Run

pip install -r requirements.txt
python app.py

Open:

http://127.0.0.1:5000