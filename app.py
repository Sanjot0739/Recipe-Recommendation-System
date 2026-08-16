from pathlib import Path
import re

import pandas as pd
from flask import Flask, abort, render_template, request, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "recipes.csv"
IMAGE_DIR = BASE_DIR / "static" / "images"

# Dataset columns used by this project.
REQUIRED_COLUMNS = [
    "Title",
    "Ingredients",
    "Instructions",
    "Image_Name",
    "Cleaned_Ingredients",
]


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_dataset():
    """Load the recipe CSV and perform safe basic cleaning."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place your CSV at dataset/recipes.csv"
        )

    df = pd.read_csv(DATASET_PATH)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(missing)
        )

    # Remove the automatically generated CSV index column.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convert text columns to strings and safely handle missing values.
    text_columns = [
        "Title",
        "Ingredients",
        "Instructions",
        "Image_Name",
        "Cleaned_Ingredients",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    # Remove rows without a recipe title.
    df = df[df["Title"].str.strip() != ""].reset_index(drop=True)

    # Create one searchable text field.
    # Title gets extra weight by appearing twice.
    df["search_text"] = (
        df["Title"].str.lower().str.strip()
        + " "
        + df["Title"].str.lower().str.strip()
        + " "
        + df["Cleaned_Ingredients"].str.lower().str.strip()
    )

    return df


data = load_dataset()


# ============================================================
# TF-IDF MODEL
# ============================================================

# TF-IDF converts recipe title + ingredient text into numerical
# vectors. We do NOT create a full recipe-to-recipe similarity
# matrix, which keeps memory usage reasonable for 13,501 recipes.
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_features=50000,
)

tfidf_matrix = vectorizer.fit_transform(data["search_text"])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_query(text):
    """Normalize user search text."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def find_image_filename(image_name):
    """
    Find the real image file in static/images.

    The CSV contains image names without extensions, while the
    user's image folder may contain .jpg/.jpeg/.png/.webp files.
    This function supports both cases and case differences.
    """

    image_name = str(image_name or "").strip()

    if not image_name:
        return "placeholder.svg"

    image_dir = IMAGE_DIR

    if not image_dir.exists():
        return "placeholder.svg"

    # If the dataset already contains an extension.
    direct = image_dir / Path(image_name).name
    if direct.is_file():
        return direct.name

    # Try common extensions.
    extensions = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]

    for extension in extensions:
        candidate = image_dir / f"{image_name}{extension}"
        if candidate.is_file():
            return candidate.name

    # Case-insensitive fallback.
    target = image_name.lower()
    for file in image_dir.iterdir():
        if file.is_file():
            stem = file.stem.lower()
            name = file.name.lower()

            if stem == target or name == target:
                return file.name

    return "placeholder.svg"


def add_image_urls(records):
    """Add an image_url field used by the Jinja templates."""

    for record in records:
        filename = find_image_filename(record.get("Image_Name", ""))
        record["image_url"] = url_for(
            "static",
            filename=f"images/{filename}"
        )

    return records


def prepare_recipe(record, recipe_id):
    """Convert a DataFrame row into a template-friendly dictionary."""

    return {
        "id": int(recipe_id),
        "recipe_name": record["Title"],
        "ingredients": record["Ingredients"],
        "ingredients_list": record["Cleaned_Ingredients"],
        "instructions": record["Instructions"],
        "image_name": record["Image_Name"],
        "image_url": url_for(
            "static",
            filename=f"images/{find_image_filename(record['Image_Name'])}"
        ),
    }


def search_recipes(query, limit=30):
    """
    Search recipes using TF-IDF similarity.

    If no query is provided, return the first recipes.
    """

    query = clean_query(query)

    if not query:
        return list(range(min(limit, len(data))))

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).ravel()

    # Sort from highest similarity to lowest.
    ranked_indices = scores.argsort()[::-1]

    # Ignore zero-score results when possible.
    ranked = [int(i) for i in ranked_indices if scores[i] > 0][:limit]

    return ranked


def recommend_similar(recipe_id, limit=6):
    """Return recipes similar to the selected recipe."""

    if recipe_id < 0 or recipe_id >= len(data):
        return []

    recipe_vector = tfidf_matrix[recipe_id]
    scores = cosine_similarity(recipe_vector, tfidf_matrix).ravel()

    # Exclude the selected recipe itself.
    scores[recipe_id] = -1

    ranked_indices = scores.argsort()[::-1]

    return [
        int(i)
        for i in ranked_indices[:limit]
        if scores[i] > 0
    ]


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    """Home page and ingredient/recipe recommendation."""

    query = ""
    recommendations = []

    if request.method == "POST":
        query = request.form.get("ingredients", "").strip()

        if query:
            indices = search_recipes(query, limit=9)

            recommendations = [
                prepare_recipe(data.iloc[index], index)
                for index in indices
            ]

    return render_template(
        "index.html",
        recommendations=recommendations,
        query=query,
        total_recipes=len(data),
    )


@app.route("/recipes")
def recipes():
    """Paginated recipe collection with server-side search."""

    query = request.args.get("q", "").strip()

    try:
        page = max(int(request.args.get("page", 1)), 1)
    except ValueError:
        page = 1

    per_page = 12

    if query:
        recipe_indices = search_recipes(query, limit=len(data))
    else:
        recipe_indices = list(range(len(data)))

    total = len(recipe_indices)
    total_pages = max((total + per_page - 1) // per_page, 1)

    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page

    current_indices = recipe_indices[start:end]

    recipes_data = [
        prepare_recipe(data.iloc[index], index)
        for index in current_indices
    ]

    return render_template(
        "recipes.html",
        recipes=recipes_data,
        query=query,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@app.route("/recipe/<int:recipe_id>")
def recipe_details(recipe_id):
    """Display one recipe and similar recipes."""

    if recipe_id < 0 or recipe_id >= len(data):
        abort(404)

    recipe = prepare_recipe(data.iloc[recipe_id], recipe_id)

    similar_indices = recommend_similar(recipe_id, limit=6)

    similar_recipes = [
        prepare_recipe(data.iloc[index], index)
        for index in similar_indices
    ]

    return render_template(
        "recipe_details.html",
        recipe=recipe,
        similar_recipes=similar_recipes,
    )


@app.route("/search")
def search():
    """Redirect search requests to the recipe collection."""

    query = request.args.get("q", "").strip()
    return render_template(
        "recipes.html",
        recipes=[
            prepare_recipe(data.iloc[index], index)
            for index in search_recipes(query, limit=12)
        ],
        query=query,
        page=1,
        total_pages=1,
        total=len(search_recipes(query, limit=12)),
    )


@app.route("/about")
def about():
    return render_template("about.html", total_recipes=len(data))


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ============================================================
# ERROR HANDLING
# ============================================================

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html"), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Recipe Recommendation System")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Recipes loaded: {len(data):,}")
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(host="127.0.0.1", port=5000, debug=True)