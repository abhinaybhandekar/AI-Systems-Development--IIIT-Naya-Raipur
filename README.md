# Movie Semantic Search Assignment

## Setup
1. Create virtual env: python -m venv venv
2. Activate and install: pip install -r requirements.txt
3. Place `movies.csv` at repo root (columns: movie_id,title,plot)

## Run
python movie_search.py "spy thriller in Paris" --top_n 5

## Tests
python -m unittest tests/test_movie_search.py -v
