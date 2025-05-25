
# Condorcet Vote Streamlit App

This project is a Streamlit web application for ranking candidate names using a Condorcet voting system. Users can rate each candidate name from 0 (bad) to 100 (great), and the results are saved to a Google Sheet for further analysis.

## What is the Condorcet Method?

The Condorcet method is a voting system that identifies the candidate who would win a head-to-head competition against every other candidate. If such a candidate exists, they are called the Condorcet winner. If there is no single winner due to cycles or ties, the method can be extended to compute a fair probability distribution (Condorcet lottery) over all candidates, reflecting the collective preferences of the voters.

## To Do
- [ ] Allow to compare candidates two by two instead of rating them individually


## Requirements
- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management


## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd condorcet_vote
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```



## Usage
1. Configure your Google Sheets credentials in `.streamlit/secrets.toml` (see example `.streamlit/secrets_example.toml` in the repo). For a detailed tutorial on how to obtain the credentials, see https://www.youtube.com/watch?v=HwxrXnYVIlU
2. Start the Streamlit app:
   ```bash
   poetry run streamlit run streamlit_app.py
   ```
3. Open the provided local URL in your browser to vote.


4. To analyze voting results from a CSV file (e.g., exported from Google Sheets), run:
   ```bash
   poetry run python analyze_results.py path/to/results.csv
   ```
   By default, this will print a randomly selected Condorcet winner based on the computed probabilities.

   To print the full Condorcet lottery (probability distribution for all candidates), use:
   ```bash
   poetry run python analyze_results.py path/to/results.csv --full-lottery
   ```

   To sample multiple winners, as if you were running n independent random elections, use the `--n-samples` option:
   ```bash
   poetry run python analyze_results.py path/to/results.csv --n-samples 3
   ```



## Project Structure
- `streamlit_app.py` — Main Streamlit application
- `pages/success.py` — Success page after voting
- `candidates.txt` — List of candidate names (one per line)
- `.streamlit/secrets.toml` — Google Sheets credentials (not tracked by git)
- `analyze_results.py` — Standalone script to process a CSV of votes, compute the Condorcet matrix, and print the winner or lottery. 


## Customization
- All user-facing text messages are defined as constants at the top of `streamlit_app.py` for easy modification.
- To change the list of candidates, edit `candidates.txt`.


## License
MIT


## Author
Streamlit app by Stéphane Debove
Analysis code by Lê Nguyên Hoang 

