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
   By default, this will print a randomly selected Condorcet winner based on the computed probabilities. If you do not specify `--n-samples`, it will default to 1 (one winner). If you do not specify `--lottery-averaging`, it will default to the number of candidates in the election, ensuring stable and consistent results.

   If your CSV contains both text and image columns (image columns are detected by their filename extensions: .jpg, .png, .webp), a warning will be printed. You can analyze only image columns or only text columns using the following options:

   - To analyze only image columns:
     ```bash
     poetry run python analyze_results.py path/to/results.csv --images
     ```
   - To analyze only text columns:
     ```bash
     poetry run python analyze_results.py path/to/results.csv --texts
     ```
   - You cannot use both `--images` and `--texts` at the same time.

   To print the full Condorcet lottery (probability distribution for all candidates), use:
   ```bash
   poetry run python analyze_results.py path/to/results.csv --full-lottery
   ```
   When using `--full-lottery`, the script will also sample and print the winner(s) according to the computed probabilities.

   To sample multiple winners, as if you were running n independent random elections, use the `--n-samples` option (defaults to 1 if not specified):
   ```bash
   poetry run python analyze_results.py path/to/results.csv --n-samples 3
   ```

   To control the number of times the Condorcet lottery is averaged for stability, use the `--lottery-averaging` option (defaults to the number of candidates if not specified):
   ```bash
   poetry run python analyze_results.py path/to/results.csv --lottery-averaging 10
   ```

   To calculate the normal average note for each candidate and print the top n candidates, use the `--regular-averages` option (defaults to 10 if not specified):
   ```bash
   poetry run python analyze_results.py path/to/results.csv --regular-averages 10
   ```

   You can combine these options as needed. For example, to sample 5 winners and average the lottery 20 times:
   ```bash
   poetry run python analyze_results.py path/to/results.csv --n-samples 5 --lottery-averaging 20
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

