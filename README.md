# Condorcet Vote Streamlit App

This project is a Streamlit web application for ranking candidate names using a Condorcet voting system. Users can rate each candidate name from 0 (bad) to 100 (great), and the results are saved to a Google Sheet for further analysis.

## Features
- Paginated voting interface with sliders for each candidate
- Google Sheets integration for storing votes
- Session management to prevent duplicate voting
- Customizable UI text and messages

## To Do
- [ ] Add script to analyze results
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

## Project Structure
- `streamlit_app.py` — Main Streamlit application
- `pages/success.py` — Success page after voting
- `candidates.txt` — List of candidate names (one per line)
- `.streamlit/secrets.toml` — Google Sheets credentials (not tracked by git)

## Customization
- All user-facing text messages are defined as constants at the top of `streamlit_app.py` for easy modification.
- To change the list of candidates, edit `candidates.txt`.

## License
MIT

## Author
Stéphane Debove

