import random
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_gsheets import GSheetsConnection

CANDIDATES_FILE = "candidates.txt"
WORKSHEET_NAME = "Sheet1"
SUCCESS_PAGE = "pages/success.py"
# default slider value before user starts interacting with it:
SLIDER_DEFAULT = 50
# string or number you want to have in your database when a user hasn’t interacted with a slider:
NO_VOTE_VALUE = "Not evaluated"
# WARNING: a user might not interact with a slider but be satisfied with the default value she has seen on screen. So "Not evaluated" can actually mean two things: either the voter thought the default value was ok for a given candidate, or the voter didn’t even see the candidate (submitted the form before reaching the end, etc). Make sure to take this into account when analyzing the data, or change the "SLIDER_DEFAULT" to a more suitable value than 50.
VERTICAL_SPACE = 20  # Space between sliders, in pixels
CANDIDATES_PER_PAGE = 30
# messages for the button "Page suivante !"
NEXT_BUTTON_MESSAGES = [
    "J’adore faire ça",
    "Qu’est-ce que je m’amuse !",
    "C’est vraiment trop bien !",
    "Je vis ma meilleure vie.",
    "Trop content·e de voter !",
    "Merci pour l’opportunité !",
    "Ça fait longtemps que je ne me suis pas autant amusé·e !",
    "Comment ai-je pu vivre si longtemps sans connaître ça",
    "Trop heureux·se de voter !",
    "Que c’est grisant !",
    "C’est le plus beau jour de ma vie !",
    "Je ne savais pas que voter pouvait être aussi amusant !",
    "Je suis sûr·e que c’est bientôt terminé",
    "Je suis trop content·e de participer à ça !",
    "Yen a vraiment qui ont proposé des noms bizarres !",
]

# TEXT CONSTANTS
PAGE_TITLE = "Votez pour le nom de la chaîne !"
PAGE_ICON = "👋"
HEADER_TITLE = (
    "<h1 style='text-align: center;'>Trouvez le nouveau nom Homo Fabulus !</h1>"
)
INSTRUCTIONS = """
    Donnez une note à chacun des noms, de 0 (« ce nom est nul ») à 100 (« ce nom est génial »).\n
    Vous n’êtes pas obligé·e de voter pour tous, mais plus vous votez, mieux c’est.\n
    **Et n’oubliez pas de cliquer sur « Soumettre » à la fin pour que votre vote soit pris en compte !**
    """
PREVIOUS_PAGE_BUTTON = "Page précédente"
SUBMIT_BUTTON = "Soumettre mon vote ! J’en peux plus."
CONFIRM_DIALOG_TITLE = "Etes-vous sûr·e de vouloir soumettre vos votes ?"
CONFIRM_DIALOG_TEXT = "Pas de retour en arrière possible après !"
CANCEL_BUTTON = "T’as raison, j’hésite encore"
CONFIRM_BUTTON = "Je ne change jamais d’avis, enregistre."
SPINNER_TEXT = "Ok j’enregistre..."
ALREADY_SUBMITTED_ERROR = "Il semblerait que vous ayez déjà soumis vos données. Un seul vote par personne ! Si vous pensez que c’est une erreur (c’est possible, car je code comme un pied), vous pouvez me contacter: homofabuluslachaine at gmail.com "
PAGE_COUNT_TEMPLATE = "<h5 style='text-align: center;'>Page {current}/{total}</h3>"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
)


# UTILITY FUNCTIONS


@st.cache_data
def get_css() -> str:
    return """
    <style>
        .stSlider label {
            display: block;
            text-align: center;
        }
    </style>
    """


@st.cache_data
def add_vertical_space():
    st.markdown(
        f"<div style='margin-bottom: {VERTICAL_SPACE}px;'></div>",
        unsafe_allow_html=True,
    )


def scroll_to(element_id: str) -> None:
    """
    Scroll to a specific HTML element on the page.

    Args:
        element_id (str): The ID of the HTML element to scroll to.
    """
    components.html(
        f"""
        <script>
            var element = window.parent.document.getElementById("{element_id}");
            element.scrollIntoView();
        </script>
    """.encode()
    )




# DATA LOADING and INITIALIZATION FUNCTIONS


@st.cache_data
def load_candidates() -> list[str]:
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "shuffled_candidates" not in st.session_state:
        candidates_list = load_candidates()
        st.session_state.shuffled_candidates = random.sample(
            candidates_list, len(candidates_list)
        )
        st.session_state["start_time"] = datetime.now()
        st.session_state["session_id"] = str(uuid.uuid4())
        user_ip = st.context.ip_address
        st.session_state["user_ip"] = (
            user_ip
            if user_ip
            else f"localhost_{st.session_state['session_id']}"
        )
        st.session_state["page_change"] = False
        st.session_state["current_page"] = 0


def initialize_slider(candidate: str) -> None:
    """Initialize slider value for a candidate."""
    slider_key = f"slider_{candidate}"
    candidate_key = f"score_{candidate}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state.get(
            candidate_key, SLIDER_DEFAULT
        )


# CORE LOGIC FUNCTIONS


def calculate_total_pages(total_candidates: int) -> int:
    """Calculate the total number of pages required for pagination."""
    return (total_candidates + CANDIDATES_PER_PAGE - 1) // CANDIDATES_PER_PAGE


def get_current_page_candidates(candidates: list[str]) -> list[str]:
    """Get the candidates for the current page."""
    start_idx = st.session_state.current_page * CANDIDATES_PER_PAGE
    end_idx = start_idx + CANDIDATES_PER_PAGE
    return candidates[start_idx:end_idx]


def update_score(candidate_key: str, slider_key: str) -> None:
    """Update the score for a candidate. This is the score that will be saved to the Google Sheet. It’s different from the score used to set the slider value, stored in st.session_state[slider_key]."""
    st.session_state[candidate_key] = st.session_state[slider_key]


# RENDERING FUNCTIONS


def render_header():
    st.markdown(get_css(), unsafe_allow_html=True)
    st.markdown(HEADER_TITLE, unsafe_allow_html=True)


def render_instructions():
    st.write(INSTRUCTIONS)


def render_candidates(current_page_candidates: list[str]) -> None:
    """Render sliders for candidates."""
    for candidate in current_page_candidates:
        initialize_slider(candidate)
        st.slider(
            candidate,
            0,
            100,
            key=f"slider_{candidate}",
            on_change=update_score,
            args=(f"score_{candidate}", f"slider_{candidate}"),
        )
        add_vertical_space()


def render_navigation_buttons(candidates: list[str], total_pages: int) -> None:
    """Render navigation buttons for pagination and submission."""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.session_state.current_page > 0 and st.button(PREVIOUS_PAGE_BUTTON):
            st.session_state.current_page = max(0, st.session_state.current_page - 1)
            st.session_state["page_change"] = True
            st.rerun()
    with col2:
        if st.session_state.current_page < total_pages - 1 and st.button(
            f"Page suivante ! {NEXT_BUTTON_MESSAGES[st.session_state.current_page % len(NEXT_BUTTON_MESSAGES)]}"
        ):
            st.session_state.current_page = min(
                total_pages - 1, st.session_state.current_page + 1
            )
            st.session_state["page_change"] = True
            st.rerun()
    with col3:
        if st.button(SUBMIT_BUTTON):
            confirm(candidates)


def render_page_count(total_pages) -> None:
    """
    Display the current page number and total pages in the voting process.
    """
    st.markdown(
        PAGE_COUNT_TEMPLATE.format(
            current=st.session_state.current_page + 1, total=total_pages
        ),
        unsafe_allow_html=True,
    )


# ACTION FUNCTIONS


def handle_page_change() -> None:
    """Handle page change events."""
    if st.session_state["page_change"]:
        scroll_to("trouvez-le-nouveau-nom-homo-fabulus")
        st.session_state["page_change"] = False


@st.dialog(CONFIRM_DIALOG_TITLE)
def confirm(candidates):
    """
    Display a confirmation dialog before submitting votes.

    This function shows two buttons:
    - One to cancel and return to the voting process.
    - One to confirm and submit the votes, triggering the save_data function.
    """
    st.write(CONFIRM_DIALOG_TEXT)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(CANCEL_BUTTON):
            st.rerun()
    with col2:
        if st.button(CONFIRM_BUTTON):
            with st.spinner(SPINNER_TEXT):
                save_data(candidates)


def save_data(candidates):
    """
    Save user voting data to a Google Sheet, ensuring no duplicate submissions.
    """
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet=WORKSHEET_NAME, ttl=0)
    # Define column names if the DataFrame is empty
    if df.empty:
        df_columns = [
            "User IP",
            "Session ID",
            "Submission Date and Time",
            "Session Duration (minutes)",
        ] + sorted(candidates)
        df = pd.DataFrame(columns=df_columns)

    # Prepare data for Google Sheets
    user_ip = st.session_state["user_ip"]
    submission_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_duration = (
        datetime.now() - st.session_state["start_time"]
    ).total_seconds() / 60
    session_id = st.session_state["session_id"]
    # check if user_ip or session_id already exists in the dataframe
    if df[(df["User IP"] == user_ip) | (df["Session ID"] == session_id)].empty:
        # Collect scores for all candidates
        scores = {
            candidate: st.session_state.get(f"score_{candidate}", NO_VOTE_VALUE)
            for candidate in candidates
        }
        # Sort candidates alphabetically
        sorted_scores = dict(sorted(scores.items()))

        # Prepare row data
        row_data = {
            "User IP": user_ip,
            "Session ID": session_id,
            "Submission Date and Time": submission_time,
            "Session Duration (minutes)": session_duration,
            **sorted_scores,
        }

        # Append row_data to the DataFrame
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        # Update the Google Sheet with the updated DataFrame
        conn.update(worksheet=WORKSHEET_NAME, data=df)

        st.switch_page(SUCCESS_PAGE)
    else:
        st.error(ALREADY_SUBMITTED_ERROR)


def main() -> None:
    initialize_session_state()

    candidates = st.session_state.shuffled_candidates
    total_pages = calculate_total_pages(len(candidates))

    handle_page_change()

    render_header()
    render_instructions()

    current_page_candidates = get_current_page_candidates(candidates)
    add_vertical_space()
    render_page_count(total_pages)
    add_vertical_space()
    render_candidates(current_page_candidates)
    render_page_count(total_pages)
    render_navigation_buttons(candidates, total_pages)


if __name__ == "__main__":
    main()
