import random
import os
import streamlit as st
from nltk.corpus import words

# load the word list from nltk library
dictionary_words = set(words.words())

# this function will randomly choose a word from a file depending on difficulty
def get_random_word(difficulty):
    file_naming = {
        "easy": "easy_words.txt",
        "medium": "medium_words.txt",
        "difficult": "difficult_words.txt"
    }

    # get the file path for the chosen difficulty word bank
    file_path = file_naming.get(difficulty)
    if not file_path:
        raise ValueError(f"Invalid difficulty selected: {difficulty}")

    if not os.path.exists(file_path):  # verify that the file exists
        raise FileNotFoundError(f"Word list file for {difficulty} not found at {os.path.abspath(file_path)}")

    with open(file_path, "r") as file:
        word_bank = file.read().splitlines()  # read the text in the chosen file

    # use random module to get a random word from list
    secret_word = random.choice(word_bank).strip().lower()
    return secret_word

# this checks that the word exists and is proper length
def is_valid_word(word):
    return word in dictionary_words and len(word) == 5

# this function formats each guess with the appropriate colors
def format_guess(guess, secret_word):
    formatted_guess = ""
    for i, letter in enumerate(guess):
        if letter == secret_word[i]:
            formatted_guess += f'<span style="color: white; background-color: green; padding: 2px 5px; border-radius: 5px;">{letter}</span> '
        elif letter in secret_word:
            formatted_guess += f'<span style="color: black; background-color: yellow; padding: 2px 5px; border-radius: 5px;">{letter}</span> '
        else:
            formatted_guess += f'<span style="color: white; background-color: red; padding: 2px 5px; border-radius: 5px;">{letter}</span> '
    return formatted_guess.strip()

def update_available_letters(available_letters, guess, secret_word):
    for i, letter in enumerate(guess):  # to update the alphabet -- iterate over each letter in the user guess
        if letter == secret_word[i]:  # if letter is in correct spot -> alphabet letter = green
            available_letters[letter] = 'green'
        elif letter in secret_word and available_letters[letter] != 'green':
            available_letters[letter] = 'yellow'
        elif letter not in secret_word:
            available_letters[letter] = 'red'
    return available_letters

# this function formats the alphabet depending on the status of the letters
def display_available_letters(available_letters):
    letter_str = "Available Letters: "
    for letter, status in available_letters.items():
        if status == 'green':
            letter_str += f'<span style="color: green; font-weight: bold;">{letter}</span> '
        elif status == 'yellow':
            letter_str += f'<span style="color: orange; font-weight: bold;">{letter}</span> '
        elif status == 'red':
            letter_str += f'<span style="color: red; font-weight: bold;">{letter}</span> '
        else:
            letter_str += f'{letter} '
    st.markdown(letter_str, unsafe_allow_html=True)

st.title("Wordle 2.0")
st.write("Guess the 5-letter secret word within 6 attempts!")

st.markdown(
    """
    **How to Play:**
    - <span style="color: white; background-color: green; padding: 2px 5px; border-radius: 5px;">Green Letter</span>: In the secret word and in the correct spot! 😊
    - <span style="color: black; background-color: yellow; padding: 2px 5px; border-radius: 5px;">Yellow Letter</span>: In the secret word but not in the correct spot! 😐
    - <span style="color: white; background-color: red; padding: 2px 5px; border-radius: 5px;">Red Letter</span>: Not in the secret word. 😢
    """,
    unsafe_allow_html=True,
)

# initialize session
if "secret_word" not in st.session_state:
    st.session_state.difficulty = None
    st.session_state.secret_word = None
    st.session_state.guesses = []
    st.session_state.feedback = []
    st.session_state.available_letters = {letter: '' for letter in "abcdefghijklmnopqrstuvwxyz"}
    st.session_state.game_over = False

if st.session_state.secret_word is None:
    difficulty = st.selectbox("Choose difficulty:", ["easy", "medium", "difficult"], index=1)
    if st.button("Start Game"):
        st.session_state.difficulty = difficulty
        st.session_state.secret_word = get_random_word(difficulty)

if st.session_state.secret_word:
    st.write(f"**The difficulty level you are playing at is: {st.session_state.difficulty.capitalize()}**")

if st.session_state.secret_word and not st.session_state.game_over:
    guess = st.text_input("Enter your 5-letter guess:", key="guess", placeholder="Type your guess here...").lower()

    if st.button("Submit Guess"):
        if len(guess) != 5 or not is_valid_word(guess):
            st.error("Invalid input! Please enter a valid 5-letter word.")
        else:
            st.session_state.guesses.append(guess)
            st.session_state.feedback.append(format_guess(guess, st.session_state.secret_word))
            st.session_state.available_letters = update_available_letters(
                st.session_state.available_letters, guess, st.session_state.secret_word
            )

            # check if the game is over
            if guess == st.session_state.secret_word:
                st.session_state.game_over = True
            elif len(st.session_state.guesses) == 6:
                st.session_state.game_over = True

if st.session_state.guesses:
    for i, (guess, feedback) in enumerate(zip(st.session_state.guesses, st.session_state.feedback), 1):
        st.markdown(f"**Guess #{i}:** {feedback}", unsafe_allow_html=True)

    display_available_letters(st.session_state.available_letters)

    if st.session_state.game_over:
        if st.session_state.guesses[-1] == st.session_state.secret_word:
            st.success(f"🎉 Congrats! You guessed the word: {st.session_state.secret_word}")
        else:
            st.error(f"❌ Game Over! The word was: {st.session_state.secret_word}")

        if st.button("Play Again"):
            # reset game state for a new game
            del st.session_state.secret_word
            del st.session_state.guesses
            del st.session_state.feedback
            del st.session_state.available_letters
            del st.session_state.game_over
            del st.session_state.difficulty
