# 📚 Study Assistant – AI Question-Answering Chatbot

A simple command-line chatbot built with **Python**, **LangChain** and **Google Gemini**.
A student types any academic question or topic and the Study Assistant replies with a
clear, simple, step-by-step explanation with examples and a short summary.

LangChain Mini-Project – topic: **Study Assistant**

## Features
- Uses one language model: Google Gemini (`gemini-3.8-flash`) through LangChain
- LangChain `ChatPromptTemplate` with Study Assistant instructions
- Takes the student's question as input in the terminal
- Prints the AI-generated answer
- API key stored safely in a `.env` file (never in the code)
- Basic error handling: missing API key, empty input, API/network errors, Ctrl + C

## Project structure
```
langchain_project/
├── app.py            # main chatbot program
├── requirements.txt  # Python libraries needed
├── commands.txt      # setup and run commands for macOS / VS Code
├── README.md         # this file
├── .env.example      # template for the .env file (no real key)
└── .env              # your real API key (create it yourself, never share it)
```

## How to run (macOS)
```bash
# 1. Create and activate an environment
conda create -n langchain_env python=3.11 -y
conda activate langchain_env
#    (or: python3 -m venv venv && source venv/bin/activate)

# 2. Go to the project folder and install the libraries
cd ~/Documents/langchain_project
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Create .env and add your Gemini API key
cp .env.example .env
#    then edit .env:  GOOGLE_API_KEY=your_gemini_api_key_here

# 4. Run
python3 app.py
```
Type `exit` or `quit` (or press Ctrl + C) to stop. See `commands.txt` for the full,
step-by-step commands and common errors.

Get a free Gemini API key at https://aistudio.google.com/apikey

## Example questions
1. Explain photosynthesis in simple words.
2. What is the difference between a list and a tuple in Python?
3. How do I solve 2x + 5 = 15 step by step?
4. Explain Newton's three laws of motion with examples.
5. What were the main causes of World War I?

## Technologies
Python 3.11 · LangChain · langchain-google-genai · Google Gemini · python-dotenv
