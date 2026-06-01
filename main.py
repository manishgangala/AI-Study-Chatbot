"""
Smart study chatbot using gemini API
A Beginner-friendly AI Study mentor built using:
- Python
- Google Gemini API
- Prompt Engineering
- Environment Variables

Features:
- Secure API key handling
- AI Study Mentor personality
- Continuous conversation loop
- Beginner-friendly structure
"""
#section 1: import required libraries

import os

from dotenv import load_dotenv
from google import genai

#section 2: load environment variables

# load variables from .env file
load_dotenv()

#section 3: get gemini api key

# read api key securely from .env
api_key = os.getenv("GEMINI_API_KEY")

# stop program if key is missing
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

#section 4: create gemini client

# create gemini client connection
client = genai.Client(api_key=api_key)

#section 5: select model

# latest stable gemini model
MODEL_NAME = "gemini-2.5-flash"

#section 6: system prompt

# this controls chatbot personality and behaviour
SYSTEM_PROMPT = """
you are an expert AI study mentor.
your job is to:
- teach students clearly
- explain concepts simply
- provide roadmap guidance
- help students stay motivated
- answer in beginner-friendly language

Rules:
- Keep answers practical and easy to understand
- Avoid overly technical jargon unless necessary
- Give structured and useful responses
- Encourage learning and curiosity
"""

#section 7: welcome message

print("\n===================================")
print("      SMART STUDY CHATBOT READY")
print("===================================")
print(f"using model: {MODEL_NAME}")
print("type 'exit' or 'quit' to stop.\n")

#section 8: chatbot loop

while True:
    # take user input
    user_input = input("You: ").strip()
    # exit condition
    if user_input.lower() in ["exit", "quit"]:  # Exit #EXIT #QUIT #Quit
        print("\nChatbot: Goodbye! Keep learning AI 🚀")
        break
    # handle empty input
    if not user_input:
        print("chatbot: please enter a valid question.")
        continue

    try:

        # combine system prompt + user question
        full_prompt = f"""
        {SYSTEM_PROMPT}

        Student Question:
        {user_input}
        """

        # generate ai response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )
        # print chatbot response
        print(f"\nChatbot: {response.text}")

    except Exception as error:
        print("\nChatbot: Something went wrong.")
        print(f"Error Details: {error}\n")





