# Import necessary modules
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# Load variables from .env
load_dotenv()  
API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    api_key= API_KEY,
    model="gemini-2.0-flash-exp"
    )


# Define a prompt template
prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)

# Build the LangChain Pipeline
chain = prompt_template | llm


# Define a user question
response = chain.invoke({"question" : "Who won the ICC world cup in 1992? And who was the captain of winning team?"})


# Display the result
print("Response:", response.content)