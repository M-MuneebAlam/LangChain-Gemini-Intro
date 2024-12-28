# Import necessary modules
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Load variables from .env
load_dotenv()  
API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    api_key= API_KEY,
    model="gemini-2.0-flash-exp"
    )


# Define a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# Build the LangChain Pipeline
chain = prompt | llm



history = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

response1 = chain_with_message_history.invoke(
    {"input": "Who won the ICC world cup in 1992?"},
    {"configurable": {"session_id": "unused"}},
)

response2 = chain_with_message_history.invoke(
    {"input": "Who was the captain of winning team?"},
    {"configurable": {"session_id": "unused"}},
)



# Display the result
print("First Response:", response1.content)
print("\n\nSecond Response:", response2.content)



# Access the chat history messages:
chat_history_messages = history.messages

# Print the chat history messages:
for message in chat_history_messages:
    print(f"{message.type}: {message.content}")

