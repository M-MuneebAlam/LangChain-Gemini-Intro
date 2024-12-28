# WARNING:
# There is an issue where the chat and history gets duplicated in the console after some time.


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


# Initialize the message history
history = ChatMessageHistory()

# Create the chain with message history
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Chat loop
print("Chat started. Type your message below. Type 'Exit' to end the chat.")
session_id = "unused"  # You can customize session handling as needed
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Chat ended.")
        break

    # Get response from the chain
    response = chain_with_message_history.invoke(
        {"input": user_input},
        {"configurable": {"session_id": session_id}},
    )

    # Display the response
    print(f"Assistant: {response.content}")



# Print chat history in a cleaner format  
print("\nChat History:")
for message in history.messages:
    if message.type == "human":
        print(f"You: {message.content}")
    elif message.type == "ai":
        print(f"Assistant: {message.content}")