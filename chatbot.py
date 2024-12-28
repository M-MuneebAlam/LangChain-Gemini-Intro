from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Function to get a response from the chatbot
def get_response(user_input, chat_history, api_key):
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
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

    # Build the LangChain pipeline
    chain = prompt | llm

    # Initialize the message history
    history = ChatMessageHistory()

    # Add previous chat history to memory if it's available
    for speaker, message in chat_history:
        if speaker == "You":
            history.add_message(message)
        elif speaker == "Assistant":
            history.add_message(message)
    
    # Create the chain with message history
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Get response from the chain
    response = chain_with_message_history.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "unused"}}
    )

    return response.content