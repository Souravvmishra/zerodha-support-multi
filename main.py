import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the environment keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Initialize the SerperDevTool
search_tool = SerperDevTool()

# Zerodha Support Bot setup
zerodha_support_agent = Agent(
    role='Zerodha Support Agent',
    goal='Assist users with their queries regarding Zerodha services.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a knowledgeable support agent for Zerodha, equipped to assist users with their queries about Zerodha's trading platform, account management, and other services."
    ),
    tools=[search_tool]
)

support_task = Task(
    description=(
        "Respond to user queries regarding Zerodha services in a helpful and engaging manner. "
        "Use the provided tools to fetch up-to-date information from the internet if needed. "
        "User query: {user_query}"
    ),
    expected_output='A detailed and informative response to the user query about Zerodha services.',
    agent=zerodha_support_agent
)

zerodha_support_crew = Crew(
    agents=[zerodha_support_agent],
    tasks=[support_task],
    process=Process.sequential
)

# Out-of-Context Agent setup
out_of_context_agent = Agent(
    role='Context Checker',
    goal='Determine if a question is relevant to Zerodha services.',
    verbose=True,
    memory=True,
    backstory=(
        "You are responsible for determining if a question is relevant to Zerodha services. "
        "If the question is not related, you respond politely indicating the same."
    )
)

context_check_task = Task(
    description=(
        "Determine if the user query is related to Zerodha services. "
        "If not, respond with a polite message indicating that the question is out of context."
        "User query: {user_query}"
    ),
    expected_output='A polite response indicating if the query is out of context.',
    agent=out_of_context_agent
)

context_check_crew = Crew(
    agents=[out_of_context_agent],
    tasks=[context_check_task],
    process=Process.sequential
)

def kickoff_context_check(user_input):
    result = context_check_crew.kickoff(inputs={'user_query': user_input})
    return result

def kickoff_zerodha_support(user_input):
    result = zerodha_support_crew.kickoff(inputs={'user_query': user_input})
    return result

# Streamlit UI
st.title("Zerodha Support Bot")

user_input = st.text_input("Enter your question about Zerodha services:")

if st.button("Get Answer"):
    if user_input:
        with st.spinner("Processing your query..."):
            context_check_response = kickoff_context_check(user_input)
            if 'out of context' in context_check_response.lower():
                st.write(context_check_response)
            else:
                response = kickoff_zerodha_support(user_input)
                st.write(response)
    else:
        st.write("Please enter a question.")


