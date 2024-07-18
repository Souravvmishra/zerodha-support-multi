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

# Say This Not That Bot setup
rephrasing_expert = Agent(
    role='Rephrasing Expert',
    goal='Rephrase statements to be more effective and appropriate.',
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in effective communication and language. Your job is to help users "
        "rephrase their statements to be more clear, polite, or impactful depending on the context."
    )
)

tone_analyzer = Agent(
    role='Tone Analyzer',
    goal='Analyze the tone of statements and suggest improvements.',
    verbose=True,
    memory=True,
    backstory=(
        "You are skilled at analyzing the tone and emotional impact of language. You help identify "
        "areas where the tone could be improved to better achieve the speaker's goals."
    )
)

# Centralized Task for determining user query context and responding appropriately
centralized_task = Task(
    description=(
        "Determine the context of the user query and respond appropriately. "
        "If the query is related to Zerodha services, provide a detailed and informative response. "
        "If the query is out of context, respond politely indicating that the question is out of context. "
        "If the query requires rephrasing or tone analysis, handle that as well."
        "If the query does not fall in any one of the two things, tell him that it is out of context. and we can only provide two types of services."
        "Don't answer general queries"
        "User query: {user_query}"
    ),
    expected_output='An appropriate response based on the context of the user query.',
    agent=Agent(
        role='Centralized Bot',
        goal='Determine the context of user queries and respond appropriately.',
        verbose=True,
        memory=True,
        backstory=(
            "You are an intelligent bot capable of determining the context of user queries and delegating tasks "
            "to the appropriate agents to provide the best response."
            "If the query does not fall in any one of the two things, tell him that it is out of context. and we can only provide two types of services."
        ),
        tools=[search_tool],
        allow_delegation=True
    )
)

# Centralized Crew setup
centralized_crew = Crew(
    agents=[zerodha_support_agent, out_of_context_agent, rephrasing_expert, tone_analyzer],
    tasks=[centralized_task],
    process=Process.sequential
)

# Streamlit UI
st.title("Multi-Bot Assistant")
user_input = st.text_area("Enter your question or statement:")

if user_input:
    with st.spinner("Processing your input..."):
        result = centralized_crew.kickoff(inputs={'user_query': user_input})
        st.write(result)
