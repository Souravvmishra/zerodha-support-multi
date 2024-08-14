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

# Initialize the SerperDevTool with Axi.com-related search settings
class AxiSerperDevTool(SerperDevTool):
    def search(self, query):
        # Add a prefix to filter results to Axi.com-related content
        axi_query = f"site:axi.com {query}"
        results = super().search(axi_query)
        # Filter results to include only Axi.com content
        relevant_results = [result for result in results if 'axi.com' in result.get('link', '')]
        return relevant_results

# Initialize the customized search tool
search_tool = AxiSerperDevTool()

# Axi.com Information Agent setup
axi_info_agent = Agent(
    role='Axi.com Information Specialist',
    goal='Provide accurate and detailed information about Axi.com products, services, and trading conditions.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a knowledgeable specialist in Axi.com's offerings. You provide detailed information "
        "about their trading platforms, financial instruments, account types, and market analysis tools."
    ),
    tools=[search_tool]
)

# Out-of-Context Agent setup
out_of_context_agent = Agent(
    role='Context Checker',
    goal='Determine if a question is relevant to Axi.com and politely decline if not.',
    verbose=True,
    memory=True,
    backstory=(
        "You are responsible for determining if a question is relevant to Axi.com. "
        "If the question is not related, you respond politely indicating that the question is out of context and "
        "that only Axi.com-related information is provided."
    )
)

# Centralized Task for determining user query context and responding appropriately
centralized_task = Task(
    description=(
        "Determine if the user query is related to Axi.com and respond appropriately. "
        "If the query is about Axi.com, provide a detailed and informative response. "
        "If the query is out of context, respond politely indicating that only Axi.com-related information is provided. "
        "User query: {user_query}"
    ),
    expected_output='A detailed response based on the context of the user query, focusing on Axi.com information.',
    agent=Agent(
        role='Axi.com Information Bot',
        goal='Provide comprehensive information about Axi.com and its offerings.',
        verbose=True,
        memory=True,
        backstory=(
            "You are an intelligent bot specializing in Axi.com information. You provide detailed responses "
            "about Axi.com's trading platforms, financial instruments, account types, and market analysis tools. "
            "You only respond to queries related to Axi.com."
        ),
        tools=[search_tool],
        allow_delegation=True
    )
)

# Centralized Crew setup
centralized_crew = Crew(
    agents=[axi_info_agent, out_of_context_agent],
    tasks=[centralized_task],
    process=Process.sequential
)

# Streamlit UI
st.title("Axi.com Information Assistant")
user_input = st.text_area("Enter your question about Axi.com:")

if user_input:
    with st.spinner("Processing your input..."):
        result = centralized_crew.kickoff(inputs={'user_query': user_input})
        st.write(result)
