import os
from flask import Flask, request, render_template, jsonify
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up the environment keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY") # Replace with your Serper API key

# Initialize the SerperDevTool
search_tool = SerperDevTool()

# Define the Zerodha Support Agent with the search tool
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

# Define the Task for the Zerodha Support Agent
support_task = Task(
    description=(
        "Respond to user queries regarding Zerodha services in a helpful and engaging manner. "
        "Use the provided tools to fetch up-to-date information from the internet if needed. "
        "User query: {user_query}"
    ),
    expected_output='A detailed and informative response to the user query about Zerodha services.',
    agent=zerodha_support_agent
)

# Assemble the Crew
zerodha_support_crew = Crew(
    agents=[zerodha_support_agent],
    tasks=[support_task],
    process=Process.sequential
)

def kickoff_zerodha_support(user_input):
    result = zerodha_support_crew.kickoff(inputs={'user_query': user_input})
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']  # Parse JSON data correctly
    response = kickoff_zerodha_support(user_input)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
