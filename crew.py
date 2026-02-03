from openai import OpenAI
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


load_dotenv()
client = OpenAI()

