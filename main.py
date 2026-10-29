
import os
from datetime import date

from load_dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool


# --------------------------------------------------
# 1. Read API keys and model from environment variables
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("MODEL", "gemini/gemini-flash-latest")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is missing.")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY environment variable is missing.")

# Ensure CrewAI's OpenAI integration can access the API key.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

today = date.today().isoformat()

# --------------------------------------------------
# 2. Initialize the LLM and search tool
# --------------------------------------------------

# llm = LLM(model=MODEL)
llm = LLM(
    model="ai/qwen2.5:0.5B-F16",
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
    custom_openai=True,
)

search_tool = SerperDevTool()


# --------------------------------------------------
# 3. Create the Researcher agent
# --------------------------------------------------

researcher = Agent(
    role="Technology News Researcher",
    goal=(
        "Find today's three most important technology news stories "
        "using reliable, current web sources."
    ),
    backstory=(
        "You are an experienced technology journalist who specializes "
        "in AI, software engineering, cybersecurity, cloud computing, "
        "and major technology industry developments. You verify dates, "
        "prefer reliable sources, and distinguish confirmed facts "
        "from speculation."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False,
)


# --------------------------------------------------
# 4. Create the Writer agent
# --------------------------------------------------

writer = Agent(
    role="Technology Newsletter Writer",
    goal=(
        "Transform research findings into an engaging, accurate, "
        "reader-friendly newsletter containing exactly five lines."
    ),
    backstory=(
        "You are a professional technology newsletter editor. "
        "You write concise, informative content for developers "
        "and technology professionals. You use only the research "
        "provided to you and never invent facts or sources."
    ),
    tools=[],  # The Writer cannot search the web.
    llm=llm,
    verbose=False,
    allow_delegation=False,
)


# --------------------------------------------------
# 5. Task 1: Research today's top three tech stories
# --------------------------------------------------

research_task = Task(
    description=f"""
    Research the top three technology news stories for {today}.

    Requirements:
    1. Search the web using your available search tool.
    2. Prioritize significant developments in AI, software,
       cybersecurity, cloud computing, and the technology industry.
    3. Verify that each story is relevant to today's date.
    4. Prefer reputable sources and original announcements.
    5. Do not present old news as breaking news.
    6. Do not invent facts, dates, headlines, or URLs.

    Return exactly three numbered stories using this format:

    STORY 1
    Headline: ...
    Published date: ...
    Summary: ... (2-3 concise sentences)
    Source: ...
    URL: ...

    STORY 2
    Headline: ...
    Published date: ...
    Summary: ... (2-3 concise sentences)
    Source: ...
    URL: ...

    STORY 3
    Headline: ...
    Published date: ...
    Summary: ... (2-3 concise sentences)
    Source: ...
    URL: ...

    If a story's publication date cannot be verified, say so.
    Do not claim a story was published today unless the source
    supports that claim.
    """,
    expected_output=(
        "A clean research report containing exactly three technology "
        "news stories. Each story must include its headline, publication "
        "date or an explicit note that the date is unverified, a concise "
        "factual summary, the source name, and the source URL. Use the "
        "specified STORY 1, STORY 2, STORY 3 structure."
    ),
    agent=researcher,
)


# --------------------------------------------------
# 6. Task 2: Write the five-line newsletter
# --------------------------------------------------

writer_task = Task(
    description="""
    Write a technology newsletter using ONLY the research report
    supplied by the Researcher.

    Newsletter requirements:
    1. Produce exactly FIVE physical lines.
    2. Use exactly five bullet points, one bullet per line.
    3. Line 1: A short, engaging newsletter headline.
    4. Lines 2-4: One concise news story per line, covering all
       three researched stories. Include each story's source URL.
    5. Line 5: A useful overall takeaway for technology professionals.
    6. Keep each line concise and informative.
    7. Do not add an introduction, conclusion, or extra commentary
       outside the five bullet points.
    8. Do not invent facts or sources.
    9. Do not perform web searches. You have no search tools.
    10. If the research is incomplete, be transparent rather than
        inventing missing details.

    Return ONLY the five-line newsletter.
    """,
    expected_output=(
        "A technology newsletter consisting of exactly five physical "
        "lines and five bullet points: one headline, three news stories "
        "with source URLs, and one overall takeaway. No extra text."
    ),
    agent=writer,
    context=[research_task],  # Explicitly pass the Researcher's output.
)


# --------------------------------------------------
# 7. Assemble the sequential crew
# --------------------------------------------------

news_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    verbose=False,
)


# --------------------------------------------------
# 8. Execute the crew
# --------------------------------------------------

if __name__ == "__main__":
    print(f"\nTechnology News Newsletter — {today}")
    print("=" * 60)

    result = news_crew.kickoff()

    print("\nFINAL NEWSLETTER")
    print("=" * 60)
    print(result.raw)
