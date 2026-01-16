# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""instagram-post-agent - A Bindu Agent for Instagram Marketing Content Creation."""

import argparse
import asyncio
import json
import os
import re
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

import requests
from bindu.penguin.bindufy import bindufy
from crewai import LLM, Agent, Crew, Process, Task

# Tool imports
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from unstructured.partition.html import partition_html

# Load environment variables from .env file
load_dotenv()

# Error constants
ERROR_NO_API_KEY = "No API key available"

# Global variables
# We ONLY keep the LLM instance global. Crews are now local/stateless.
llm_instance: Any = None
_initialized = False
_init_lock = asyncio.Lock()


# --- TOOL DEFINITIONS ---


def _search_helper(query, n_results=5):
    """Perform Serper search with timeout."""
    if not os.environ.get("SERPER_API_KEY"):
        return "Error: SERPER_API_KEY not set."

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": os.environ["SERPER_API_KEY"], "content-type": "application/json"}

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        results = response.json().get("organic", [])
    except Exception as e:
        return f"Error performing search: {e}"

    strings = []
    for result in results[:n_results]:
        try:
            strings.append(
                "\n".join([
                    f"Title: {result.get('title', '')}",
                    f"Link: {result.get('link', '')}",
                    f"Snippet: {result.get('snippet', '')}",
                    "\n-----------------",
                ])
            )
        except KeyError:
            continue
    content = "\n".join(strings)
    return f"\nSearch result: {content}\n"


class ScrapeWebsiteInput(BaseModel):
    """Input schema for ScrapeWebsiteTool."""

    website: str = Field(..., description="The full URL of the website to scrape (e.g., https://google.com)")


class ScrapeWebsiteTool(BaseTool):
    """Tool for scraping and summarizing website content."""

    name: str = "Scrape website content"
    description: str = "Useful to scrape and summarize a website content. Pass only the full url."
    args_schema: type[BaseModel] = ScrapeWebsiteInput

    def _run(self, website: str) -> str:
        global llm_instance

        if not os.environ.get("BROWSERLESS_API_KEY"):
            return "Error: BROWSERLESS_API_KEY not set."

        url = f"https://chrome.browserless.io/content?token={os.environ['BROWSERLESS_API_KEY']}"
        payload = json.dumps({"url": website})
        headers = {"cache-control": "no-cache", "content-type": "application/json"}

        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            return f"Error scraping website: {e}"

        try:
            elements = partition_html(text=response.text)
            content = "\n\n".join([str(el) for el in elements])
            content_chunks = [content[i : i + 8000] for i in range(0, len(content), 8000)]
            summaries = []

            for chunk in content_chunks:
                # Use a minimal Crew to run the summarization task
                researcher_agent = Agent(
                    role="Principal Researcher",
                    goal="Summarize content",
                    backstory="Researcher",
                    llm=llm_instance,
                    allow_delegation=False,
                )
                summary_task = Task(
                    agent=researcher_agent,
                    description=f"Summarize the following content:\n\n{chunk}",
                    expected_output="A detailed summary of the content.",
                )
                # Create a minimal crew to execute the task
                summary_crew = Crew(
                    agents=[researcher_agent],
                    tasks=[summary_task],
                    verbose=False,
                    process=Process.sequential,
                    memory=False,
                )
                summary_result = summary_crew.kickoff()
                summaries.append(str(summary_result))
        except Exception as e:
            return f"Error processing scraped content: {e}"

        final_content = "\n\n".join(summaries)
        return f"\nScrapped Content: {final_content}\n"

    def _arun(self, website: str):
        raise NotImplementedError("Async not implemented")


class SearchInternetInput(BaseModel):
    """Input schema for SearchInternetTool."""

    query: str = Field(..., description="The search query topic")


class SearchInternetTool(BaseTool):
    """Tool for searching the internet."""

    name: str = "Search internet"
    description: str = "Useful to search the internet about a given topic and return relevant results."
    args_schema: type[BaseModel] = SearchInternetInput

    def _run(self, query: str) -> str:
        return _search_helper(query)

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")


class SearchInstagramInput(BaseModel):
    """Input schema for SearchInstagramTool."""

    query: str = Field(..., description="The search query for Instagram")


class SearchInstagramTool(BaseTool):
    """Tool for searching Instagram posts."""

    name: str = "Search instagram"
    description: str = "Useful to search for instagram posts about a given topic."
    args_schema: type[BaseModel] = SearchInstagramInput

    def _run(self, query: str) -> str:
        return _search_helper(f"site:instagram.com {query}")

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")


# --- HELPER FUNCTIONS ---


def _format_ad_copy_options(content: str) -> str:
    """Format ad copy options if missing proper structure."""
    if "Option 1:" not in content:
        lines = content.split("\n")
        formatted = []
        for i, line in enumerate(lines[:3], 1):
            if line.strip():
                formatted.append(f"Option {i}:\n{line.strip()}\n")
        content = "\n".join(formatted)
    return content


def _format_image_descriptions(content: str) -> str:
    """Format image descriptions if missing proper structure."""
    if "Photograph 1:" not in content:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        formatted = []
        for i, para in enumerate(paragraphs[:3], 1):
            formatted.append(f"Photograph {i}:\n{para}\n")
        content = "\n".join(formatted)
    return content


def validate_post_content(content: str, content_type: str) -> str:
    """Validate that generated content meets format requirements."""
    if not content:
        return f"ERROR: Could not generate {content_type}"

    if content_type == "ad_copy":
        option_pattern = re.compile(r"Option\s+\d+:|Copy\s+\d+:", re.IGNORECASE)
        options_found = list(option_pattern.finditer(content))

        if not options_found or len(options_found) < 2:
            print(f"⚠️  Warning: Ad copy has only {len(options_found)} options, expected at least 2")
            content = _format_ad_copy_options(content)

    elif content_type == "image_descriptions":
        desc_pattern = re.compile(r"Photograph\s+\d+:|Image\s+\d+:", re.IGNORECASE)
        descs_found = list(desc_pattern.finditer(content))

        if not descs_found or len(descs_found) < 2:
            print(f"⚠️  Warning: Image descriptions has only {len(descs_found)} options")
            content = _format_image_descriptions(content)

    return content


def extract_marketing_parameters_with_llm(input_text: str) -> dict:
    """Extract marketing parameters using LLM."""
    global llm_instance

    if not llm_instance:
        print("⚠️ LLM not initialized, falling back to defaults")
        return {
            "product_name": "Unknown Product",
            "target_audience": "General audience",
            "campaign_tone": "Engaging and professional",
            "key_message": "Highlight product benefits",
            "call_to_action": "Visit website or learn more",
            "hashtags": "#product #brand",
        }

    prompt = dedent(f"""
    You are an AI Marketing Assistant. Analyze the user's input and extract structured marketing details.

    USER INPUT: "{input_text}"

    OUTPUT FORMAT (JSON ONLY):
    {{
        "product_name": "Extracted product/service name",
        "target_audience": "Audience description",
        "campaign_tone": "Tone/voice for content",
        "key_message": "Main marketing message",
        "call_to_action": "Suggested call to action",
        "hashtags": "comma-separated hashtags"
    }}
    """)

    try:
        extractor_agent = Agent(
            role="Marketing Data Extractor",
            goal="Extract clean JSON marketing data from user input",
            backstory="You are a marketing expert who analyzes user requests to extract structured marketing parameters.",
            llm=llm_instance,
            verbose=False,
        )

        extraction_task = Task(
            description=prompt, expected_output="Valid JSON string with marketing parameters", agent=extractor_agent
        )

        crew_mini = Crew(agents=[extractor_agent], tasks=[extraction_task], verbose=False)
        result = str(crew_mini.kickoff())

        json_str = result.replace("```json", "").replace("```", "").strip()
        params = json.loads(json_str)

        final_params = {
            "product_name": params.get("product_name", "Product"),
            "target_audience": params.get("target_audience", "General audience"),
            "campaign_tone": params.get("campaign_tone", "Professional"),
            "key_message": params.get("key_message", ""),
            "call_to_action": params.get("call_to_action", "Learn more"),
            "hashtags": params.get("hashtags", "#marketing #product"),
        }

    except Exception as e:
        print(f"❌ LLM Extraction failed: {e}. Falling back to defaults.")
        return {
            "product_name": "Product",
            "target_audience": "General audience",
            "campaign_tone": "Professional",
            "key_message": "Discover amazing features",
            "call_to_action": "Visit our website",
            "hashtags": "#product #brand #marketing",
        }
    else:
        print(f"🔍 LLM Extraction Result: {final_params}")
        return final_params


def load_config() -> dict:
    """Load agent configuration from project root."""
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",
        Path(__file__).parent / "agent_config.json",
        Path.cwd() / "agent_config.json",
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {type(e).__name__}")
                continue

    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "instagram-post-agent",
        "description": "AI Instagram marketing agent",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENAI_API_KEY", "description": "OpenAI API key", "required": False},
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key", "required": True},
            {"key": "SERPER_API_KEY", "description": "Serper API key", "required": True},
            {"key": "BROWSERLESS_API_KEY", "description": "Browserless API key", "required": True},
        ],
    }


def _initialize_llm_with_error() -> LLM:
    """Initialize LLM with proper error handling."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    if openai_api_key and not openrouter_api_key:
        return LLM(model="gpt-4o", api_key=openai_api_key, temperature=0.7)

    if openrouter_api_key:
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openrouter_api_key
        return LLM(
            model=model_name,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

    raise ValueError(ERROR_NO_API_KEY)


async def initialize_crew() -> None:
    """Initialize the LLM connection only."""
    global llm_instance

    # If LLM is already setup, do nothing
    if llm_instance is not None:
        return

    try:
        llm_instance = _initialize_llm_with_error()
        model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")
        print(f"✅ Using LLM: {model_name}")

    except ValueError as e:
        print(f"❌ LLM initialization error: {e}")
        llm_instance = None
    except Exception as e:
        print(f"❌ Unexpected LLM initialization error: {e}")
        llm_instance = None


async def run_copy_crew(product_website: str, product_details: str, marketing_params: dict) -> str:
    """Run the copy crew and get the Instagram ad copy."""
    global llm_instance

    if not llm_instance:
        await initialize_crew()

    try:
        print(f"📝 Running copy crew for: {product_website}")

        # --- Instantiate Tools ---
        scrape_tool = ScrapeWebsiteTool()
        search_tool = SearchInternetTool()
        insta_tool = SearchInstagramTool()

        # --- Define Agents (Fresh per request) ---
        product_competitor_agent = Agent(
            role="Lead Market Analyst",
            goal=dedent("""\
                Conduct amazing analysis of products and competitors,
                providing in-depth insights to guide marketing strategies."""),
            backstory="Lead Market Analyst at a premier digital marketing firm.",
            tools=[scrape_tool, search_tool],
            allow_delegation=False,
            llm=llm_instance,
            verbose=False,
        )

        strategy_planner_agent = Agent(
            role="Chief Marketing Strategist",
            goal=dedent("""\
                Synthesize amazing insights from product analysis
                to formulate incredible marketing strategies."""),
            backstory="Chief Marketing Strategist at a leading digital marketing agency.",
            tools=[scrape_tool, search_tool, insta_tool],
            llm=llm_instance,
            verbose=False,
        )

        creative_content_creator_agent = Agent(
            role="Creative Content Creator",
            goal=dedent("""\
                Develop compelling and innovative content for social media campaigns,
                with a focus on creating high-impact Instagram ad copies."""),
            backstory="Creative Content Creator at a top-tier digital marketing agency.",
            tools=[scrape_tool, search_tool, insta_tool],
            llm=llm_instance,
            verbose=False,
        )

        # --- Define Tasks (Fresh per request) ---
        website_analysis_task = Task(
            description=dedent(f"""\
                Analyze the given product website: {product_website}.
                Extra details provided by the customer: {product_details}.

                Focus on identifying unique features, benefits, and the overall narrative presented.
                Your final report should clearly articulate the product's key selling points.
                """),
            expected_output="Product analysis report with key selling points",
            agent=product_competitor_agent,
        )

        market_analysis_task = Task(
            description=dedent(f"""\
                Explore competitors of: {product_website}.
                Extra details provided by the customer: {product_details}.

                Identify the top 3 competitors and analyze their strategies.
                """),
            expected_output="Competitor analysis report with top 3 competitors",
            agent=product_competitor_agent,
        )

        campaign_development_task = Task(
            description=dedent(f"""\
                You're creating a targeted marketing campaign for: {product_website}.
                Extra details: {product_details}.

                Start this campaign with a strategy and creative content ideas.
                It should be meticulously designed to captivate and engage the target audience.
                """),
            expected_output="Marketing campaign strategy with creative ideas",
            agent=strategy_planner_agent,
        )

        instagram_ad_copy_task = Task(
            description=dedent(f"""\
                Craft engaging Instagram post copy for {marketing_params["product_name"]}.

                Target audience: {marketing_params["target_audience"]}
                Campaign tone: {marketing_params["campaign_tone"]}
                Key message: {marketing_params["key_message"]}
                Call to action: {marketing_params["call_to_action"]}
                Hashtags: {marketing_params["hashtags"]}

                Your final answer MUST be 3 options for an ad copy for Instagram.
                """),
            expected_output="3 options for Instagram ad copy",
            agent=creative_content_creator_agent,
        )

        # --- Create Copy Crew (Fresh per request) ---
        copy_crew = Crew(
            agents=[product_competitor_agent, strategy_planner_agent, creative_content_creator_agent],
            tasks=[website_analysis_task, market_analysis_task, campaign_development_task, instagram_ad_copy_task],
            verbose=True,
            process=Process.sequential,
            memory=False,  # Disable memory to prevent cross-request contamination
        )

        # Inputs are injected via tasks now, but we pass them to kickoff for safety/logging
        inputs = {
            "product_website": product_website,
            "product_details": product_details,
        }

        # Run the copy crew
        result = copy_crew.kickoff(inputs=inputs)
        ad_copy = str(result)

        # Validate the ad copy
        return validate_post_content(ad_copy, "ad_copy")

    except Exception as e:
        error_msg = f"Copy crew execution failed: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return f"Error generating Instagram ad copy: {error_msg}"


async def run_image_crew(product_website: str, product_details: str, ad_copy: str) -> str:
    """Run the image crew and get the photograph descriptions."""
    global llm_instance

    if not llm_instance:
        await initialize_crew()

    try:
        print(f"🖼️ Running image crew for: {product_website}")

        # --- Instantiate Tools ---
        scrape_tool = ScrapeWebsiteTool()
        search_tool = SearchInternetTool()
        insta_tool = SearchInstagramTool()

        # --- Define Agents (Fresh per request) ---
        senior_photographer_agent = Agent(
            role="Senior Photographer",
            goal=dedent("""\
                Take the most amazing photographs for Instagram ads that
                capture emotions and convey a compelling message."""),
            backstory="Senior Photographer at a leading digital marketing agency.",
            tools=[scrape_tool, search_tool, insta_tool],
            llm=llm_instance,
            allow_delegation=False,
            verbose=False,
        )

        chief_creative_director_agent = Agent(
            role="Chief Creative Director",
            goal=dedent("""\
                Oversee the work done by your team to make sure it's the best possible.
                Review, approve, ask clarifying questions."""),
            backstory="Chief Content Officer of leading digital marketing agency.",
            tools=[scrape_tool, search_tool, insta_tool],
            llm=llm_instance,
            verbose=False,
        )

        # --- Define Tasks (Fresh per request) ---
        take_photo_task = Task(
            description=dedent(f"""\
                You MUST take the most amazing photo ever for an Instagram post regarding the product.

                Product URL: {product_website}
                Product Details: {product_details}

                Here is the Ad Copy we are using:
                {ad_copy}

                Imagine what the photo you want to take and describe it in a paragraph.
                Think creatively and focus on how the image can capture the audience's attention.
                Don't show the actual product in the photo unless specifically asked.

                Your final answer must be 3 options of photographs, each with 1 paragraph description.
                """),
            expected_output="3 options for photograph descriptions",
            agent=senior_photographer_agent,
        )

        review_photo_task = Task(
            description=dedent("""\
                Review the photos you got from the senior photographer.
                Make sure it's the best possible and aligned with the product's goals.

                Your final answer must be 3 reviewed options of photographs,
                each with 1 paragraph description.
                """),
            expected_output="3 reviewed photograph descriptions",
            agent=chief_creative_director_agent,
        )

        # --- Create Image Crew (Fresh per request) ---
        image_crew = Crew(
            agents=[senior_photographer_agent, chief_creative_director_agent],
            tasks=[take_photo_task, review_photo_task],
            verbose=True,
            process=Process.sequential,
            memory=False,  # Disable memory
        )

        inputs = {
            "product_website": product_website,
            "product_details": product_details,
            "ad_copy": ad_copy,
        }

        # Run the image crew
        result = image_crew.kickoff(inputs=inputs)
        image_descriptions = str(result)

        # Validate the image descriptions
        return validate_post_content(image_descriptions, "image_descriptions")

    except Exception as e:
        error_msg = f"Image crew execution failed: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return f"Error generating image descriptions: {error_msg}"


def _extract_user_input(messages: list[dict[str, str]]) -> str:
    """Extract user input from messages."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _extract_product_details(user_input: str) -> tuple[str, str]:
    """Extract product website and details from user input."""
    lines = user_input.split("\n")
    product_website = ""
    product_details = ""

    for line in lines:
        if "http" in line.lower():
            product_website = line.strip()
        elif line.strip() and not product_details:
            product_details = line.strip()

    if not product_details:
        product_details = user_input.replace(product_website, "").strip()
        if not product_details:
            product_details = "No additional details provided"

    return product_website, product_details


async def handler(messages: list[dict[str, str]]) -> str:
    """Handle incoming agent messages."""
    global _initialized

    # Type checking for messages
    if not isinstance(messages, list):
        return "Error: Invalid input format. Messages must be a list."

    # Lazy initialization of LLM
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Instagram Marketing LLM...")
            await initialize_crew()
            _initialized = True

    # Extract user input
    user_input = _extract_user_input(messages)
    if not user_input:
        return "Please provide marketing details. Example: 'Create Instagram content for https://example.com'"

    print(f"✅ Processing: {user_input}")

    # Extract product website and details
    product_website, product_details = _extract_product_details(user_input)

    # If no URL found, prompt for it
    if not product_website:
        return "Please provide a product website URL. Example: 'Create Instagram content for https://example.com'"

    try:
        # Extract marketing parameters
        marketing_params = extract_marketing_parameters_with_llm(user_input)

        # Generate ad copy
        ad_copy = await run_copy_crew(product_website, product_details, marketing_params)

        # Generate image descriptions
        image_descriptions = await run_image_crew(product_website, product_details, ad_copy)

        # Combine results
        result = f"""**Instagram Marketing Content**

**Ad Copy Options:**
{ad_copy}

**Photograph Descriptions:**
{image_descriptions}

**Marketing Parameters:**
- Product: {marketing_params["product_name"]}
- Target Audience: {marketing_params["target_audience"]}
- Campaign Tone: {marketing_params["campaign_tone"]}
- Key Message: {marketing_params["key_message"]}
- Call to Action: {marketing_params["call_to_action"]}
- Hashtags: {marketing_params["hashtags"]}
"""

    except Exception as e:
        error_msg = f"Handler error: {e!s}"
        print(f"❌ {error_msg}")
        return f"Error processing your request: {error_msg}"
    else:
        print("✅ Success! Generated Instagram marketing content")
        return result


async def cleanup() -> None:
    """Clean up resources."""
    print("🧹 Cleaning up...")
    # No global crews to clean up anymore
    print("✅ Cleanup complete")


def main() -> None:
    """Run the main entry point for the Instagram Marketing Agent."""
    parser = argparse.ArgumentParser(description="Bindu Instagram Marketing Agent")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key",
    )
    parser.add_argument(
        "--serper-api-key",
        type=str,
        default=os.getenv("SERPER_API_KEY"),
        help="Serper API key for search",
    )
    parser.add_argument(
        "--browserless-api-key",
        type=str,
        default=os.getenv("BROWSERLESS_API_KEY"),
        help="Browserless API key for web scraping",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID",
    )
    args = parser.parse_args()

    # Set environment variables
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = args.openrouter_api_key
    if args.serper_api_key:
        os.environ["SERPER_API_KEY"] = args.serper_api_key
    if args.browserless_api_key:
        os.environ["BROWSERLESS_API_KEY"] = args.browserless_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Instagram Marketing Agent")
    print("📱 Creates Instagram ad copy and visual content for marketing campaigns")

    config = load_config()

    try:
        print("🚀 Starting server...")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import sys

        sys.exit(1)
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
