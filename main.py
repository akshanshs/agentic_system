import json

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # loads .env and sets api key
client = OpenAI()  # reads api key


def get_website_html(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the url {url}: {e}")
        return ""


def extract_core_website_content(html: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
            You are an expert web content extractor. Your task is to extract the core content from a given HTML page.
            The core content should be the main text, excluding navigation, footers, and other non-essential elements like scripts etc.
            
            Here is the HTML content:
            <html>
            {html}
            </html>
            
            Please extract the core content and return it as plain text.
        """
    )
    return response.output_text


def summarize_content(content: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
        You are an expert summarizer. Your task is to summarize the provided content into a concise and clear summary.
        
        Here is the content to summarize:
        <content>
        {content}
        </content>
        
        Please provide a brief summary of the main points in the content. Prefer bullet points and avoid unnecessary explanations.
    """
    )

    return response.output_text


def generate_x_post(summary: str) -> str:
    with open("post-examples.json", "r") as f:
        examples = json.load(f)

    example_str = ""
    for i, example in enumerate(examples, 1):
        example_str += f"""
        <example-{i}>
            <topic>
            {example['topic']}
            </topic>
            
            <generated-post>
            {example['post']}
            </generated-post>
        </example-{i}>
        """

    prompt = f"""
        You are an expert social media manager, and you excel at crafting viral and highly engaging posts for X (formerly Twitter).

        Your task is to generate a post that is concise, impactful, and tailored to the topic provided by the user.
        Avoid using hashtags and lots of emojis (a few emojis are okay, but not too many).

        Keep the post short and focused, structure it in a clean, readable way, using line breaks and empty lines to enhance readability.

        Here's the topic provided by the user for which you need to generate a post:
        <summary>
        {summary}
        </summary>
        
        Here are some examples of topics and generated posts:
        <examples>
        {example_str}
        </examples>
        
        Please use the tone, language, structure , and style of the examples provided above to generate a post that is engaging and relevant to the topic provided by the user.
        Don't use the content from the examples!
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text


def main():
    website_url = input("Website URL: ")
    print("Fetching website HTML . . . .")
    try:
        html_content = get_website_html(website_url)
    except Exception as e:
        print(f"An error occurred while fetching the website: {e}")
        return

    if not html_content:
        print("Failed to fetch the website content. Exiting.")
        return

    print("---------")
    print("Extracting core content from the website...")
    core_content = extract_core_website_content(html_content)
    print("Extracted core content:")
    print(core_content)

    print("---------")
    print("Summarizing the core content...")
    summary = summarize_content(core_content)
    print("Generated summary:")
    print(summary)

    print("---------")
    print("Generating X post based on the summary...")
    x_post = generate_x_post(summary)
    print("Generated X post:")
    print(x_post)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
