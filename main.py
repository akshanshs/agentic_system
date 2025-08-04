from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # loads .env and sets api key
client = OpenAI()  # reads api key


def generate_x_post(usr_input: str) -> str:
    prompt = f"""
        You are an expert social media manager, and you excel at crafting viral and highly engaging posts for X (formerly Twitter).

        Your task is to generate a post that is concise, impactful, and tailored to the topic provided by the user.
        Avoid using hashtags and lots of emojis (a few emojis are okay, but not too many).

        Keep the post short and focused, structure it in a clean, readable way, using line breaks and empty lines to enhance readability.

        Here's the topic provided by the user for which you need to generate a post:
        <usr_input>
        {usr_input}
        </usr_input>
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": usr_input}
        ],
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

def main():
    usr_input = input("What should the post be about? ")
    x_post = generate_x_post(usr_input)
    print("The generated post is")
    print(x_post)

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
