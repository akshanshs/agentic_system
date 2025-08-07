from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # loads .env and sets api key
client = OpenAI()  # reads api key


def get_temperature(city: str) -> float:
    """
    Get current temperature for a given city.
    :param city:
    :return:
    """
    return 20.0


def main():
    print("Hello from agent")
    user_input = input("Your question: ")
    prompt = f"""
    you are a helpful assistant. Answer the user;s question in a friendlu way.
    you can also use tools if oyu feel like they help you provide a better answer:
        - get_temperature(city: str) -> float: Get the current temperture for a given city.
        
    If you want to use one of these tools, you should output the tool name and its arguments in the following format:
        tool_name: arg1, arg2, ....
    for example:
        get_temperature: Berlin
    with that in mind, answer the user's question:
    <user-question>
    {user_input}
    </user-question>
    
    If you request a tool, please output ONLY the tool call (as described above) and nothing else.
    """

    response = client.responses.create(
        model='gpt-4o-mini',
        input=prompt
    )
    reply = response.output_text

    if reply.startswith("get_temperature"):
        arg = reply.split(":", 1)[1].strip()
        temperature = get_temperature(arg)
        prompt = f"""
        You are a helpful assistant. Answer the user's question in a friendly way.
        Here's the user's question:
        <user-question>
        {user_input}
        </user-question>
        
        You requested to use the get_temperature tool for the city "{arg}".
        Here's the result of using that tool:
        The current temperature in {arg} is {temperature}°C.
        """

        response = client.responses.create(
           model="gpt-4o-mini",
           input=prompt
        )
        reply = response.output_text
        print(reply)
    else:
        print(reply)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
