import os
import json
import openai


with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']


client = openai.OpenAI(
    api_key=llm_api_key,
    base_url=llm_api_base,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "celsius",
                            "fahrenheit"
                        ]
                    }
                }
            }
        }
    }
]
messages = [
    {"role": "system", "content": "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls."},
    {"role": "user", "content": "What is the current temperature of New York, San Francisco and Chicago?"}
]
response = client.chat.completions.create(
    model="/models/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(response.json)
print(json.dumps(response.choices[0].message.model_dump()['tool_calls'], indent=2))
