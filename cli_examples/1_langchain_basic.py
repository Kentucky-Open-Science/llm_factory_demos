import json

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

with open('../config.json') as user_file:
    config = json.load(user_file)


llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

llm = ChatOpenAI(
    model_name="",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base,
    verbose=True
)

def process_query(query):

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    response = llm_chain.invoke(query)

    return response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    q1 = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    response = process_query(q1)
    print('question:', q1)
    print('response:', response.content)


