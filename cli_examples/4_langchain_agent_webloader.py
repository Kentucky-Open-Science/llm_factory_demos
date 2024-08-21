#See https://python.langchain.com/v0.1/docs/modules/agents/quick_start/

import json

from langchain_core.load import dumpd, dumps, load, loads


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_caai.caai_emb_client import caai_emb_client

with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

llm = ChatOpenAI(
    model="",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base,
    verbose=True,
    streaming=False
)

embeddings = caai_emb_client(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
    max_batch_size=100,
    num_workers=10
)

def save_prompt():

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")

    string_representation = dumps(prompt, pretty=True)

    with open('../utils/prompt_openai-functions-agent.json', 'w') as fp:
        json.dump(string_representation, fp, indent=4)


def get_tools(url):

    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=100
    ).split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "admissions_search",
        "Search for information about university of kentucky admissions.",
    )

    return [retriever_tool]

if __name__ == '__main__':

    url = "https://admission.uky.edu/freshman/admission-checklist"
    tools = get_tools(url)

    #load saved prompt
    with open("../utils/prompt_openai-functions-agent.json", "r") as fp:
        prompt = loads(json.load(fp))

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)


    '''
    q1 = "hi!"
    r1 = agent_executor.invoke({"input": q1})
    print('q1:', q1, 'r1:', r1)

    q2 = "how can langsmith help with testing?"
    r2 = agent_executor.invoke({"input": q2})
    print('q2:', q2, 'r2:', r2)

    q3 = "whats the weather in lexington, kentucky?"
    r3 = agent_executor.invoke({"input": q3})
    print('q3:', q3, 'r3:', r3)
    '''

    q4 = "What are the university of kentucky admission requirements for an in-state freshman?"
    r4 = agent_executor.invoke({"input": q4})
    print('q4:', q4, 'r4:', r4)






