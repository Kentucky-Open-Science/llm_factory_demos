#See https://python.langchain.com/v0.1/docs/integrations/retrievers/pubmed/

import json
import ssl

import pandas as pd
from langchain_caai.caai_emb_client import caai_emb_client
from langchain.tools.retriever import create_retriever_tool
from langchain_core.load import loads
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.retrievers import PubMedRetriever


with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

llm = ChatOpenAI(
    model_name="/models/functionary-small-v2.5",
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

def config_ssl():

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context


def get_tools():

    retriever = PubMedRetriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "pubmed_search",
        "Search for information about pubmed articles",
    )

    return [retriever_tool]

if __name__ == '__main__':

    with open("gene_data.json", "r") as fp:
        gene_data = json.load(fp)

    #gene_data = dict()

    df = pd.read_csv('brown_ME.csv')

    config_ssl()

    tools = get_tools()

    # load saved prompt
    with open("../utils/prompt_openai-functions-agent.json", "r") as fp:
        prompt = loads(json.load(fp))

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)


    for index, row in df.iterrows():
        gene_name = row["gene"]

        if gene_name in gene_data:
            print('gene:', gene_name, ' output:', gene_data[gene_name]['output'])
        else:
            try:
                q4 = 'Search PubMed for articles related to the gene ' + gene_name + ' and substance abuse or addiction, format output in JSON'
                r4 = agent_executor.invoke({"input": q4})

                print(r4)

                gene_data[gene_name] = r4

                with open('gene_data.json', 'w') as fp:
                    json.dump(gene_data, fp, indent=4)
            except:
                print('Something went wrong')



