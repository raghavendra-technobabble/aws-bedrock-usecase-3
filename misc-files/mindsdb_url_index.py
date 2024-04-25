import os
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
from bs4 import BeautifulSoup as Soup
import csv

os.environ['OPENAI_API_KEY'] = 'dummy_key'


data_loader = WebBaseLoader(['https://mindsdb.com/blog/time-series-forecasting-with-nixtla-and-mindsdb-using-mongodb-query-language', ])
#data_loader = RecursiveUrlLoader(url='https://docs.mindsdb.com/', max_depth=3, extractor=lambda x: Soup(x, "html.parser").text)
print(data_loader)

data_load = data_loader.load()
print(data_load)
data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=20)

print(data_split)

data_embeddings = BedrockEmbeddings(
    credentials_profile_name = 'default',
    model_id = 'amazon.titan-embed-text-v1')

data_index = VectorstoreIndexCreator(
    text_splitter=data_split,
    embedding=data_embeddings,
    vectorstore_cls=FAISS
)

db_index = data_index.from_loaders([data_loader])

llm = Bedrock(
    credentials_profile_name = 'default',
    model_id = 'anthropic.claude-v2',
    model_kwargs = {
        "max_tokens_to_sample": 3000,
        "temperature": 0.5,
        "top_p": 0.9
    }
)


def hr_rag_response(index, question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question=question, llm=rag_llm)
    return hr_rag_query

# index = hr_index()
# rag_query1 = hr_rag_response(index, 'create a mindsdb tutorial through project based learning. The project classifies text sentiment. The data is pre loaded in a mongodb database. Use hugging face model for sentiment analysis. Give the full tutorial in detail with 10 steps and code samples of mindsdb.')
# print(rag_query1)
# rag_query2
