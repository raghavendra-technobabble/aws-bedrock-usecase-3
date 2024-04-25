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

# Function to read URLs from a CSV file
def read_urls_from_csv(filename):
    urls = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            urls.extend(row)
    return urls

def hr_index(urls):
    data_loader = WebBaseLoader(urls)
    #data_loader = RecursiveUrlLoader(url='https://docs.mindsdb.com/', max_depth=3, extractor=lambda x: Soup(x, "html.parser").text)
    #print(data_loader)

    data_load = data_loader.load()
    #print(data_load)
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200)

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
    return db_index

def hr_llm():
    llm = Bedrock(
        credentials_profile_name = 'default',
        model_id = 'anthropic.claude-v2',
        model_kwargs = {
            "max_tokens_to_sample": 3000,
            "temperature": 0.5,
            "top_p": 0.9
        }
    )
    return llm

def hr_rag_response(index, question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question=question, llm=rag_llm)
    return hr_rag_query

urls = read_urls_from_csv('urls-mindsdb-unique.csv')
#print(urls)
index = hr_index(urls[1:10])
rag_query1 = hr_rag_response(index, 'create a mindsdb tutorial through project based learning. The project classifies text sentiment. The data is pre loaded in a mongodb database. Use hugging face model for sentiment analysis. Give the full tutorial in detail with 10 steps and code samples of mindsdb.')
print(rag_query1)
# rag_query2
