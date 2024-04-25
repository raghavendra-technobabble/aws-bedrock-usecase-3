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
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA  
import pinecone

os.environ['OPENAI_API_KEY'] = 'dummy_key'
os.environ['PINECONE_API_KEY'] = '05657a5e-950f-4f74-a9ac-0b54785546c6'


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
    data_load = data_loader.load()
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200)
    docs = data_split.split_documents(data_load)
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name = 'default',
        model_id = 'amazon.titan-embed-text-v1')
    pc = pinecone.Pinecone(api_key='05657a5e-950f-4f74-a9ac-0b54785546c6')
    index_name = "questo-index"
    if index_name in pc.list_indexes().names(): 
        data_index = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=data_embeddings)
        return data_index
    else:   
        # create a new index  
        pc.create_index(  
            index_name,  
            dimension=1536,  # dimensionality of text-embedding-ada-002  
            metric='dotproduct',  
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')  
        ) 
        data_index = PineconeVectorStore.from_documents(docs, embedding=data_embeddings, index_name=index_name)
        return data_index

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
    #rag_llm = hr_llm()
    qa = RetrievalQA.from_chain_type(  
    llm=hr_llm(),  
    chain_type="stuff",  
    retriever=index.as_retriever() 
    )
    response = qa.run(question)
    return response

urls = read_urls_from_csv('urls-mindsdb-unique.csv')
index = hr_index(urls[1:])
query = 'create an ml engine in mindsdb'

qa = RetrievalQA.from_chain_type(  
    llm=hr_llm(),  
    chain_type="stuff",  
    retriever=index.as_retriever()  
)

response = qa.run(query)
print(response)