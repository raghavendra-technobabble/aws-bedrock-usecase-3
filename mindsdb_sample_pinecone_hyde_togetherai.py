import os
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup as Soup
import csv
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone
import boto3
import streamlit as st
from langchain_together import Together
from langchain_groq import ChatGroq


os.environ['OPENAI_API_KEY'] = 'dummy_key'
os.environ['PINECONE_API_KEY'] = st.secrets.PINECONE_API_KEY

# AWS Initialization
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1',
    aws_access_key_id=st.secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=st.secrets.AWS_SECRET_ACCESS_KEY,
)


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

    #print(len(data_split))

    docs = data_split.split_documents(data_load)

    #print(docs)

    data_embeddings = BedrockEmbeddings(
        client = bedrock_client,
        model_id = 'amazon.titan-embed-text-v1')

    pc = pinecone.Pinecone(api_key=st.secrets.PINECONE_API_KEY)

    index_name = "questo-index"

    if index_name in pc.list_indexes().names():  
        # pc.delete_index(index_name) 
        return 
    else:   
        # create a new index  
        pc.create_index(  
            index_name,  
            dimension=1536,  # dimensionality of text-embedding-ada-002  
            metric='dotproduct',  
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')  
        ) 

    # data_index = VectorstoreIndexCreator(
    #     text_splitter=data_split,
    #     embedding=data_embeddings,
    #     vectorstore_cls=FAISS
    # )

    data_index = PineconeVectorStore.from_documents(docs, embedding=data_embeddings, index_name=index_name)

    # db_index = data_index.from_loaders([data_loader])
    return data_index

def hr_llm_bedrock():
    llm = Bedrock(
        client = bedrock_client,
        model_id = 'anthropic.claude-v2',
        model_kwargs = {
            "max_tokens_to_sample": 3000,
            "temperature": 0.2,
            "top_p": 0.9
        }
    )
    # llm = Together(
    #     model="meta-llama/Llama-3-8b-hf",
    #     temperature=0.5,
    #     max_tokens=1500,
    #     top_p=0.9,
    #     together_api_key="331974719e4159b1fc3a3af46d571fe0295de27929ed79d5f184d3d90c457297"
    # )
    return llm

def hr_llm_groq():
    # llm = Bedrock(
    #     client = bedrock_client,
    #     model_id = 'anthropic.claude-v2',
    #     model_kwargs = {
    #         "max_tokens_to_sample": 3000,
    #         "temperature": 0.2,
    #         "top_p": 0.9
    #     }
    # )
    # llm = Together(
    #     model="mistralai/Mistral-7B-Instruct-v0.1",
    #     temperature=0.5,
    #     max_tokens=1500,
    #     top_k=50,
    #     together_api_key="331974719e4159b1fc3a3af46d571fe0295de27929ed79d5f184d3d90c457297"
    # )
    llm = ChatGroq(
        temperature=0.4, 
        groq_api_key="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", 
        model_name="llama3-8b-8192"
        )
    return llm

# def hr_rag_response(index, question):
#     rag_llm = hr_llm()
#     hr_rag_query = index.query(question=question, llm=rag_llm)
#     return hr_rag_query

# def hr_rag_response(index, question):
#     #rag_llm = hr_llm()
#     qa = RetrievalQA.from_chain_type(  
#     llm=hr_llm(),  
#     chain_type="stuff",  
#     retriever=index.as_retriever() 
#     )
#     response = qa.run(question)
#     return response

def hr_rag_response(index, question):
    # Hyde document generation
    template = """Please write a concise reply to the question
    Question: {query}
    Reply:"""

    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = ( prompt_hyde | hr_llm_groq() | StrOutputParser() )

    print(generate_docs_for_retrieval.invoke({"query": question}))

    retrieval_chain = generate_docs_for_retrieval | index.as_retriever()
    retrieved_docs = retrieval_chain.invoke({"query": question})
    print("\nretrieved docs are \n",retrieved_docs)

    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
    #chain = ConversationalRetrievalChain.from_llm(llm=hr_llm(), retriever=index.as_retriever(search_kwargs={'k': 5}), memory=memory, verbose=True)
    
    final_template = """ Answer the following in a form of passage and code (only if required) related to mindsdb based on this context:
    
    {context}
    
    Question: {query}
    """

    final_prompt = ChatPromptTemplate.from_template(final_template)

    final_rag_chain = (
        final_prompt
        | hr_llm_groq()
        | StrOutputParser()
    )
    response = final_rag_chain.invoke({"context": retrieved_docs, "query": question})
    return response, [doc.metadata['source'] for doc in retrieved_docs]