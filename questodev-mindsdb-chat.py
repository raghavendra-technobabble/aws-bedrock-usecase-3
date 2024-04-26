import streamlit as st
import mindsdb_sample_pinecone_hyde as demo ### replace rag_backend with your backend filename
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.bedrock import BedrockEmbeddings
import boto3

st.title("MindsDB Docs Bot")
#index = demo.loaded_index()
print('sript ran')

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1',
    aws_access_key_id=st.secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=st.secrets.AWS_SECRET_ACCESS_KEY,
)

pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
#index = pinecone.Index(index_name="questo-index")

# def hr_rag_response(index, question):
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
#     chain = ConversationalRetrievalChain.from_llm(llm=hr_llm(), retriever=index.as_retriever(search_kwargs={'k': 5}), memory=memory, verbose=True)
#     response = chain.run({'question': question})
#     return response

data_embeddings = BedrockEmbeddings(
        client = bedrock_client,
        model_id = 'amazon.titan-embed-text-v1')

if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Wait for magic...All beautiful things in life take time :-)"): ###spinner message
        st.session_state.vector_index =  PineconeVectorStore.from_existing_index(index_name='questo-index', embedding=data_embeddings) ### Your Index Function name from Backend File

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("📢Anytime someone tells me that I can't do something, I want to do it more - Taylor Swift"): ### Spinner message
        response_content = demo.hr_rag_response(index=st.session_state.vector_index, question=prompt) ### replace with RAG Function from backend file
        #st.write(response_content)
        #response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    