# The below frontend code is provided by AWS and Streamlit. I have only modified it to make it look attractive.
import streamlit as st 
import mindsdb_sample_pinecone as demo ### replace rag_backend with your backend filename
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import BedrockEmbeddings

#index = demo.loaded_index()
print('sript ran')

pc = Pinecone(api_key="05657a5e-950f-4f74-a9ac-0b54785546c6")
#pinecone_index = pinecone.Index(index_name="questo-index")

data_embeddings = BedrockEmbeddings(
        credentials_profile_name = 'default',
        model_id = 'amazon.titan-embed-text-v1')

st.set_page_config(page_title="HR Q and A with RAG") ### Modify Heading

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">HR Q & A with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True) ### Modify Title

if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Wait for magic...All beautiful things in life take time :-)"): ###spinner message
        st.session_state.vector_index =  PineconeVectorStore.from_existing_index(index_name='questo-index', embedding=data_embeddings) ### Your Index Function name from Backend File

input_text = st.text_area("Input text", label_visibility="collapsed") 
go_button = st.button("📌Learn GenAI with Rahul Trisal", type="primary") ### Button Name

if go_button: 
    
    with st.spinner("📢Anytime someone tells me that I can't do something, I want to do it more - Taylor Swift"): ### Spinner message
        response_content = demo.hr_rag_response(index=st.session_state.vector_index, question=input_text) ### replace with RAG Function from backend file
        st.write(response_content)