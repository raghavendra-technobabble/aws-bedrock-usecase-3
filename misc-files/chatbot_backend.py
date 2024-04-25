import os
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def demo_chatbot():
    demo_llm = Bedrock(
        credentials_profile_name = 'default',
        model_id = 'meta.llama2-70b-chat-v1',
        model_kwargs = {
            "temperature": 0.9,
            "top_p": 0.5,
            "max_gen_len": 50
        })
    #return demo_llm
    return demo_llm

# def demo_memory():
#     llm_data = demo_chatbot()
#     #memory = ConversationBufferMemory(llm = llm_data, max_token_limit = 150)
#     print(memory)
#     return memory

# memory = ConversationBufferMemory(llm = demo_chatbot(), max_token_limit = 150)
# print("memory is", memory.buffer)

def demo_conversation(input_text):
    llm_chain_data = demo_chatbot()
    #memory = demo_memory()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory = ConversationBufferMemory(), verbose = True)
    chat_reply = llm_conversation.predict(input=input_text)
    print("memory is", llm_conversation.memory.buffer)
    print("\n")
    return chat_reply

reply = demo_conversation('this is step 9 related to intro - checking memory')
print("reply is", reply)

