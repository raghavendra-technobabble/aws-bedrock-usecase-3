import os
from langchain.llms.bedrock import Bedrock
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class Chatbot:

    def __init__(self):
        self.llm = Bedrock(
        credentials_profile_name = 'default',
        model_id = 'meta.llama2-70b-chat-v1',
        model_kwargs = {
            "temperature": 0.1,
            "top_p": 0.5,
            "max_gen_len": 50
        })
        self.memory = ConversationBufferMemory(max_token_limit = 150)
        self.chain = ConversationChain(llm=self.llm, memory = self.memory, verbose = False)

demo_bot = Chatbot()

reply = demo_bot.chain.predict(input = 'Hey, How are you?')

print(demo_bot.chain.prompt.template)

print("reply is", reply)
print("\n")
print("memory is ", demo_bot.chain.memory.buffer)
print("\n")
print("memory object is", demo_bot.memory)
print("\n")

reply2 = demo_bot.chain.predict(input = 'Good that you are doing good')
print("reply is", reply2)
print("\n")
print("memory is ", demo_bot.chain.memory.buffer)
print("\n")
print("memory object is", demo_bot.memory)
print("\n")

reply3 = demo_bot.chain.predict(input = 'What else?')
print("reply is", reply3)
print("\n")
print("memory is ", demo_bot.chain.memory.buffer)
print("\n")
print("memory object is", demo_bot.memory)
print("\n")

