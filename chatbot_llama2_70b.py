import os
import boto3
import json
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def chatbot_llama2_70b():
    demo_llm = Bedrock(
        credentials_profile_name='default',
        model_id="meta.llama2-70b-chat-v1",
        model_kwargs={
            "temperature":0.5,
            "top_p":0.9,
            "max_gen_len":2048
        }
    )
    return demo_llm

def chatbot_llama2_70b_memory():
    llm_data = chatbot_llama2_70b()
    memory = ConversationBufferMemory(llm=llm_data,max_token_limit=2048)
    return memory

def chatbot_llama2_70b_conversation(input_text,memory):
    llm_chain_data = chatbot_llama2_70b()
    llm_conversation = ConversationChain(llm=llm_chain_data,memory=memory,verbose=True)
    chat_reply = llm_conversation.invoke(input_text)
    return chat_reply