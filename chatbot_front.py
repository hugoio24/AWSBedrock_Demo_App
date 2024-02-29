import streamlit as st

import chatbot_llama2_13b as chatbot_llama2_13b
import chatbot_llama2_70b as chatbot_llama2_70b
import chatbot_sd as chatbot_sd

import streamlit as st
import numpy as np
from PIL import Image
from skimage import io

import test as test

st.title("🚀 Hugo's Bedrock Chatbot Demonstrator")

def clear_chat_history():
    st.session_state.chat_history = []

st.sidebar.markdown("## Selectionnez votre modèle et vos données")
model = st.sidebar.selectbox("Modèle", ["Chat - Llama2 13B", "Chat - Llama2 70B", "Chat - Claude (wip)", "Image - Stable Diffusion", "Analyse image"])

st.sidebar.button('Clear chat',on_click=clear_chat_history)

if model == "Chat - Llama2 13B":
    st.caption("🍄 based on Bedrock Llama2 13B")

    if 'memory' not in st.session_state:
        st.session_state.memory = chatbot_llama2_13b.chatbot_llama2_13b_memory()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    input_text = st.chat_input("Whats up ?")

    if input_text:
        with st.chat_message("user"):
            st.markdown(input_text)
        
        st.session_state.chat_history.append({"role":"user","text":input_text})

        chat_response = chatbot_llama2_13b.chatbot_llama2_13b_conversation(input_text=input_text,memory=st.session_state.memory)

        with st.chat_message("assistant"):
            st.markdown(chat_response["response"])

        st.session_state.chat_history.append({"role":"assistant","text":chat_response["response"]})

if model == "Chat - Llama2 70B":
    st.caption("🍄 based on Bedrock Llama2 70B")
    if 'memory' not in st.session_state:
        st.session_state.memory = chatbot_llama2_70b.chatbot_llama2_70b_memory()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    input_text = st.chat_input("Whats up ?")

    if input_text:
        with st.chat_message("user"):
            st.markdown(input_text)
        
        st.session_state.chat_history.append({"role":"user","text":input_text})

        chat_response = chatbot_llama2_70b.chatbot_llama2_70b_conversation(input_text=input_text,memory=st.session_state.memory)

        with st.chat_message("assistant"):
            st.markdown(chat_response["response"])

        st.session_state.chat_history.append({"role":"assistant","text":chat_response["response"]})

if model == "Image - Stable Diffusion":
    st.caption("🍄 based on Bedrock Stable Diffusion")

    sd_presets = [
        "None",
        "3d-model",
        "analog-film",
        "anime",
        "cinematic",
        "comic-book",
        "digital-art",
        "enhance",
        "fantasy-art",
        "isometric",
        "line-art",
        "low-poly",
        "modeling-compound",
        "neon-punk",
        "origami",
        "photographic",
        "pixel-art",
        "tile-texture",
    ]

    style = st.sidebar.selectbox("Select Style", sd_presets)

    if prompt := st.chat_input("Whats up ?"):
        st.chat_message("user").write(prompt)
        response = chatbot_sd.base64_to_pil(chatbot_sd.generate_image(prompt,style))
        st.chat_message("assistant").image(response)


if model == "Analyse image":
    img_file_buffer = st.file_uploader('Upload a JPG image', type='jpg')
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        img = io.imread(image)

        labels = test.imageAnalyzer(img)

        prompt_claude = """ Human:  Here are the comma seperated list of labels/objects seen in the image: <labels>""" + labels + """</labels> Please provide a human readible and Understandable summary based on these labels Assistant:"""

        response = chatbot_llama2_13b.chatbot__llama2_13b_quickcall(prompt_claude)

        print(response)