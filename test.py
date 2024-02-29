import boto3
import json
from langchain import PromptTemplate
import os
from skimage import io

import chatbot_llama2_13b as chatbot_llama2_13b
import chatbot_llama2_70b as chatbot_llama2_70b

bedrock_client = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
)

def imageAnalyzer(input_img):
    rek_client = boto3.client('rekognition')

    print(input_img)
    with open(input_img, 'rb') as image:
        response = rek_client.detect_labels(Image={'Bytes': image.read()})

    labels = response['Labels']
    label_names = ''
    for label in labels:
        name = label['Name']
        confidence = label['Confidence']
        if confidence>95:
            label_names = label_names + name + ","
    return label_names

input_img = 'people.jpg'
img = io.imread(input_img)

labels = imageAnalyzer(input_img)

prompt_claude = """ Human:  Here are the comma seperated list of labels/objects seen in the image: <labels>""" + labels + """</labels> Please provide a human readible and Understandable summary based on these labels Assistant:"""

response = chatbot_llama2_13b.chatbot__llama2_13b_quickcall(prompt_claude)

print(response)