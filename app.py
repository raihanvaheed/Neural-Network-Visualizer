
import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import ssl

URI = 'http://127.0.0.1:5000'

st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input Image')


if st.button('Get random prediction'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28,28))
    
    st.sidebar.image(image, width=150)
    
    
    for layer, p in enumerate(preds):
        
        numbers = np.squeeze(np.array(p)) # p is the prediction value, layer tells which layer in CNN in this cas Layer#2 is output
        
        plt.figure(figsize=(32,4))
        
        if layer == 2:  # Output layer in this case
            row = 1
            col = 10
        else:
            row = 2
            col = 16
            
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow(number * np.ones((8, 8, 3)).astype('float32'))# all 1s would be white boxes, but multiplying by number shows to what classification, 8x8 is number of pixels
            plt.xticks([])
            plt.yticks([])
            
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text('Layer {}'.format(layer + 1))
        st.pyplot()
