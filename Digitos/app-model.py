import os
st.write("Arquivos no diretório:", os.listdir())

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import torch.nn as nn

import torch
st.set_page_config(layout="centered", page_title="Desenho de Dígitos MNIST")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1 , 10 , kernel_size=5)
        self.conv2 = nn.Conv2d(10 , 20 ,kernel_size=5)
        self.Conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320 , 50)
        self.fc2 = nn.Linear(50,10)

    def forward(self , x):
        x = F.relu(F.max_pool2d(self.conv1(x) , 2))
        x = F.relu(F.max_pool2d(self.Conv2_drop(self.conv2(x)) , 2))
        x = x.view(-1 ,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x , training=self.training)
        x = self.fc2(x)
        
        return x # Retorna os logits brutos



@st.cache_resource 
def load_model():
    try:
        model = CNN()

        model_dict = torch.load("modelo_mnist.pth", map_location=torch.device("cpu"))

        model.load_state_dict(model_dict)
        model.eval()
        return model


    except Exception as e:
        st.error(f"Erro ao carregar o modelo PyTorch completo. Verifique o caminho ou a definição da classe: {e}")

        return None 
    

model = load_model()


####### interface #########

st.title("🧠🔢 Classificando Dígitos com CNN (MNIST)")

st.info("Use o mouse para desenhar um dígito (0-9) na caixa abaixo.")


# canvas onde o usuário pode desenhar
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # Cor de preenchimento (fundo branco)
    stroke_width=15,                 # Largura do traço 
    stroke_color="rgb(0, 0, 0)",     # Cor do traço (preto)
    background_color="rgb(255, 255, 255)", # Cor de fundo do canvas (branco)
    height=200,                      # Altura do canvas em pixels
    width=200,                       # Largura do canvas em pixels
    drawing_mode="freedraw",         # Modo de desenho livre
    key="canvas_mnist",              # Uma chave única para o componente
)

st.markdown("---")

if canvas_result.image_data is not None:
    drawn_image_rgba = Image.fromarray(canvas_result.image_data)
    drawn_image_gray = drawn_image_rgba.convert("L")


    st.subheader("Dígito Pré-processado para o Modelo MNIST:")

    img_array = np.array(drawn_image_gray)
    resized_img = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
    inverted_img = cv2.bitwise_not(resized_img)
    normalized_img = inverted_img / 255.0

    st.image(normalized_img, caption=" Seu Dígito Desenhado e Pré-processado (28x28)", width=300)
    st.write("Formato da imagem para o modelo:", normalized_img.shape)

    if st.button("Classificar Dígito 🔮"):
        if model is not None:
            with st.spinner("Classificando o dígito..."):
                try:
                    model_input = torch.tensor(normalized_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    model = model.to(device)
                    model_input = model_input.to(device)

                    with torch.no_grad():
                        output = model(model_input)
                        predicted_digit = torch.argmax(output, dim=1).item()

                    

                    st.success(f"""Previsão Concluída! \n
                              O DIGITO PREVISTO É : {predicted_digit}""")

            

                except Exception as e:
                    st.error(f"Erro na classificação: {e}")
        else:
            st.warning("Modelo não carregado.")
else:
    st.info("Desenhe um dígito para classificar.")
