import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Ocultar advertencias de deprecación
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de Productos",
    page_icon="icono.png",
    initial_sidebar_state='auto'
)

# Estilo personalizado para ocultar el menú y el pie de página
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./modeloIA.h5')
    return model

with st.spinner('Modelo está cargando...'):
    model = load_model()

with st.sidebar:
    st.title("Reconocimiento de imagen")
    st.image('foto.jpg')
    st.subheader("Reconocimiento de imagen para Productos")
    confianza = st.slider("Nivel de confianza mínimo", min_value=0, max_value=100, value=50)
    st.write("""SnapFind utiliza la inteligencia artificial para ofrecer una experiencia de reconocimiento de productos rápida, precisa y eficiente. A continuación, te explicamos cómo funciona:

Procesamiento de Imágenes:
Cuando tomas una foto de un producto, SnapFind emplea técnicas avanzadas de procesamiento de imágenes para mejorar y preparar la imagen para el análisis. Esto incluye la corrección de iluminación, enfoque y eliminación de ruido, asegurando que la imagen esté en las mejores condiciones para el reconocimiento.

Redes Neuronales Convolucionales (CNN):
En el núcleo de nuestro sistema de reconocimiento de imágenes, utilizamos Redes Neuronales Convolucionales (CNN). Estas redes están especialmente diseñadas para procesar y analizar datos visuales. Las CNN escanean la imagen en busca de patrones y características específicas que corresponden a productos conocidos.

Base de Datos Extensa y Entrenamiento:
SnapFind está entrenado con una vasta base de datos de imágenes de productos etiquetadas. Durante el entrenamiento, la IA aprende a identificar distintos productos a partir de miles de ejemplos, mejorando su precisión con el tiempo. Este proceso, conocido como aprendizaje profundo, permite que SnapFind reconozca una amplia variedad de artículos, incluso aquellos que no ha visto antes.

Algoritmos de Clasificación y Recuperación:
Una vez que la CNN ha procesado la imagen, los algoritmos de clasificación entran en acción para determinar cuál es el producto en la foto. Luego, los algoritmos de recuperación buscan en nuestra base de datos para encontrar información relevante sobre el producto, incluyendo descripciones, especificaciones y opciones de compra.

Mejora Continua:
SnapFind no se detiene en el momento del lanzamiento. Continuamente recopilamos datos y retroalimentación de los usuarios para entrenar y mejorar nuestros modelos de IA. Esto significa que SnapFind se vuelve más inteligente y preciso con cada uso.

Con SnapFind, la inteligencia artificial transforma una simple foto en una puerta de acceso a un mundo de información y oportunidades de compra, haciendo que el reconocimiento de productos sea más sencillo y accesible que nunca.""")

st.title("SnapFind")
st.header("Bienvenido a SnapFind.")
st.write("""La solución innovadora de inteligencia artificial diseñada para transformar la manera en que reconoces y encuentras productos. Con SnapFind, simplemente toma una foto de cualquier 
         artículo y nuestra avanzada tecnología de reconocimiento de imágenes identificará instantáneamente el 
         producto, brindándote información detallada y opciones de compra al instante. Ya sea que estés explorando 
         una tienda física, navegando por catálogos de productos, o descubriendo nuevos artículos en línea, 
         SnapFind es tu aliado perfecto para hacer que el proceso de identificación y adquisición de productos sea rápido, 
         preciso y sin esfuerzo. Nuestra herramienta está impulsada por algoritmos de última generación y una base de datos extensa, 
         garantizando resultados confiables y relevantes en cuestión de segundos.""")
st.image('logo.png')
st.write("""
         # Detección de Productos
         """)

def import_and_predict(image_data, model, class_names):
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Crear un batch

    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index].strip()
    
    return class_name, score

# Abrir el archivo con la codificación adecuada
with open("clases.txt", "r", encoding="utf-8") as f:
    class_names = f.readlines()

# Opciones para ingresar la imagen
option = st.selectbox(
    'Selecciona cómo deseas subir la imagen:',
    ('Subir una foto', 'Tomar una foto', 'Ingresar URL de una foto')
)

image = None

if option == 'Tomar una foto':
    img_file_buffer = st.camera_input("Capture una foto para identificar un Producto")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

elif option == 'Subir una foto':
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == 'Ingresar URL de una foto':
    url = st.text_input("Ingresa el URL de la imagen")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("No se pudo cargar la imagen desde el URL proporcionado.")

# Procesar la imagen y mostrar resultados
if image is not None:
    st.image(image, use_column_width=True)
    
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado
    if np.max(score) > confianza / 100:
        st.subheader(f"Producto: {class_name}")
        st.text(f"Puntuación de confianza: {np.max(score) * 100:.2f}%")
    else:
        st.text(f"No se pudo determinar el producto con suficiente confianza (mínimo {confianza}%)")
else:
    st.text("Por favor proporciona una imagen")
