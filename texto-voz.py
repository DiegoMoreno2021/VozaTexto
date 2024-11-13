
#Importamos las librerias necesarias
import os
import streamlit as st
import whisper
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langdetect import detect  # Librería para detectar el idioma

# Para trabajar con OpenAi debemos tener una ApiKey, la cual almacenamos en el equipo
#aqui le indicamos donde esta almacenada
with open('C:/ProyectoChatOpenAI/Scripts/Api_key.txt') as f:
    api_key = f.read().strip()

# Procedemos ahora a configurar la clave de API de OpenAI en LangChain
os.environ["OPENAI_API_KEY"] = api_key

# Aqui cargamos el modelo pre-entrenado de Whisper para la transcripción
model = whisper.load_model("base")

# Configuración de la aplicación en Streamlit, para todo el frontend de la aplicación
st.title("Aplicación de Transcripción de Audio y Análisis de Lenguaje")
st.write("Esta aplicación transcribe audio y genera un texto usando LangChain.")

# Desde la aplicación solicitamos la subida de archivo de audio al usuario
audio_file = st.file_uploader("Sube un archivo de audio para transcribir", type=["mp3", "wav"])

if audio_file:
    # Primero vamos a guardar el archivo de audio temporalmente
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.getbuffer())

    # Ahora ya vamos a transcribir el audio usando Whisper
    result = model.transcribe("temp_audio.mp3")
    transcription_text = result["text"]
    st.write("Transcripción del Audio (antes de traducir, si es necesario):")
    st.write(transcription_text)

    # En estos pasos debido a nuestra ubicacion vamos a validar que el idiona sea español, primero vamos a detectar el idioma del texto transcrito.
    detected_language = detect(transcription_text)

    # Despues de hacer la validación si el texto no está en español, debe proceder a traducirlo e informar al usuario
    if detected_language != 'es':
        st.write("El texto no está en español, se procederá a traducirlo.")
    # Aqui se procede a solicitar traducción al español   
        prompt_template_translation = PromptTemplate(
            input_variables=["text"],
            template="Traduce el siguiente texto al español: {text}"  
        )

    # Aqui configuramos el modelo de lenguaje con OpenAI
        llm_translation = OpenAI(
            model_name="gpt-3.5-turbo",  # Usamos este modelo
            temperature=0.5,             #Dejamos una temperatrura baja para que no alucine
            openai_api_key=api_key  # Aqui llamamos la api key
        )
        chain_translation = LLMChain(llm=llm_translation, prompt=prompt_template_translation)

        # Si es necesario trabduce el texto a español e informa al usuario
        translated_text = chain_translation.run(text=transcription_text)
        
        st.write("Texto traducido al español:")
        st.write(translated_text)

        # Aqui le indicamos que actualize la transcripción con el texto traducido e informe al usuario
        transcription_text = translated_text
    else:
        st.write("El texto ya está en español.")

    # Para complementar el proyecto vamos a crear un modelo de lenguaje para el análisis ya en español
    prompt_template_summary = PromptTemplate(
        input_variables=["text"],
        template="Haz un resumen del siguiente texto en español: {text}"  # Aqui validamos que el resumen esté en español
    )

    # Configuramos el modelo de lenguaje para el resumen
    llm_summary = OpenAI(
        model_name="gpt-3.5-turbo",  # Nuevamente definimos el modelo igual al anterior.
        temperature=0.5, #Dejamos una temperatura cercana a cero para que no alucine
        openai_api_key=api_key #Cargamos nuesta api key
    )
    chain_summary = LLMChain(llm=llm_summary, prompt=prompt_template_summary)

    # Finalmente se genera el texto y se entrega en pantalla al usuario
    summary = chain_summary.run(text=transcription_text)
    st.write("Resumen Generado:")
    st.write(summary)



