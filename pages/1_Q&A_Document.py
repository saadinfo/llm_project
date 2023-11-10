import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from tempfile import NamedTemporaryFile
import os

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
   
st.title("📝 File Q&A 2")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md","pdf"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)


if uploaded_file and question and not openai_api_key:
    st.info("Please add your Anthropic API key to continue.")



if uploaded_file and question and openai_api_key:
   
   bytes_data = uploaded_file.read()
   with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
    tmp.write(bytes_data)                      # write data from the uploaded file into it
    documents = PyPDFLoader(tmp.name).load()        # <---- now it works!
   os.remove(tmp.name)                            # remove temp file


   #documents = loader.load()
   
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(documents)
   
   embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
   docsearch = Chroma.from_documents(texts, embeddings)

   qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff", retriever=docsearch.as_retriever())


   st.write("### Answer")
   st.write(qa.run(question))