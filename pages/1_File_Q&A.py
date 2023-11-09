import streamlit as st
from langchain.llms import OpenAI
#import anthropic

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
   
st.title("📝 File Q&A with Anthropic")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md","pdf"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)


if uploaded_file and question and not openai_api_key:
    st.info("Please add your Anthropic API key to continue.")

# if uploaded_file and question and openai_api_key:
#     article = uploaded_file.read().decode()
#     prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
#     {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""

#     client = anthropic.Client(api_key=openai_api_key)
#     response = client.completions.create(
#         prompt=prompt,
#         stop_sequences=[anthropic.HUMAN_PROMPT],
#         model="claude-v1", #"claude-2" for Claude 2 model
#         max_tokens_to_sample=100,
#     )
#     st.write("### Answer")
#     st.write(response.completion)

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


if uploaded_file and question and openai_api_key:
    doc_reader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
    if text:
        raw_text += text
    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = FAISS.from_texts(texts, embeddings)


    # set up FAISS as a generic retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

    # create the chain to answer questions
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)


    st.write("### Answer")
    st.write(rqa(question))