FROM python:3.8-slim-buster

EXPOSE 8501


RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]