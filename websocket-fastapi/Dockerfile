FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers soundfile librosa
RUN pip3 install git+https://github.com/mesolitica/malaya-speech@malaysian-speech

COPY ./app /app