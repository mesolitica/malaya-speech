FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers soundfile librosa
RUN pip3 install torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install six webrtcvad
RUN pip3 install ctranslate2
RUN pip3 install git+https://github.com/mesolitica/malaya-speech@6aea111d8f17ceeaf53e5f95de8887610a93577c

COPY ./app /app

ENV PORT=9091
ENTRYPOINT ["/start-reload.sh"]