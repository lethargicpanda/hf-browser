FROM python:3.11-slim
RUN useradd -m -u 1000 user
RUN pip install --no-cache-dir torch==2.8.0 transformers==4.57.1 safetensors==0.6.2
USER user
WORKDIR /home/user/app
COPY --chown=user:user model.json server.py ./
EXPOSE 7860
CMD ["python", "server.py"]
