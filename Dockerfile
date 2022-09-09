FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV PORT=5000
EXPOSE $PORT
COPY . .
CMD ["python" , "-u","server.py"]



