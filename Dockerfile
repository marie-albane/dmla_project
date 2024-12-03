FROM python:3.10.6-buster
COPY dmla /dmla
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
CMD uvicorn dmla.api.fast:app --host 0.0.0.0 --port 8000
