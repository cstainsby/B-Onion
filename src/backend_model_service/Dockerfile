FROM python:3.8-slim-buster

# create app folder and copy in code 
WORKDIR /app

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# install requirements from requirements.txt
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 

COPY . /app 

ENV FLASK_RUN_PORT=8000
EXPOSE $FLASK_RUN_PORT

# entrypoint command - tells dockerfile to execute backend flask app
# binded to port 8000
CMD gunicorn --bind $FLASK_RUN_HOST:$FLASK_RUN_PORT --timeout=150 app:app