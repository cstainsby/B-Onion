# because 

FROM python:3.8-slim-buster

# create app folder and copy in code 
WORKDIR /app

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

ENV praw_client_id=pL92jPPhj6EK2yCx-AhHfw
ENV praw_client_secret=j62C4wlEBYFRoOWeMWWL1qghzjjlwg
ENV praw_username=B-0nion
ENV praw_user_agent=bonion

# install requirements from requirements.txt
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 

COPY . /app 

EXPOSE 8000

# entrypoint command - tells dockerfile to execute backend flask app
# binded to port 8000
CMD gunicorn --bind 0.0.0.0:8000 --timeout=150 app:app