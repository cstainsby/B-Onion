FROM python:3.8-slim-buster

# create app folder and copy in code 
WORKDIR /app

ENV openai_access_token=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
ENV praw_password=wkd44p9fxKgepVg
ENV praw_client_secret=j62C4wlEBYFRoOWeMWWL1qghzjjlwg

# install requirements from requirements.txt
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 

COPY . /app 

EXPOSE 3000

# entrypoint command - tells dockerfile to execute backend flask app
# binded to port 8000
CMD gunicorn --bind 0.0.0.0:3000 --timeout=150 app:app