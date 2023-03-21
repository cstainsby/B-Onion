FROM python:3.8-slim-buster

# create app folder and copy in code 
WORKDIR /app
COPY . /app 

# install requirements from requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt 


EXPOSE 8000

# entrypoint command - tells dockerfile to execute backend flask app
# binded to port 8000
CMD gunicorn --bind 0.0.0.0:8000 --timeout=150 app:app