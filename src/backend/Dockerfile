FROM continuumio/anaconda3:2021.11

# create app folder and copy in code 
WORKDIR /app
COPY . /app 

# install requirements from requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt 

# entrypoint command - tells dockerfile to execute backend flask app
# binded to port 8000
CMD gunicorn --bind 0.0.0.0:8000 --timeout=150 backend:backend -w 5

EXPOSE 8000