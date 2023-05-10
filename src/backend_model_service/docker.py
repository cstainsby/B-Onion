import os

HOSTNAME = "gcr.io"
PROJECT_ID = "bonion"
TARG_IMG = "backend_model_service"
TAG = "latest-demo"


def build_container():
  CMD = """
    docker build -t {}/{}/{}:{} .
  """.format(HOSTNAME, PROJECT_ID, TARG_IMG, TAG)
  print("RUNNING:", CMD)
  os.system(CMD)

# def run_local_container():
#   CMD = """
#     docker run -v "$HOME/.config/gcloud:/gcp/config:ro" \
#   -v /gcp/config/logs \
#   -p 3000:3000 \
#   --env  \
#   --env \ 
#    {}/{}/{}:{}
#   """.format(HOSTNAME, PROJECT_ID, TARG_IMG, TAG)
#   print("RUNNING:", CMD)
#   os.system(CMD)

# def run_local_container():
#   CMD = """
#     docker run -p 3000:3000 {}/{}/{}:{}
#   """.format(HOSTNAME, PROJECT_ID, TARG_IMG, TAG)
#   print("RUNNING:", CMD)
#   os.system(CMD)

def push_container():
  CMD = """
    docker push {}/{}/{}:{}
  """.format(HOSTNAME, PROJECT_ID, TARG_IMG, TAG)
  print("RUNNING:", CMD)
  os.system(CMD)

if __name__=="__main__":
  # login()
  build_container()
  # run_local_container()
  push_container()