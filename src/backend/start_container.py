import os

PROJECT_ID = "bonion"
REGION="us-west-2"
REPOSITORY=""
IMAGE="bonion-image"

image_tag = "{region}-docker.pkg.dev/{project_id}/{repo}/{image_name} .".format(
              region=REGION,
              project_id=PROJECT_ID,
              repo=REPOSITORY,
              image_name=IMAGE
            )

def login():
  # login to docker 
  os.system("docker login")

# build container
def build():
  os.system(
      "docker build -t {image_tag}".format(image_tag=image_tag)
    )
  
def run_local():
  # NOTE: I dont have a gpu so I will not enable the option
  os.system(
    "docker run -p 8000:8000 --name=bonion_container {image_tag}".format(image_tag=image_tag)
  )

if __name__ == "__main__":
  login()
  build()
  run_local()