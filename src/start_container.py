import os
import csv 

current_dir_path = os.path.dirname(os.path.realpath(__file__))

PROJECT_ID = "bonion"
REGION="us-west-2"
REPOSITORY="my-repo"
IMAGE="bonion-image"

image_tag = "{region}-docker.pkg.dev/{project_id}/{repo}/{image_name} .".format(
              region=REGION,
              project_id=PROJECT_ID,
              repo=REPOSITORY,
              image_name=IMAGE
            )

def read_cred_file():
  credential_file_name = "docker_creds.csv"
  file_path = current_dir_path + "/" + credential_file_name
  username, password = "", ""

  print("file path: " + file_path)
  file_exist = os.path.exists(file_path)

  if file_exist:
    csv_contents = []

    with open(file_path, "r") as cred_file:
      csv_reader = csv.reader(cred_file)
      
      for row in csv_reader:
        csv_contents.append(row)
      print(csv_contents)
      username = csv_contents[1][0]
      password = csv_contents[1][1]
  else:
    print("Error: no credential file built")
    exit()
  
  return username, password


def login():
  username, password = read_cred_file()

  # login to docker 
  os.system("docker login -u " + username + " -p " + password)

# build container
def build():
  os.system(
      "docker build -tag={image_tag} .".format(image_tag=image_tag)
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