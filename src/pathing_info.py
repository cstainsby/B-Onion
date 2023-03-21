import os

backend_folder_path = os.path.dirname(os.path.realpath(__file__))
scripts_folder_path = backend_folder_path + "/scripts"
credential_path = backend_folder_path + "/creds"
model_path = backend_folder_path + "/models"

pathing_info = {
  "backend_path": backend_folder_path,
  "scripts_path": scripts_folder_path,
  "credential_path": credential_path
}