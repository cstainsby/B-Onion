import os

scripts_folder_path = os.path.dirname(os.path.realpath(__file__))
backend_folder_path = scripts_folder_path + "/.."
credential_path = backend_folder_path + "/creds"

pathing_info = {
  "backend_path": backend_folder_path,
  "scripts_path": scripts_folder_path,
  "credential_path": credential_path
}