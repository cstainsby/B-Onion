import os

from backend.pathing_info import pathing_info


def run_api_only():
  backend_flask_file = pathing_info["backend_path"] + "/backend.py"

  os.system("""
    python3 {}
  """.format(backend_flask_file))

if __name__ == "__main__":
    run_api_only()