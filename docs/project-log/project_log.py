
import os
from datetime import date, timedelta  

"""opens readme, creates one if it doesn't exist"""
def open_readme():
  file = None

  project_log_dir = os.path.dirname(os.path.realpath(__file__))

  file_exist = os.path.exists(project_log_dir + "/README.md")

  if not file_exist:
    print("File doesn't exist: creating it now")
    file = create_project_log(project_log_dir)
  else:
    file = open("README.md", "r")

  return file

"""Helper function for open readme"""
def create_project_log(path_to_log: str):
  START_DATE = date(2023, 2, 6)
  END_DATE = date(2023, 5, 8)

  out_file = open(path_to_log + "/README.md", "w")
  out_file.write("# Project Log \n")

  for n in range(int ((END_DATE - START_DATE).days / 7) + 1):
    yield START_DATE + timedelta(n * 7)

  return out_file

def main():
  readme = open_readme()





if __name__ == "__main__":
  main()