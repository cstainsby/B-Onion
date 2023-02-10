
import os
from datetime import datetime, date, timedelta  

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

  # for n in range(int ((END_DATE - START_DATE).days / 7) + 1):
  #   curr_date = START_DATE + timedelta(n * 7)
  #   out_file.write("## week of: " + str(curr_date.month) + "/" + str(curr_date.day) + "\n\n")

  return out_file

"""The README will have headers denoting what week every post was made under
    This function will grab all of them from the readme and put them in a list"""
def get_weekly_headers(readme):
  weekly_headers = []

  lines = readme.readlines()

  for line in lines:
    if len(line) > 4 and line[0] == '#' and line[1] == '#': # doulbe hash is reserved for line
      weekly_headers.append(line[3:])

  return weekly_headers

def is_week_header_in_readme(readme):
  weekly_headers = get_weekly_headers(readme)

  curr_day_of_the_week = datetime.now().weekday()
  current_date = date.today()

  day_of_last_monday = current_date.day - timedelta(curr_day_of_the_week)


  

def log_post_to_readme():
  curr_day_of_the_week = datetime.now().weekday()
  current_date = date.today()




def main():
  readme = open_readme()

  readme.close()





if __name__ == "__main__":
  main()