

def ensure_dir_exists(file_path):
  """ create the containing folder for the absolute path to a file `file_path` """
  import os
  dirname = os.path.dirname(file_path)
  if not os.path.exists(dirname):
    os.mkdir(dirname)
