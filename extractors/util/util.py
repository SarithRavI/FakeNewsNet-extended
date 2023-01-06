import os 
import errno

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def is_folder_exists(folder_name):
    return os.path.exists(folder_name)