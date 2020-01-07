from utils.io_utils import IOUtils
from utils.string_utils import StringUtils


def create_word_list(input_dir):
    counter = 1
    word_list = []
    import os
    in_files = IOUtils.get_all_file_names_in_dir(input_dir, 'json')
    for in_file in in_files:
        print("Working on: {}/{}".format(counter, len(in_files)))
        in_json = IOUtils.get_json_from_json_file(os.path.join(input_dir, in_file))
        word_list += list(map(StringUtils.get_word_list, in_json))
        counter += 1
    return [item for sublist in word_list for item in sublist]
