import sys

def convert_file(path_to_file: str, path_to_save: str):
    with open(path_to_file, "r") as orig_file:
        with open(path_to_save, "w") as new_file:
            while (line := orig_file.readline()):
                elements = line.strip().split(" ")
                # new_line = " ".join(elements[0:len(elements)-1])
                new_line = " ".join(elements[0:2]) + " "
                new_line += " ".join(elements[3:])
                # new_line = " ".join(elements)
                new_file.write(new_line + "\n")
    

if __name__ == "__main__":
    path_to_file, path_to_save = sys.argv[1], sys.argv[2]
    convert_file(path_to_file, path_to_save)