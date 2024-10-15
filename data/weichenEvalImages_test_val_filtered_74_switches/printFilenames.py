import os

def print_filenames_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print("Der angegebene Pfad ist kein Ordner.")
        return
    
    filenames = os.listdir(folder_path)
    if not filenames:
        print("Der Ordner ist leer.")
        return
    
    filenames = sorted(filenames)
    
    print("Dateien im Ordner:")
    for filename in filenames:
        name, extension = os.path.splitext(filename)
        if extension.lower() == ".jpg":
            print(name)

if __name__ == "__main__":
    folder_path = input("Geben Sie den Pfad zum Ordner ein: ")
    print_filenames_in_folder(folder_path)
