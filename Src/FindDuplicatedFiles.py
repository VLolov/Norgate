# comes from: https://blog.stackademic.com/20-python-scripts-that-will-give-you-superpowers-4c6f4b15fe63
import os
import hashlib

def hash_file(filename):
    h = hashlib.md5()
    with open(filename, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)
    return h.hexdigest()

def find_duplicates(folder):
    hashes = {}
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            file_hash = hash_file(full_path)
            if file_hash in hashes:
                print(f"Duplicate found: {full_path} == {hashes[file_hash]}")
            else:
                hashes[file_hash] = full_path

find_duplicates('Z:/Misc/Archive/trading/e-books')