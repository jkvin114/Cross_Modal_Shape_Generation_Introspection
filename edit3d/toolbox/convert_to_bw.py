from PIL import Image
import os
import sys


def convert(directory):
    if "sketch" not in directory:
        raise Exception("It looks like you are trying to convert a directory that might not be BW images.")
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            white = Image.new("L", image.size, "WHITE")
            white.paste(image, (0, 0), image)
            white.convert('L').point(lambda x: 255 if x > 200 else 0).save(file_path)


if __name__ == "__main__":
    convert(sys.argv[1])
