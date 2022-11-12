from asyncore import read
from concurrent.futures import process
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image


def read_csv(filepath):
    file = open(filepath)
    data_dict = {}

    reader = csv.reader(file)
    for row in reader:
        data_dict[row[0]] = row[1]
    return data_dict
        

def process_training_data(data_dict):
    all_images = np.zeros((45, 400, 400), dtype=float)
    for i, path in enumerate(data_dict.keys()): 
        new_img = create_img_array(path)
        all_images[i] = new_img
        break
    return all_images


def create_img_array(img_path):
    img = Image.open(img_path)
    resized = img.resize((400, 400))
    adjusted_img = resized.convert("L")
    adjusted_img.show()
    np_img = np.array(adjusted_img)
    return np_img


def main():
    data_dict = read_csv("data.csv")
    np_data = process_training_data(data_dict)
    print("FIRST IMAGE: {}".format(np_data[2]))
    

if __name__== "__main__":
    main()