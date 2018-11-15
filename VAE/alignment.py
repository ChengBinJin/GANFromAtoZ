# ---------------------------------------------------------
# Tensorflow pix2pix Implementation for Day2Night Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import csv
import cv2
import numpy as np


def main(file_name_):
    with open(file_name_, 'r') as csvfile:
        # read csv file
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        cv2.namedWindow('window')
        cv2.moveWindow('window', 0, 0)
        for row in reader:
            night_img_path = os.path.join('Night_frames', 'Image' + row[0].zfill(5) + '.jpg')
            day_img_path = os.path.join('Day_frames', 'Image' + row[1].zfill(5) + '.jpg')

            # read night and day images
            night_img = cv2.imread(night_img_path)
            day_img = cv2.imread(day_img_path)

            height, width, channel = night_img.shape
            canvas = np.zeros((height, 2*width, channel), dtype=np.uint8)
            canvas[:, :width, :] = day_img
            canvas[:, width:2*width, :] = night_img

            print('Day image index: {}, Night image index: {}'.format(row[1], row[0]))
            cv2.imshow('window', canvas)
            cv2.waitKey(1)

            cv2.imwrite(os.path.join('../data/paired', row[1].zfill(5) + '.png'), canvas)


if __name__ == '__main__':
    file_name = 'framematches.csv'
    main(file_name)

