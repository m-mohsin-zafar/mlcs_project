#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:22:22 2019

@author: mohsin
"""

import os


DATASET_PATH = "/home/mohsin/Kaggle_datasets/malware_c/"
os.chdir(DATASET_PATH)

SAVED_PATH_CSV = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/feature_csvs/'
BYTES_CSV_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/'
ASM_CSV_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/'
TRAIN_ID_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/trainLabels.csv'
TEST_ID_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/sorted_test_id.csv'
COMBINED_PATH_CSV = DATASET_PATH + 'combination/'
BYTE_TIME_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/byte_time.txt'
ASM_TIME_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/asm_time.txt'
APIS_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/APIs.txt'
TRAIN_FILE = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/train' + '/LargeTrain.csv'
TEST_FILE = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/test' + '/LargeTest.csv'
SUB_PATH = '/home/mohsin/Kaggle_datasets/malware_c/code/io_content_debug/submissions/'