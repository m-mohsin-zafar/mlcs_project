#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:22:22 2019

@author: mohsin
"""

import os

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')
ROOT_PATH = os.getcwd()
SAVED_PATH_CSV = os.path.join(ROOT_PATH, 'io_content', 'feature_csvs')
BYTES_CSV_PATH = os.path.join(ROOT_PATH, 'io_content')
ASM_CSV_PATH = os.path.join(ROOT_PATH, 'io_content')
TRAIN_ID_PATH = os.path.join(ROOT_PATH, 'io_content', 'trainLabels.csv')
TEST_ID_PATH = os.path.join(ROOT_PATH, 'io_content/sorted_test_id.csv')
COMBINED_PATH_CSV = os.path.join(ROOT_PATH + 'combination')
BYTE_TIME_PATH = os.path.join(ROOT_PATH, 'io_content', 'byte_time.txt')
ASM_TIME_PATH = os.path.join(ROOT_PATH, 'io_content', 'asm_time.txt')
APIS_PATH = os.path.join(ROOT_PATH, 'io_content', 'APIs.txt')
TRAIN_FILE = os.path.join(ROOT_PATH, 'io_content', 'train', 'LargeTrain.csv')
TEST_FILE = os.path.join(ROOT_PATH, 'io_content', 'test', 'LargeTest.csv')
