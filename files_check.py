#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:19:49 2019

@author: mohsin
"""

import re
import os


log_file_path = os.path.join(os.getcwd(), 'feature_ext', 'logs.txt')

file_text = []
files = []

with open(log_file_path, encoding='utf-8' ,mode='r+') as logfile:
    file_text = logfile.readlines()
    
    
file_text = str(file_text)
files = re.findall(r'[\a-zA-Z0-9]{20}[\.]', file_text)

filenames = []
for file in files:
    filenames.append(file.split('.')[0])
    

files_not_done = re.findall('Not Done', file_text)
