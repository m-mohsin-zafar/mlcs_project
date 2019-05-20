#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:20:46 2019

@author: mohsin
"""

import pandas as pd
import numpy as np
import os
from paths import *
import matplotlib.pyplot as plt
import random

# Per Class Distribution
# Complete Data, Train Set, Test Set
train_asm_df = pd.read_csv(asm_csv_path, index_col='filename')
train_labels_df = pd.read_csv(labels_csv_path, index_col='Id')
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Merged Both CSVs into one based on Filenames (index)
merged_df = pd.merge(train_asm_df, train_labels_df, right_index=True, left_index=True)

# Group Merged DataFrame By 'Class' i.e Label
grouped_df = merged_df.groupby('Class')
# Group Train and Test DataFrames
train_grouped_df = train_df.groupby('Class')
test_grouped_df = test_df.groupby('Class')

# We have a total of 9 different Classes, so 9 groups will be made
per_class_count = []
for i in range(0, 9):
    per_class_count.append([('C-' + str(i + 1)), grouped_df.get_group(i + 1).shape[0],
                            train_grouped_df.get_group(i + 1).shape[0],
                            test_grouped_df.get_group(i + 1).shape[0]])

# Conversion to an array
per_class_count_df = pd.DataFrame(np.array(per_class_count), columns=['class', 'original', 'train', 'test'])
per_class_count_df[["original", "train", "test"]] = per_class_count_df[["original", "train", "test"]].apply(
    pd.to_numeric)
per_class_count_df.sort_values(by=['original'])

# Prepare Data
n = 9
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)

ALREADY_COMPUTED = True

if not ALREADY_COMPUTED:

    # Plot Bars
    plt.figure(figsize=(16, 10), dpi=200)
    plt.bar(per_class_count_df['class'], per_class_count_df['original'], color=c, width=.5)
    for i, val in enumerate(per_class_count_df['original'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
                 fontdict={'fontweight': 500, 'size': 12})

    # Decoration
    plt.gca().set_xticklabels(per_class_count_df['class'], rotation=60, horizontalalignment='right')
    plt.title("Examples Counts Per Class", fontsize=22)
    plt.ylabel('Count')
    plt.ylim()
    plt.show()

    # Plot Bars
    plt.figure(figsize=(16, 10), dpi=200)
    plt.bar(per_class_count_df['class'], per_class_count_df['train'], color=c, width=.5)
    for i, val in enumerate(per_class_count_df['train'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
                 fontdict={'fontweight': 500, 'size': 12})

    # Decoration
    plt.gca().set_xticklabels(per_class_count_df['class'], rotation=60, horizontalalignment='right')
    plt.title("Examples Counts Per Class (Train Set)", fontsize=22)
    plt.ylabel('Count')
    plt.ylim()
    plt.show()

    # Plot Bars
    plt.figure(figsize=(16, 10), dpi=200)
    plt.bar(per_class_count_df['class'], per_class_count_df['test'], color=c, width=.5)
    for i, val in enumerate(per_class_count_df['test'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
                 fontdict={'fontweight': 500, 'size': 12})

    # Decoration
    plt.gca().set_xticklabels(per_class_count_df['class'], rotation=60, horizontalalignment='right')
    plt.title("Examples Counts Per Class (Test Set)", fontsize=22)
    plt.ylabel('Count')
    plt.ylim()
    plt.show()

    # Draw Plot
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=100)

    data = per_class_count_df['original']
    categories = per_class_count_df['class']
    explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0]


    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}% ({:d} )".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),
                                      colors=c,
                                      startangle=140,
                                      explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="Malware Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("Malware Examples Per Class: Pie Chart")
    plt.show()

    # Draw Plot
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=100)

    data = per_class_count_df['train']
    categories = per_class_count_df['class']
    explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0]

    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),
                                      colors=c,
                                      startangle=140,
                                      explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="Malware Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("Malware Examples Per Class (Train Set): Pie Chart")
    plt.show()

    # Draw Plot
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=100)

    data = per_class_count_df['test']
    categories = per_class_count_df['class']
    explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0]

    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),
                                      colors=c,
                                      startangle=140,
                                      explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="Malware Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("Malware Examples Per Class (Test Set): Pie Chart")
    plt.show()
