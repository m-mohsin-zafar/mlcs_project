#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:38:50 2019

@author: mohsin
"""

import time
import traceback
from csv import writer
import numpy
import six
import os
import gzip
from feature_ext.feature_extraction import *
from feature_ext.header_construction import *
from feature_ext.settings import *


def asm_extraction(dataset_type):
    read_mode, write_mode = ('r', 'w') if six.PY2 else ('rt', 'wt')

    defined_apis = io.read_all_lines(APIS_PATH)
    defined_apis = defined_apis[0].split(',')

    directory_name = os.path.join(DATASET_PATH, (dataset_type + '_asm/'))
    files = os.listdir(directory_name)
    files = numpy.sort(files)
    asm_files = [i for i in files if i.endswith('.asm.gz')]
    #    original = [i for i in files if i.endswith('.asm')]

    asm_csv = os.path.join(ASM_CSV_PATH, (dataset_type + '_asm_csv.gz'))

    #    if not os.path.isdir(SAVED_PATH_CSV + dataset_type):
    #        makedirs(SAVED_PATH_CSV + dataset_type)

    symbols_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_symbols.csv', write_mode)
    meta_data_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_meta_data.csv', write_mode)
    registers_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_registers.csv', write_mode)
    opcodes_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_opcodes.csv', write_mode)
    sections_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_sections.csv', write_mode)
    data_define_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_data_define.csv', write_mode)
    apis_csv = open(SAVED_PATH_CSV + dataset_type + '/asm_apis.csv', write_mode)

    symbols_time = []
    m_data_time = []
    registers_time = []
    opcodes_time = []
    sections_time = []
    data_define_time = []
    apis_time = []

    log_file = open("logs.txt", "w+")

    with gzip.open(asm_csv, write_mode) as f_asm_csv:
        # Header construction
        fw = writer(f_asm_csv)
        colnames = ['filename']
        colnames += header_asm_meta_data()
        colnames += header_asm_sym()
        colnames += header_asm_registers()
        colnames += header_asm_opcodes()
        colnames += header_asm_sections()
        colnames += header_asm_data_define()
        colnames += header_asm_apis()

        meta_data_colnames = header_asm_meta_data()
        sym_colnames = header_asm_sym()
        registers_colnames = header_asm_registers()
        opcodes_colnames = header_asm_opcodes()
        sections_colnames = header_asm_sections()
        data_define_colnames = header_asm_data_define()
        apis_colnames = header_asm_apis()

        meta_data_csv_w = writer(meta_data_csv)
        meta_data_csv_w.writerows([meta_data_colnames])
        symbols_csv_w = writer(symbols_csv)
        symbols_csv_w.writerows([sym_colnames])
        registers_csv_w = writer(registers_csv)
        registers_csv_w.writerows([registers_colnames])
        opcodes_csv_w = writer(opcodes_csv)
        opcodes_csv_w.writerows([opcodes_colnames])
        sections_csv_w = writer(sections_csv)
        sections_csv_w.writerows([sections_colnames])
        data_define_csv_w = writer(data_define_csv)
        data_define_csv_w.writerows([data_define_colnames])
        apis_csv_w = writer(apis_csv)
        apis_csv_w.writerows([apis_colnames])

        fw.writerow(colnames)

        # Creating row set
        rows = []
        for t, fname in enumerate(asm_files):
            f = gzip.open(directory_name + fname, read_mode, encoding='utf-8', errors='ignore')
            #            f = open(directory_name + fname, read_mode, errors='ignore')

            # Meta data
            try:
                f2 = gzip.open(directory_name + fname, 'r')
                #                f2 = open(directory_name + fname, 'r')
                start_time = time.time()
                meta_data = asm_meta_data(directory_name + fname, f2)
                required_time = time.time() - start_time
                m_data_time.append(required_time)
                meta_data_csv_w.writerows([meta_data])

                f.seek(0)

                start_time = time.time()
                symbols = asm_symbols(f)
                required_time = time.time() - start_time
                symbols_time.append(required_time)
                symbols_csv_w.writerows([symbols])

                f.seek(0)

                start_time = time.time()
                registers = asm_registers(f)
                required_time = time.time() - start_time
                registers_time.append(required_time)
                registers_csv_w.writerows([registers])

                f.seek(0)

                start_time = time.time()
                opcodes = asm_opcodes(f)
                required_time = time.time() - start_time
                opcodes_time.append(required_time)
                opcodes_csv_w.writerows([opcodes])

                f.seek(0)

                start_time = time.time()
                sections, names = asm_sections(f)
                required_time = time.time() - start_time
                sections_time.append(required_time)
                sections_csv_w.writerows([sections])

                f.seek(0)

                start_time = time.time()
                data_defines = asm_data_define(f)
                required_time = time.time() - start_time
                data_define_time.append(required_time)
                data_define_csv_w.writerows([data_defines])

                f.seek(0)

                start_time = time.time()
                apis = asm_APIs(f, defined_apis)
                required_time = time.time() - start_time
                apis_time.append(required_time)
                apis_csv_w.writerows([apis])

            except Exception as err:
                print(err, traceback.print_exc())
                print("Error", fname)
                log_file.write((str(t + 1) + '\t Not Done \t Filename: ' + fname))

            # Row added
            whole = meta_data + symbols + registers + opcodes + sections + data_defines + apis
            rows.append([fname[:fname.find('.asm.gz')]] + whole)

            # Writing rows after every 22 files processed
            if (t + 1) % 22 == 0:
                fw.writerows(rows)
                print(t + 1)
                rows = []

            log_file.write((str(t + 1) + '\t Done \t Filename: ' + fname))

    # Save the time
    m_data_time_str = ', '.join(str(x) for x in m_data_time)
    symbols_time_str = ', '.join(str(x) for x in symbols_time)
    registers_time_str = ', '.join(str(x) for x in registers_time)
    opcodes_time_str = ', '.join(str(x) for x in opcodes_time)
    sections_time_str = ', '.join(str(x) for x in sections_time)
    data_define_time_str = ', '.join(str(x) for x in data_define_time)
    apis_time_str = ', '.join(str(x) for x in apis_time)

    io.save_txt(('asm_meta_data_' + dataset_type, m_data_time_str, 'asm_sym_' + dataset_type, symbols_time_str,
                 'asm_registers_' + dataset_type, registers_time_str, 'asm_opcodes_' + dataset_type, opcodes_time_str,
                 'asm_sections_' + dataset_type, sections_time_str, 'asm_datadefine_' + dataset_type,
                 data_define_time_str,
                 'asm_apis_' + dataset_type, apis_time_str
                 ), ASM_TIME_PATH)

    log_file.close()


if __name__ == '__main__':
    asm_extraction('train')
