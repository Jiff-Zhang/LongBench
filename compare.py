#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ***********************************************
#      Filename: compare.py
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-04-27 09:37:13
# Last Modified: Year-month-day
# ***********************************************

import json
import os
import argparse

def compare(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = [json.loads(line) for line in f1]
        data2 = [json.loads(line) for line in f2]

    data1 = {item["_id"]: item for item in data1}
    data2 = {item["_id"]: item for item in data2}
    id1 = set(data1.keys())
    id2 = set(data2.keys())
    idx = sorted(list(id1 & id2)) # id1 & id2

    count = 0
    counts = {'positive': 0, 'negative': 0}
    for i in idx:
        if data1[i]['judge'] != data2[i]['judge']:
            print(f'{i}: {data1[i]["judge"]} != {data2[i]["judge"]}')
            count += 1
            if data1[i]['judge']:
                counts["negative"] += 1
            else:
                counts["positive"] += 1
            
    print(f'\ninfo: {count} / {len(idx)}')
    print(f'positive: {counts["positive"]} / {count}')
    print(f'negative: {counts["negative"]} / {count}')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f1", "--file1", type=str, help="file1")
    argparser.add_argument("-f2", "--file2", type=str, help="file2")
    args = argparser.parse_args()
    compare(args.file1, args.file2)