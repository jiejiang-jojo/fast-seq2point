""" A train log parser """
import csv
import sys
import re
from datetime import datetime


FIELDS = {
    'tr_mae': re.compile(r'tr_mae: (\d+\.\d+),'),
    'va_mae': re.compile(r'va_mae: (\d+\.\d+)$'),
    'iteration': re.compile(r'iteration: (\d+),'),
    'train_time': re.compile(r'train time: (\d+\.\d+) s,'),
    'validate_time': re.compile(r'validate time: (\d+\.\d+) s'),
    'time_stamp': re.compile(r'(\d\d \w\w\w \d\d\d\d \d\d:\d\d:\d\d)')
}


def extract(ptn, line):
    """ Extract the pattern in the line """
    match = ptn.search(line)
    return match.group(1)


def parse_log(filename):
    """ Parse training log file to output csv """
    with open(filename) as fin:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS.keys())
        writer.writeheader()
        for line in fin:
            try:
                if 'tr_mae' in line:
                    row = {}
                    for field in ('tr_mae', 'va_mae', 'time_stamp'):
                        row[field] = extract(FIELDS[field], line)
                    row['time_stamp'] = datetime.strptime(row['time_stamp'],
                                                          '%d %b %Y %H:%M:%S').isoformat()
                elif 'iteration' in line:
                    for field in ('iteration', 'train_time', 'validate_time'):
                        row[field] = extract(FIELDS[field], line)
                elif '------------' in line:
                    writer.writerow(row)
            except Exception:
                pass

def console():
    """ Run from commandline """
    if len(sys.argv) < 2:
        print('Usage: parse_train_log.py <logfile>')
        sys.exit(-1)
    parse_log(sys.argv[1])

if __name__ == "__main__":
    console()
