""" A train log parser """
import csv
import sys
import re
import json
from datetime import datetime

"""
Wed, 10 Oct 2018 15:06:52 main_pytorch.py[line:385] INFO Namespace(batch_size=128, binary_threshold=200, config='/vol/vssp/msos/qk/JieJiang/REFIT/fast-seq2point/wvn_mw_config.json', cuda=True, inference_house='house20', inference_model='0005_microwave_WaveNet_iter_50000_wd_100_sl_127.tar', mode='inference', model='WaveNet', model_params={'layers': 6, 'to_binary': True}, model_threshold=0.5, target_device='microwave', train_house_list=['house2', 'house3', 'house4', 'house5', 'house6', 'house8', 'house9', 'house10', 'house11', 'house12', 'house15'], validate_house_list=['house17', 'house18', 'house19', 'house20'], validate_max_iteration=200, width=100, workspace='/vol/vssp/msos/qk/JieJiang/REFIT/fast-seq2point')
Wed, 10 Oct 2018 15:06:52 main_pytorch.py[line:261] INFO config={"model": "WaveNet", "model_params": {"layers": 6, "to_binary": true}, "batch_size": 128, "width": 100, "validate_max_iteration": 200, "target_device": "microwave", "train_house_list": ["house2", "house3", "house4", "house5", "house6", "house8", "house9", "house10", "house11", "house12", "house15"], "validate_house_list": ["house17", "house18", "house19", "house20"], "inference_house": "house20", "binary_threshold": 200, "model_threshold": 0.5, "inference_model": "0005_microwave_WaveNet_iter_50000_wd_100_sl_127.tar", "mode": "inference", "workspace": "/vol/vssp/msos/qk/JieJiang/REFIT/fast-seq2point", "config": "/vol/vssp/msos/qk/JieJiang/REFIT/fast-seq2point/wvn_mw_config.json", "cuda": true}
Wed, 10 Oct 2018 15:06:58 data_generator.py[line:54] INFO Load data time: 1.4530041217803955 s
Wed, 10 Oct 2018 15:06:58 data_generator.py[line:61] INFO mean 6.2072062492370605, std 91.63136291503906, max3778.0:
Wed, 10 Oct 2018 15:07:24 data_generator.py[line:77] INFO Number of indexes: 53140826
Wed, 10 Oct 2018 15:07:25 main_pytorch.py[line:297] INFO Inference time: 0.7719681262969971 s
Wed, 10 Oct 2018 15:07:26 main_pytorch.py[line:299] INFO Metrics: {'tp': 6539, 'fn': 8790, 'fp': 1366, 'tn': 3960483, 'precision': '0.8272', 'recall': '0.4266', 'f1_score': '0.5629', 'auc': '0.7131', 'average_precision': '0.3551'}
"""

"""
device house threshold precision recall f1
wm     18    0.1       0.6345    0.9489 0.7604
"""

CONFIG_PTN = re.compile(r'INFO config=(.*)$')
METRIC_PTN = re.compile(r'INFO Metrics: (.*)$')

def flattern(obj):
    """ Make a flattern dict out of nested dicts and lists """
    new_obj = {}
    for k, v in obj.items():
        if isinstance(v, list):
            new_obj[k] = ','.join(v)
        elif isinstance(v, dict):
            new_obj.update({'{}.{}'.format(k, sk): sv for sk, sv in flattern(v).items()})
        else:
            new_obj[k] = v
    return new_obj

def extract(ptn, line):
    """ Extract the pattern in the line """
    match = ptn.search(line)
    return match.group(1)


def parse_log(filename):
    """ Parse training log file to output csv """
    result = {}
    with open(filename) as fin:
        for line in fin:
            try:
                if 'INFO config=' in line:
                    result.update(flattern(json.loads(CONFIG_PTN.search(line).group(1))))
                elif 'INFO Metrics:' in line:
                    result.update(flattern(json.loads(
                        METRIC_PTN.search(line).group(1).replace("'", '"'))))
            except Exception:
                pass
    return result

def console():
    """ Run from commandline """
    if len(sys.argv) < 2:
        print('Usage: parse_train_log.py <logfile>')
        sys.exit(-1)
    writer = None
    for logfile in sys.argv[1:]:
        row = parse_log(logfile)
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    console()
