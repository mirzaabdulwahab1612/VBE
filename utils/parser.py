import argparse

def parse_args():

    parser = argparse.ArgumentParser(description = 'Main parser')

    parser.add_argument('--run_num' , type = int, help = 'Run number')
    parser.add_argument('--json_file', type = str, default = '', help = 'json path')
    parser.add_argument('--config_num', type = int, help = 'config number')
    parser.add_argument('--id', type = int, default = 0, help = 'id number')

    return parser.parse_args()
