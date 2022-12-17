import os
import json
import time

def make_dir(path):
    if os.path.isdir(path):
        # print(path, 'already exists.')
        pass
    else:
        os.mkdir(path)
        print(path, 'created')

def check_folder(path) -> None:
    if os.path.isdir(path):
        print(path, 'already exists.')
    else:
        path_list = path.split('/')
        # for idx, folder in enumerate(path_list[:-1]):
        for idx, folder in enumerate(path_list):
            make_dir(os.path.join('/', *path_list[:idx+1]))

def print_reminders(todos, prompt=True):
    if len(todos)==0: print('you have no standing reminders')
    else:
        # print('TODOs:')
        print('┌─────────┐\n│  TODOs: │\n└─────────┘')
        for todo in todos:
            print(todo)

        if prompt:
            if input('\n>>> do you still want to proceed [Y/n]: ')=='n': exit()
        # print('\n')
        print()

def cooldown(secs):
    for i in range(secs):
        print('cooling down... {} '.format(29-i), end='\r'); time.sleep(1)
    print('cooling down... (done)')

def save_str_net(net, pathj='str_net--1.txt'):
    str_net = net.__repr__()
    with open(pathj, 'w', encoding='utf-8') as f:
        f.write(str_net)
    print('\x1b[32mstr_net saved to %s\x1b[39m'%pathj)

def pjson(data, indent=4, title=None):
    if title: print('{}:'.format(title), json.dumps(data, indent=indent))
    else: print(json.dumps(data, indent=indent))

class tcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colorize(text, color):
    return f"{color}{text}{tcolors.ENDC}"
