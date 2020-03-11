import os


def mk_if_not_exits(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)