#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def get_data_path(filename, subdir='aux'):
    base_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_dir, '..', subdir, filename))
