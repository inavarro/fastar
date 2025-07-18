#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as os_path


def get_asset_path(filename):
    file_dir = os_path.dirname(__file__)
    rel_path = os_path.join(file_dir, '..', 'assets', filename)
    abs_path = os_path.abspath(rel_path)

    return abs_path
