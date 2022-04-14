# -*- coding: utf-8 -*-
import pickle
import os
import logging

"""
常用文件工具类
"""
logger = logging.getLogger(__name__)


def save_pickle(obj, save_path):
    if os.path.exists(save_path):
        logger.warning("文件{}已存在!".format(save_path))
    else:
        save_file = open(save_path, 'wb', encoding='utf-8')
        pickle.dump(save_file, save_path)
        save_file.close()
        logger.info("文件{}已保存成功!".format(save_path))


def load_pickle(load_path):
    if not os.path.exists(load_path):
        logger.error("load path: {}不存在!".format(load_path))
    else:
        load_file = open(load_path, 'rb', encoding='utf-8')
        pickle_obj = pickle.load(load_path)
        load_file.close()
        logger.info("加载{}文件成功!".format(load_path))
        return pickle_obj