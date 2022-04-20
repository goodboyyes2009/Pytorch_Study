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
        raise FileExistsError("文件{}已存在!".format(save_path))
    else:
        try:
            save_file = open(save_path, 'wb')
            pickle.dump(obj, save_file)
            save_file.close()
            logger.info("文件{}已保存成功!".format(save_path))
        except Exception as e:
            logger.error("保存文件:{}失败, 异常信息:{}".format(save_path, e))


def load_pickle(load_path):
    if not os.path.exists(load_path):
        logger.error("load path: {}不存在!".format(load_path))
        raise FileNotFoundError("load path: {}不存在!".format(load_path))
    else:
        pickle_obj = None
        try:
            load_file = open(load_path, 'rb')
            pickle_obj = pickle.load(load_file)
            load_file.close()
            logger.info("加载{}文件成功!".format(load_path))
        except Exception as e:
            logger.error("加载{}文件失败,异常信息:{}".format(load_path, e))
        return pickle_obj