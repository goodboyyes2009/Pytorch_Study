# -*- coding: utf-8 -*-
import json
from json import JSONEncoder

class PoetryInfo(object):
    def __init__(self, title, author, paragraphs, *args, **kwargs):
        self.title = title
        self.author = author
        self.paragraphs = paragraphs

class PoetryEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


if __name__ == "__main__":
    s = """{
        "title": "兩儀殿賦柏梁體",
        "author": "李世民",
        "biography": "",
        "paragraphs": [
            "絕域降附天下平，——李世民",
            "八表無事悅聖情。——淮安王",
            "雲披霧斂天地明，——長孫無忌",
            "登封日觀禪雲亭，——房玄齡",
            "太常具禮方告成。——蕭瑀"
        ],
        "notes": [
            ""
        ],
        "volume": "卷一",
        "no#": 87
    }
    """
    # https://pynative.com/python-convert-json-data-into-custom-python-object/
    # student = Student(1, "Emma")
    #
    # # encode Object it
    # studentJson = json.dumps(student, cls=StudentEncoder, indent=4)
    p = json.loads(s)
    pp = PoetryInfo(**p)
    print(pp.paragraphs)