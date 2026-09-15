""" """

import logging

from config import ROOT_PATH

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_PATH + "./logs/KShortestAlgorithm.log")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class Logger:
    def __init__(self, file_name, level) -> None:
        self.level = level
        self.root_path = " "
        self.file_name = file_name

    def logg_info(self, level="warn"):
        pass
