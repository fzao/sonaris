"""
Example of a script for converting two video files in parallel

Additional dependencies:
    - pathos

Author(s) : Fabrice Zaoui (EDF R&D LNHE)

Copyright EDF 2018
"""
from sonaris import Sonaris
from pathos.multiprocessing import ProcessingPool as Pool


def run(video_list):
    video_list.convert()


# ARIS files to convert and associated AVI file
conversion_1 = Sonaris('video_test.aris', '2014_1.avi')
conversion_2 = Sonaris('video_test.aris', '2014_2.avi')
# list Sonaris jobs
tab = [conversion_1, conversion_2]
# use a number of processors (ideally one proc. per ARIS file)
pool = Pool(nodes=2)
# launch conversion
pool.map(run, tab)
# close pool
pool.terminate()
pool.join()
