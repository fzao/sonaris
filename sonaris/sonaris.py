#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
    A video converter for the acoustic sonar ARIS file

    Author(s): Fabrice Zaoui

    Copyright EDF 2018

    Comments :
    - only ARIS v5 is supported
    - frame conversion algorithm from the Matlab toolbox ARISreader
        (https://github.com/nilsolav/ARISreader)
"""

import os
import numpy as np
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


class Sonaris(object):
    """
    The base class for the Sonar Aris Reader
    """
    def __init__(self, aris_file, avi_file):
        # input ARIS file name
        self.aris_file = aris_file
        # output AVI file name
        self.avi_file = avi_file

    def read_file_header(self):
        try:
            with open(self.aris_file, 'rb') as f:
                self.file_header = {}
                # type de fichier
                self.ftype = np.fromfile(f, dtype=np.uint8, count=3)
                self.file_header['type'] =\
                    ''.join([chr(item) for item in self.ftype])
                # version
                self.file_header['version'] =\
                    np.fromfile(f, dtype=np.uint8, count=1)
                # total frames in file
                self.file_header['numframes'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # initial recorded frame rate
                self.file_header['framerate'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # Non-zero if HF, zero if LF
                self.file_header['resolution'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # ARIS 3000 = 128/64, ARIS 1800 = 96/48, ARIS 1200 = 48
                self.file_header['numbeams'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # 1/Sample Period
                self.file_header['samplerate'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                # number of range samples in each beam
                self.file_header['sampleperchannel'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # relative gain in dB:  0 - 40
                self.file_header['receivergain'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # image window start range in meters (code [0..31] in DIDSON)
                self.file_header['windowstart'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                # image window length in meters  (code [0..3] in DIDSON)
                self.file_header['windowlength'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                # non-zero = lens down (DIDSON) or lens up (ARIS),
                # zero = opposite
                self.file_header['reverse'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # sonar serial number
                self.file_header['serialnumber'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # date that file was recorded
                self.date =\
                    np.fromfile(f, dtype=np.uint8, count=32)
                self.file_header['strdate'] =\
                    ''.join([chr(item) for item in self.date])
                # user input to identify file in 256 characters
                self.ids = np.fromfile(f, dtype=np.uint8, count=256)
                self.file_header['idstring'] =\
                    ''.join([chr(item) for item in self.ids])
                # user-defined integer quantity
                self.file_header['id1'] =\
                    np.fromfile(f, dtype=np.int32, count=1)
                # user-defined integer quantity
                self.file_header['id2'] =\
                    np.fromfile(f, dtype=np.int32, count=1)
                # user-defined integer quantity
                self.file_header['id3'] =\
                    np.fromfile(f, dtype=np.int32, count=1)
                # user-defined integer quantity
                self.file_header['id4'] =\
                    np.fromfile(f, dtype=np.int32, count=1)
                # first frame number from source file
                # (for DIDSON snippet files)
                self.file_header['startframe'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # last frame number from source file (for DIDSON snippet files)
                self.file_header['endframe'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # non-zero indicates time lapse recording
                self.file_header['timelapse'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # number of frames/seconds between recorded frames
                self.file_header['recordinterval'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # frames or seconds interval
                self.file_header['radioseconds'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # record every Nth frame
                self.file_header['frameinterval'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # see DDF_04 file format document
                self.file_header['flags'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # see DDF_04 file format document
                self.file_header['auxflags'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # sound velocity in water
                self.file_header['sspd'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # see DDF_04 file format document
                self.file_header['flags3d'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # DIDSON software version that recorded the file
                self.file_header['softwareversion'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # water temperature code:  0 = 5-15C, 1 = 15-25C, 2 = 25-35C
                self.file_header['watertemperature'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # salinity code:  0 = fresh, 1 = brackish, 2 = salt
                self.file_header['salinity'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # added for ARIS but not used
                self.file_header['pulselength'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # added for ARIS but not used
                self.file_header['txmode'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # reserved for future use
                self.file_header['versionfgpa'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # reserved for future use
                self.file_header['versionpsuc'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # frame index of frame used for thumbnail image of file
                self.file_header['thumbnailfi'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # total file size in bytes
                self.file_header['watertemperature'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                # reserved for future use
                self.file_header['optionalheadersize'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                # reserved for future use
                self.file_header['optionaltailsize'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                # DIDSON version minor
                self.file_header['versionminor'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # non-zero if telephoto lens
                # (large lens, hi-res lens, big lens) is present
                self.file_header['largelens'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                # free space for user
                self.file_header['userassigned'] =\
                    np.fromfile(f, dtype=np.uint8, count=568)
                # lenght of header file
                self.file_header['length'] = f.tell()
        except IOError:
            print('Error ->read_file_header<- : unable to open the ARIS file!')
            return
        f.close()
        return

    def read_frame_header(self):
        try:
            with open(self.aris_file, 'rb') as f:
                self.frame_header = {}
                f.seek(self.file_header['length'])
                self.frame_header['framenumber'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['frametime'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                self.frame_header['version'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['status'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['sonartimestep'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                self.frame_header['tsday'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['tshour'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['tsminute'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['tssecond'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['tshsecond'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['transmitmode'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['windowstart'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['windowlength'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['threshold'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['intensity'] =\
                    np.fromfile(f, dtype=np.int32, count=1)
                self.frame_header['receivergain'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['degc1'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['degc2'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['humidity'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['focus'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['battery'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['uservalue1'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue2'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue3'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue4'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue5'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue6'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue7'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['uservalue8'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['velocity'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['depth'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['altitude'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['pitch'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['pitchrate'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['roll'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollrate'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['heading'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['headingrate'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['compassheading'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['compasspitch'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['compassroll'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['latitude'] =\
                    np.fromfile(f, dtype=np.float64, count=1)
                self.frame_header['longitude'] =\
                    np.fromfile(f, dtype=np.float64, count=1)
                self.frame_header['sonarposition'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['configflags'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['beamtilt'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['targetrange'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['targetbearing'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['targetpresent'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['firmwareversion'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['flags'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['sourceframe'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['watertemp'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['timerperiod'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['sonarx'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonary'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonarz'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonarpan'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonartilt'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonarroll'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['panpnnl'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tiltpnnl'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollpnnl'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['vehicletime'] =\
                    np.fromfile(f, dtype=np.float64, count=1)
                self.frame_header['timeggk'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['dateggk'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['qualityggk'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['numsatsggk'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['dopggk'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['ehtggk'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['heavetss'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['yeargps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['monthgps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['daygps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['hourgps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['minutegps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['secondgps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['hsecondgps'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['sonarpanoffset'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonartiltoffset'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonarrolloffset'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonarxoffset'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonaryoffset'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sonarzoffset'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tmatrix'] =\
                    np.fromfile(f, dtype=np.float32, count=16)
                self.frame_header['samplerate'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['accellx'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['accelly'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['accellz'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['pingmode'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['frequencyhilow'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['pulsewidth'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['cycleperiod'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['sampleperiod'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['transmitenable'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['framerate'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['soundspeed'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['samplesperbeam'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['enable150v'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['samplestartdelay'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['largelens'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['thesystemtype'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['sonarserianumber'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['encryptedkey'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                self.frame_header['ariserrorflagsuint'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['missedpackets'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['arisappversion'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['available2'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['reorderedsamples'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['salinity'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['pressure'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['batteryvoltage'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['mainvoltage'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['switchvoltage'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['focusmotormoving'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['voltagechanging'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['focustimeoutfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['focusovercurrentfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['focusnotfoundfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['focusstalledfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['fpgatimeoutfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['fpgabusyfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['fpgastuckfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['cputempfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['psutempfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['watertempfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['humidityfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['pressurefault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['voltagereadfault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['voltagewritefault'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['focuscurrentposition'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['targetpan'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['targettilt'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['targetroll'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['panmotorerrorcode'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['tiltmotorerrorcode'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['rollmotorerrorcode'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['panabsposition'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tiltabsposition'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollabsposition'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['panaccelx'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['panaccely'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['panaccelz'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tiltaccelx'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tiltaccely'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tiltaccelz'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollaccelx'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollaccely'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollaccelz'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['appliedsettings'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['constrainedsettings'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['invalidsettings'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['enableinterpacketdelay'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['interpacketdelayperiod'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['uptime'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['arisappversionmajor'] =\
                    np.fromfile(f, dtype=np.uint16, count=1)
                self.frame_header['arisappversionminor'] =\
                    np.fromfile(f, dtype=np.uint16, count=1)
                self.frame_header['gotime'] =\
                    np.fromfile(f, dtype=np.uint64, count=1)
                self.frame_header['panvelocity'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['tiltvelocity'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['rollvelocity'] =\
                    np.fromfile(f, dtype=np.float32, count=1)
                self.frame_header['sentinel'] =\
                    np.fromfile(f, dtype=np.uint32, count=1)
                self.frame_header['userassigned'] =\
                    np.fromfile(f, dtype=np.uint8, count=292)
                self.frame_header['length'] =\
                    f.tell() - self.file_header['length']
        except IOError:
            print('Error ->read_frame_header<- :'
                  ' unable to open the ARIS file!')
            return
        f.close()
        return

    def extract_file_bin(self):
        try:
            with open(self.aris_file, 'rb') as f:
                f.seek(self.file_header['length'])
                self.movie = []
                nbframe = self.file_header['numframes']
                ix = self.file_header['numbeams']
                iy = self.file_header['sampleperchannel']
                framedim = self.file_header['numbeams'] * \
                    self.file_header['sampleperchannel']
                for i in range(0, nbframe[0]):
                    f.seek(f.tell() + self.frame_header['length'])
                    frame = np.fromfile(f, dtype=np.uint8, count=framedim[0])
                    self.movie.append(frame)
        except IOError:
            print('Error ->extract_file_bin<- : unable to open the ARIS file!')
            return
        f.close()
        self.movie = np.array(self.movie)
        self.movie = self.movie.reshape(nbframe[0], iy[0], ix[0])
        return

    def lens_distorsion(self, nbeams, theta):
        if nbeams == 48:
            factor = 1
            a = [.0015, -0.0036, 1.3351, 24.0976]
        elif nbeams == 189:
            factor = 4.026
            a = [.0015, -0.0036, 1.3351, 24.0976]
        elif nbeams == 96:
            factor = 1.012
            a = [.0030, -0.0055, 2.6829, 48.04]
        elif nbeams == 381:
            factor = 4.05
            a = [.0030, -0.0055, 2.6829, 48.04]
        elif nbeams == 509:
            factor = 5.45
            a = [.0030, -0.0055, 2.6829, 48.04]

        return np.round(factor * (a[0] * theta**3 +
                        a[1] * theta**2 + a[2] * theta + a[3]) + 1)

    def make_movie(self):
        format = "XVID"
        is_color = True
        vid = None
        size = None
        fourcc = VideoWriter_fourcc(*format)
        size = int(self.nx), int(self.ny)
        nbframe = self.file_header['numframes']
        frameRate = self.frame_header['framerate']
        nout = int(self.nout)
        n = int(self.n)
        m = int(self.m)
        self.vid = VideoWriter(self.avi_file, fourcc, float(frameRate),
                               size, is_color)
        for i in range(0, nbframe[0]):
            inframe = self.movie[i, :, :]
            inframe = np.flip(inframe, axis=1)
            outframe = np.zeros((m, nout))
            inframe = inframe.astype(float)
            outframe[:, np.arange(0, nout, 4)] = inframe
            outframe[:, np.arange(1, nout-3, 4)] = \
                0.75 * inframe[:, 0:n-1] + 0.25 * inframe[:, 1:n]
            outframe[:, np.arange(2, nout-2, 4)] = \
                0.50 * inframe[:, 0:n-1] + 0.50 * inframe[:, 1:n]
            outframe[:, np.arange(3, nout-1, 4)] = \
                0.25 * inframe[:, 0:n-1] + 0.75 * inframe[:, 1:n]
            # black fill
            outframe[0, 0] = 0.
            # angular transform and save
            outframe = outframe.ravel(order='F')
            outframe = outframe[self.svector-1]
            outframe = outframe.reshape(int(self.ny), int(self.nx), order='F')
            outframe = np.round(outframe)
            outframe = outframe.astype(dtype=np.uint8)
            frame = Image.fromarray(outframe)
            frame = frame.convert('RGB')
            img = np.array(frame)
            self.vid.write(img)
        self.vid.release()
        return

    def angular_converter(self):
        nbframe = self.file_header['numframes']
        frameRate = self.frame_header['framerate']
        minRange = self.frame_header['windowstart']
        maxRange = self.frame_header['windowstart'] + \
            self.frame_header['windowlength']
        self.m = np.int32(self.file_header['sampleperchannel'])
        self.n = np.int32(self.file_header['numbeams'])
        nrows = self.m
        self.nx = np.int32(np.round(0.1773 * self.m + 309))
        self.nout = 4 * self.n - 3
        half_angle = 14.  # for ARIS v5 only
        degtorad = np.pi / 180.  # conversion of degrees to radians
        radtodeg = 180. / np.pi  # conversion of radians to degrees
        # see drawing (distance from point scan touches
        # image boundary to origin)
        d2 = maxRange * np.cos(half_angle * degtorad)
        # see drawing (bottom of image frame to r,theta origin in meters)
        d3 = minRange * np.cos(half_angle*degtorad)
        # precalcualtion of constants used in do loop below
        c1 = (nrows - 1) / (maxRange - minRange)
        c2 = (self.nout - 1) / (2 * half_angle)
        # Ratio of pixel number to position in meters
        gamma = self.nx / (2 * maxRange * np.sin(half_angle * degtorad))
        # number of pixels in image in vertical direction
        self.ny = np.int32(np.fix(gamma * (maxRange - d3) + 0.5))
        # make vector and fill in later
        self.svector = np.zeros((np.int32(self.nx*self.ny)), dtype=int)
        ix = np.arange(1, self.nx + 1)  # pixels in x dimension
        x = ((ix - 1) - self.nx / 2) / gamma  # convert from pixels to meters
        for iy in range(1, int(self.ny)+1):
            y = maxRange - (iy-1)/gamma  # convert from pixels to meters
            r = np.sqrt(y*y + x*x)  # convert to polar cooridinates
            theta = radtodeg * np.arctan2(x, y)  # theta is in degrees
            binnum = np.fix((r - minRange) * c1 + 1.5)  # the rangebin number
            # remove lens distortion using empirical formula
            beamnum = self.lens_distorsion(self.nout, theta)
            pos = (beamnum > 0) * (beamnum <= self.nout)*(binnum > 0) * \
                (binnum <= nrows) * ((beamnum - 1) * nrows + binnum)
            # The offset in this array is the pixel offset in the image array
            self.svector[(ix-1) * self.ny + iy - 1] = pos
        # The value at this offset is the offset in the sample array
        self.svector[self.svector == 0] = 1
        return

    def convert(self):
        # check for file availability
        if os.path.isfile(self.aris_file) is False:
            print('Error ->' + self.aris_file + '<- ARIS file not found')
            return
        # read file header
        self.read_file_header()
        # check for ARIS version
        if self.file_header['version'] != 5:
            print('Error: only ARIS v5 is supported')
        # read frame header
        self.read_frame_header()
        # angular converter
        self.angular_converter()
        # read raw data
        self.extract_file_bin()
        # convert frames and make video avi
        self.make_movie()
