import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_nvi(filePath):
    nviData = sio.readsav(filePath)
    return pyData(nviData)
 
class pyData(object):
    def __init__(self,nviData):
        self.raw = nviData['imagetosave']
        self.data = nviData['imagetosave'].currentdata[0]
        self.header = {name:self.raw.header[0][name][0] for name in self.raw.header[0].dtype.names}
        self.info = {'FILENAME'    : self.raw.filename[0],
                     'FILSIZE'     : int(self.raw.header[0].filesize[0]),
                     'CHANNELS'    : self.raw.header[0].scan_channels[0],
                     'XSIZE'       : self.raw.xsize[0],
                     'YSIZE'       : self.raw.ysize[0],
                     'TEMPERATURE' : self.raw.header[0].temperature[0],
                     'LOCKIN_AMPLITUDE' : self.raw.header[0].lockin_amplitude[0],
                     'LOCKIN_FREQUENCY' : self.raw.header[0].lockin_frequency[0],
                     'DATE'        : self.raw.header[0].date[0],
                     'TIME'        : self.raw.header[0].time[0],
                     'BIAS_SETPOINT'    : self.raw.header[0].bias_setpoint[0],
                     'BIAS_OFFSET' : self.raw.header[0].bias_offset[0],
                     'BFIELD'      : self.raw.header[0].bfield[0],
                     'ZUNITS'      : self.raw.zunits[0],
        }
        