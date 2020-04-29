# -*- coding: utf-8 -*-

import os
import numpy as np
import subprocess as sp
from threading import Timer
import re
import pdb
import sys
import itertools
from operator import add
import random

XFOILBIN = './bin/xfoil'
POLARPATH = './save/polar.txt'
CPPATH = './save/cpx.txt'
COORDPATH = './save/coordinates.txt'
DATAPATH = './data/naca_'

def polar(afile, re, *args, **kwargs):
    """calculate airfoil polar and load results

    Parameters
    ----------
    afile: string path to aifoil dat file, or NACA
    re: float fixed reynolds number for polar calculation
    af: first alpha value (deg)
    al: last alpha value (deg)
    ainc: alpha increment (deg)
    *args, **kwargs: have a look at calcpolar for further information

    Returns
    -------
    dict
    airfoil polar
    """
    if calc_polar(afile, re, POLARPATH, CPPATH, COORDPATH, *args,**kwargs):
        print("Computed polar. Reading it now...")
        data = read(POLARPATH, CPPATH, COORDPATH)
        #delete_polar(POLARPATH)
        return data
    return None


def calc_polar(afile, re, polarfile, cpfile, coordfile, alfa, refine=False, max_iter=1000, n=None):
    """run xfoil to generate polar file

    Parameters
    ----------
    afile: string
        path to airfoil dat file
    re: float
        fixed reynolds number
    alfaseq: iterateable, optional
        sequence of angles of attack
    refine: bool
        shall xfoil refine airfoil
    maxiter: int
        maximal number of boundary layer iterations
    n: int
        boundary layer parameter
    """


    FNULL = open(os.devnull, 'w')
    pxfoil = sp.Popen([XFOILBIN], stdin=sp.PIPE, stdout=FNULL, stderr=None)
    timer = Timer(10, pxfoil.kill())
    is_done = False
    try:
        timer.start()
        def write2xfoil(string):
            if(sys.version_info > (3,0)):
                string = string.encode('ascii')
            pxfoil.stdin.write(string)

        if(afile.isdigit()):
            write2xfoil('NACA ' + afile + '\n')
        else:
            write2xfoil('LOAD ' + afile + '\n')

            if(refine):
                write2xfoil('GDES\n')
                write2xfoil('CADD\n')
                write2xfoil('\n')
                write2xfoil('\n')
                write2xfoil('\n')
                write2xfoil('X\n')
                write2xfoil('\n')
                write2xfoil('PANE\n')

        write2xfoil('OPER\n')
        if n != None:
            write2xfoil('VPAR\n')
            write2xfoil('N ' + str(n) + '\n')
            write2xfoil('\n')
        write2xfoil('ITER ' + str(max_iter) + '\n')
        write2xfoil('VISC\n')
        write2xfoil(str(re) + '\n')
        write2xfoil('PACC\n')
        write2xfoil('\n')
        write2xfoil('\n')
        # write2xfoil('ASeq ' + str(af) + ' ' + str(al) + ' ' + str(ainc) + '\n')
        # write2xfoil('\n')
        #for alfa in alfaseq:
        write2xfoil('Alfa ' + str(alfa) + '\n')
        write2xfoil('PWRT\n')
        write2xfoil(polarfile + '\n')
        write2xfoil('\n')
        write2xfoil('CPWR ' + cpfile + '\n')
        write2xfoil('\n')
        write2xfoil('SAVE ' + coordfile + '\n')
        pxfoil.communicate(str('quit').encode('ascii'))
        is_done = True
    finally:
        timer.cancel()
        return is_done



def read(polarfile, cpfile, coordfile):
    """read xfoil polar results from file

    Parameters
    ----------
    infile: string path to polar file

    Returns
    -------
    data: airfoil polar splitted up into dictionary
    """

    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(polarfile) as f:
        lines = f.readlines()

        a   = []
        cl  = []
        cd  = []
        cdp = []
        cm  = []
        # xtr_top     = []
        # xtr_bottom  = []

        for line in lines[12:]:
            linedata = regex.findall(line)
            a.append(float(linedata[0]))
            cl.append(float(linedata[1]))
            cd.append(float(linedata[2]))
            cdp.append(float(linedata[3]))
            cm.append(float(linedata[4]))
            # xtr_top.append(float(linedata[5]))
            # xtr_bottom.append(float(linedata[6]))

        data = {'a': np.array(a), 'cl': np.array(cl) , 'cd': np.array(cd), 'cdp': np.array(cdp),
             'cm': np.array(cm)}

    with open(cpfile) as f:
        lines = f.readlines()

        cp  = []

        for line in lines[1:]:
            linedata = regex.findall(line)
            if 'N' not in linedata[1]:
                cp.append(float(linedata[1]))

        data['cp'] = np.array(cp)

    with open(coordfile) as f:
        lines = f.readlines()

        x   = []
        y   = []

        for line in lines[1:]:
            linedata = regex.findall(line)
            x.append(float(linedata[0]))
            y.append(float(linedata[1]))

        data['x'] = np.array(x)
        data['y'] = np.array(y)

        return data


def delete_polar(infile):
    """ deletes polar file """
    os.remove(infile)


if __name__ == "__main__":
    # test making sure everythins is allright
    # NACA profile, Reynolds, (aoa start, aoa end, aoa increment)
    # create a list of all possible four-digit NACA airfoils
    digits = list(map("".join, list(itertools.product(['0','1','2','3','4','5','6','7','8','9'], repeat=4))))
    # remove airfoils with 0 thickness
    digits = [elem for elem in digits if not elem.endswith('00')]
    #random.shuffle(digits)
    sample_num = 0
    #for elem in digits:
    data = polar('1313', 2E6, 0)
    if data != None and data['a'].size > 0:
        #np.save(DATAPATH + "%04d" % sample_num, data)
        #np.save(DATAPATH + elem, data)
        #sample_num += 1
        f = open(DATAPATH + '1313' + ".txt", "w")
        f.write(str(data))
        f.close()
    print('*** Over ***')
