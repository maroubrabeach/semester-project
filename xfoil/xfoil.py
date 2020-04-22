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
SAVEPATH = './save/polar.txt'
CPSAVEPH = './save/cpx.txt'
DATAPATH = './data/sample_'

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
    if calc_polar(afile, re, SAVEPATH, CPSAVEPH, *args,**kwargs):
        print("Computed polar. reading it now..")
        data = read_polar(SAVEPATH)
        data = read_cpx(CPSAVEPH, data)
        #delete_polar(SAVEPATH)
        return data
    return None


def calc_polar(afile, re, polarfile, cpfile, alfaseq=[], refine=False, max_iter=3000, n=None):
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
    timer = Timer(60, pxfoil.kill)
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
                write2xfoil('X\n ')
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
        for alfa in alfaseq:
            write2xfoil('A ' + str(alfa) + '\n')

        write2xfoil('PWRT 1\n')
        write2xfoil(polarfile + '\n')
        write2xfoil('\n')
        # needs to be included in the for loop for aoa
        write2xfoil('CPWR ' + cpfile + '\n')

        write2xfoil('\n')

        pxfoil.communicate(str('quit').encode('ascii'))
        is_done = True
    finally:
        timer.cancel()
        return is_done



def read_polar(infile):
    """read xfoil polar results from file

    Parameters
    ----------
    infile: string path to polar file

    Returns
    -------
    data: airfoil polar splitted up into dictionary
    """

    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(infile) as f:
        lines = f.readlines()

        a           = []
        cl          = []
        cd          = []
        cdp         = []
        cm          = []
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

        return data


def read_cpx(infile, data):
    """read xfoil pressure coefficient results from file

    Parameters
    ----------
    infile: string path to Cp file

    Returns
    -------
    data: adds Cp results to current data dictionary
    """

    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(infile) as f:
        lines = f.readlines()

        x           = []
        cp          = []


        for line in lines[1:]:
            linedata = regex.findall(line)
            x.append(float(linedata[0]))
            # to avoid the error: "could not convert string to float: 'N'" when
            # no convergence is reached and Cp = 'NaN'
            if 'N' not in linedata[1]:
                cp.append(float(linedata[1]))

        data['x'] = np.array(x)
        data['cp'] = np.array(cp)

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
    random.shuffle(digits)
    sample_num = 0
    for elem in digits:
        data = polar(elem, 2E6, [0, 1])
        if data != None:
            np.save(DATAPATH + "%04d" % sample_num, data)
            sample_num += 1
    print('done') # this is never printed…
