""" Tests for XPM sequence generation"""

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import argparse
import psdaq.seq.seq
from psdaq.seq.globals import *
from psdaq.seq.seqplot import *
from pprint import pprint

from opcpa_tpr_config import xpm_prog


def plot_seq(seq_data: tuple):
    """ Plots the sequence waveform """
    app = QtWidgets.QApplication([])
    plot = PatternWaveform()
    
    plot.add("Simulation", *seq_data)

    MainWindow = QtWidgets.QMainWindow()
    centralWidget = QtWidgets.QWidget(MainWindow)
    vb = QtWidgets.QVBoxLayout()
    vb.addWidget(plot.gl)
    centralWidget.setLayout(vb)
    MainWindow.setCentralWidget(centralWidget)
    MainWindow.updateGeometry()
    MainWindow.show()

    app.exec_()
    

def simulate_sequence(title, instrset, descset, stop=910001, engine=0, acmode=False):
    """
    Simulates executing the given instruction set using classes from seqplot.py.

    Parameters
    ----------
    title : str
        The title or name of the sequence simulation.
    instrset : list
        A list of instructions to be executed in the simulation.
    descset : list
        A list of descriptions corresponding to each instruction.
    stop : int, optional
        the number of frames to simulate
    acmode : bool, optional
        if following Fixed rate or AC marker  (won't simulate both)
    
    Returns
    -------
    the sequence object after execution
        
    """
    seq = SeqUser(start=0,stop=stop,acmode=acmode)
    seq.execute(title,instrset,descset)
    ydata = np.array(seq.ydata)+int(engine)*4+272
    return (seq.xdata, ydata)

def sequence_stats(data:tuple, event_defs=None):
    """
    Gives basic stats about the generated sequence to confirm the
    correct number of triggers are generated.

    For now just list event code counts assuming 1 TPG second of frames.
    """
    frame_number, event_codes = data
    event_code_stats = {}

    # Get unique y values and their counts
    unique_codes, counts = np.unique(event_codes, return_counts=True)
    sorted_indices = np.argsort(unique_codes)
    unique_codes = unique_codes[sorted_indices]
    counts = counts[sorted_indices] 

    #if event_defs is a list, assume names are given in order
    if event_defs is not None and isinstance(event_defs,list):
        event_dict = {}
        for event_code, event_name in zip(unique_codes, event_defs): 
            event_dict[int(event_code)] = event_name
        event_defs = event_dict

    # generate a dict for each event code, add counts and event name 
    for value, count in zip(unique_codes, counts):
        event_name = "NA" if event_defs is None else event_defs[value]
        event_code_stats[int(value)] = {"count":int(count), "event_name":event_name}

    # get first instance of each marker, add to dict
    for xi, yi in zip(frame_number, unique_codes):
        if yi not in event_code_stats:
            # (counts, first instance)
            event_code_stats[yi]["start_frame"] = xi

    return event_code_stats


def test_nc_base_sequences():
    """ Test that the NC Base sequences are simulated properly."""
    base_sequence_instrset = xpm_prog.make_base_sequence_nc() 
    event_names = ['70kH', '35kH', '100H', '5H']
    data = simulate_sequence(
        "NC Base Sequence", 
        base_sequence_instrset,
        event_names,
        stop=910000,
        engine=0,
        acmode=False
        )
    pprint(sequence_stats(data, event_names))
    # plot_seq(data)

def test_nc_sequences():
    """ 
    Test base rate trigger for the laser
    Note: the initial ACRateSync is not simulated here, 
    as that is not a feature of the simulation library I am using
    """
    from opcpa_tpr_config import xpm_prog

    #some definitions
    event_names = ["On Time", "Goose", "All Shots"]

    #iterate through all rates and offsets
    all_rates =  xpm_prog.make_base_rates(xpm_prog.nc_factors)
    for rate in all_rates:
        base_div = 120//rate
        goose_rates =  xpm_prog.allowed_goose_rates(rate,all_rates)
        goose_rates.append(None) # also check no goose
        for goose_rate in goose_rates:
            if goose_rate is not None:
                goose_div = 120//goose_rate
            else:
                goose_div = None
            for start_ts1 in [True, False]:
                nc_sequence_instrset = xpm_prog.make_sequence_nc(base_div, start_ts1, goose_div) 
                data = simulate_sequence("NC Sequence", nc_sequence_instrset, event_names , stop=361, engine=0, acmode=True)
                print(f"\nNC Sequence Rate: {rate} Hz, Goose Rate: {goose_rate}  TS1_start: {start_ts1}")
                pprint(sequence_stats(data, event_names))
                #plot select sequences if here if needed
                if rate == 120 and goose_rate is None:
                    plot_seq(data)



if __name__ == "__main__":
    test_nc_base_sequences()
    test_nc_sequences()
    #print some choice AC sequences


    