import argparse
import itertools

import numpy as np
from psdaq.cas.pvedit import Pv
from psdaq.seq.seq import Branch, ControlRequest, FixedRateSync, ACRateSync, acRateHzToMarker
from psdaq.seq.seqprogram import SeqUser

factors = [2, 2, 2, 2, 5, 5, 5, 5, 7, 13]  # 910,000
sc_factors = [2, 2, 2, 5, 5, 5, 5, 7]  # 35000 (remove 13, 2, add 1)
nc_factors = [2, 2, 2, 3, 5]# 120Hz  (independent of 910k factors)


def make_base_rates(laser_factors):
    """
    Generate a list of factors --> rates (TPG second)
    """
    iters = [
        itertools.combinations(
            laser_factors, i+1
        ) for i in range(len(laser_factors))
    ]
    f = set()  # Unique set of factors; trims out duplicates
    for i in iters:
        for c in i:
            q = np.prod(np.array(c))
            f.add(q)

    return sorted(list(f))


def allowed_goose_rates(base_rate, rate_list):
    """
    Return a dict of allowed goose rates, based on the base rate of the laser
    and a dictionary of allowed base rates created using make_base_rates().
    """

    return [rate for rate in rate_list if rate < base_rate]


# Selected base rate + goose rate --> pulse sequence
def make_sequence_sc(base_div, goose_div=None, offset=None, debug=False):
    # Do some setup
    instrset = []

    # Insert bucket offset if it is present
    if offset is not None and offset != 0:
        instrset.append(FixedRateSync(marker="910kH", occ=offset))
        if debug:
            print(f"FixedRateSync(marker=\"910kH\", occ={offset})")

    # If we're goosing, put that pulse in _first_ because it makes delay
    # management easier
    if goose_div not in (None, 0):  # Start with goose
        instrset.append(ControlRequest([1, 2]))
        instrset.append(FixedRateSync(marker="910kH", occ=base_div))
        if debug:
            print("ControlRequest([1, 2])")
            print(f"FixedRateSync(marker=\"910kH\", occ={base_div})")

    # Loop over base:goose rate ratio once. Because we're using divisors,
    # we divide goose divider by base divider, rather than base rate by
    # goose rate.
    if goose_div in (None, 0):
        n = 1
    else:
        n = (goose_div//base_div) - 1
    for i in range(n):
        instrset.append(ControlRequest([0, 2]))
        instrset.append(FixedRateSync(marker="910kH", occ=base_div))
        if debug:
            print("ControlRequest([0, 2])")
            print(f"FixedRateSync(marker=\"910kH\", occ={base_div})")

    # Change branching based on offset
    if offset is not None and offset != 0:
        if debug:
            print("Branch.unconditional(1)")
        instrset.append(Branch.unconditional(1))
    else:
        if debug:
            print("Branch.unconditional(0)")
        instrset.append(Branch.unconditional(0))

    return instrset

def make_sequence_nc(base_div, start_ts1 = True, goose_div=None, debug=False):
    """
    Generate an AC sequence at spacing of base_div times the base (120hz) rate for
    NC operation. 

    NOTE:  AC base rates always need to specify a timeslot. 
           There are 6 time slots each with 60H rate markers.

    Parameters
    ----------
    base_div
       divisor in units of beams (120hz)
       1 => full rate 120HZ, 2 for 60Hz, 3 for 40Hz, etc.
       rate = 120 / base_div
    start_ts1, optional
        if True, start synced to TS1, else TS4
        Only changes pattern if rate is a sub-harmonic of 60 Hz
    goose_div, optional
        Goose trigger divisor (same units as base_div)
        Must be integer multiple of base_div!
    debug, optional
        add print statements, by default False

    Notes
    -------
    
    Confluence Docs are inconsistent with the library definitions used here! 
    Confluence examples show that "marker 0" corresponds to 60H rate,
    but in the acRateHzToMarker dictionary in seq.py, clearly maps "60H" maps to "marker 5".

    I used the library definitions here for correct simulations
    but this should be verified on hardware!
    """
 
    # Do some setup
    instrset = []
    
    fiducial_marker = acRateHzToMarker["60Hz"] 
   
    # sync the fist shot
    if start_ts1:
        timeslot_mask = (1<<0) 
    else:
        timeslot_mask = (1<<3)
    instrset.append(ACRateSync(timeslotm=timeslot_mask, marker=fiducial_marker, occ=1))

    # Linac only fires on TS 1 and 4
    timeslot_mask = (1<<0) | (1<<3)

    branch_0 = len(instrset)
    if goose_div not in (None, 0):
        # goosing
        ontime_per_goose= (goose_div//base_div) - 1 
        # goosing shot is first, then # of ontime shots per goose
        instrset.append(ControlRequest([1, 2])) # goose + all
        instrset.append(ACRateSync(timeslotm=timeslot_mask, marker=fiducial_marker, occ=base_div))
        for i in range(ontime_per_goose):
            instrset.append(ControlRequest([0, 2])) # on_time +all
            instrset.append(ACRateSync(timeslotm=timeslot_mask, marker=fiducial_marker, occ=base_div))
        instrset.append(Branch.unconditional(line=branch_0))
    else:
        # no goose, only on_time shots
        instrset.append(ControlRequest([0,2])) #on_time + all
        instrset.append(ACRateSync(timeslotm=timeslot_mask, marker=fiducial_marker, occ=base_div))
        instrset.append(Branch.unconditional(line=branch_0))

    if debug:
        for instr in instrset:
            print(instr.print_())
        
    return instrset

def make_base_sequence(offset=None):
    """
    Setup base rate sequences always needed to operate the laser system.

    Rates: AC Rate 71,428, 35,714, 102, 5.1 Hz
            70k, 35k, 100, 5 (TPG seconds)

    Notes
    ----

    The AC power line is sampled at 2/14Mhz = 35,714 Hz 
    to guaranteed all AC crossings coincide every two 71428 Hz markers.

    To ensure we start at the correct phase, we need to sync to the AC crossing
    before starting the 71428 Hz fixed rate sequence.

    """
    # initialize instruction set array
    instrset = []


    # # Insert bucket offset if it is present (only possible in sc timing)
    # if offset is not None and offset != 0:
    #     instrset.append(FixedRateSync(marker="910kH", occ=offset))
    # else:
    #     # first AC Sync to linac to ensure in phase
    #     # this may not be needed depending on how AC markers are sampled
    #     fiducial_marker = acRateHzToMarker["60Hz"] 
    #     # Linac only fires on TS 1 and 4
    #     timeslot_mask = (1<<0) | (1<<3)
    #     instrset.append(ACRateSync(timeslotm=timeslot_mask, marker=fiducial_marker, occ=1))

    branch_0 = len(instrset)
    instrset.append(ControlRequest([0, 1, 2, 3])) # 70kH + 35kH + 100H + 5H
    instrset.append(FixedRateSync(marker="70kH", occ=1))

    branch_1 = len(instrset)
    _add_inner_sequence(instrset)

    ## Loop 2: 100H markers per 5H marker
    spacing_100 = 700
    spacing_5 = 14000
    loop_count = spacing_5 // spacing_100 - 2
    instrset.append(Branch.conditional(line=branch_1, counter=1, value=loop_count))
    # again last iteration has to be done manually
    _add_inner_sequence(instrset , final=True)

    instrset.append(Branch.unconditional(line=branch_0))

    return instrset

def _add_inner_sequence(instrset: list, final=False):
    """
    To make the nc sequence a little more readable, 
    a repeated section is broken out here

    Loop 1:
    75kH + 35kH markers per 100H marker 
    
    final = true, do not add 100H last marker
    """
    # define all spacings in terms of smallest marker
    # spacing_70k = 1
    spacing_35k = 2
    spacing_100 = 700

    loop_count = spacing_100//spacing_35k - 2 
    branch = len(instrset)
    instrset.append(ControlRequest([0])) # 70kH 
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    instrset.append(ControlRequest([0, 1])) # 70kH + 35kH
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    instrset.append(Branch.conditional(line=branch, counter=0, value=loop_count))
    # last iteration done manually since last iteration is different
    instrset.append(ControlRequest([0])) # 70kH 
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    if not final:
        instrset.append(ControlRequest([0, 1, 2])) # 70kH, 35kH, 100H 
        instrset.append(FixedRateSync(marker="70kH", occ=1))

    return instrset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "base_rate",
        help="Desired laser output rep rate (total)"
    )
    parser.add_argument(
        "goose_rate",
        help="Desired laser goose rate (sub-harmonic of base_rate)")
    parser.add_argument("offset", help="Desired 910 kHz bucket offset")
    parser.add_argument("bay", help="Laser bay to program for (2 or 3)")

    args = parser.parse_args()

    base_rate = int(args.base_rate)
    goose_rate = int(args.goose_rate)
    offset = int(args.offset)

    allowed_bays = [2, 3]
    if int(args.bay) not in allowed_bays:
        raise ValueError(f"Bay {args.bay} not in {allowed_bays}!")
    else:
        bay = int(args.bay)

    engines = {2: 6, 3: 7}  # Bay --> sequence engine mapping

    # Dict will eventually be applied to drop down menu
    base_list = make_base_rates(sc_factors)

    if base_rate not in base_list:
        raise ValueError(
           ("Base rate {base_rate} is not one of the available laser "
            f"rates: {base_list}")
        )

    # Dict will eventually be applied to drop down menu
    goose_list = allowed_goose_rates(base_rate, base_list)

    if goose_rate not in goose_list:
        raise ValueError(
           (f"Goose rate {goose_rate} is not one of the available goose "
            f"rates: {goose_list}")
        )

    seqdesc = {0: f"Bay {bay} On Time", 1: f"Bay {bay} Off Time", 2: "", 3: ""}
    base_div = 910000//int(base_rate)
    goose_div = 910000//int(goose_rate)
    inst = make_sequence_sc(base_div, goose_div, offset, True)

    xpm_pv = "DAQ:NEH:XPM:0"
    seqcodes_pv = Pv(f'{xpm_pv}:SEQCODES', isStruct=True)
    seqcodes = seqcodes_pv.get()
    desc = seqcodes.value.Description

    engine = int(engines[bay])
    seq = SeqUser(f'{xpm_pv}:SEQENG:{engine}')
    seq.execute('title', inst, None, sync=True, refresh=False)

    engineMask = 0
    engineMask |= (1 << engine)

    for e in range(4*engine, 4*engine+4):
        desc[e] = ''
    for e, d in seqdesc.items():
        desc[4*engine+e] = d

    tmo = 5.0  # epics pva timeout

    v = seqcodes.value
    v.Description = desc
    seqcodes.value = v
    seqcodes_pv.put(seqcodes, wait=tmo)

    pvSeqReset = Pv(f'{xpm_pv}:SeqReset')
    pvSeqReset.put(engineMask, wait=tmo)
