import argparse
import itertools

import numpy as np
from psdaq.cas.pvedit import Pv
from psdaq.seq.seq import Branch, ControlRequest, FixedRateSync, ACRateSync, acRateHzToMarker
from psdaq.seq.seqprogram import SeqUser

factors = [2, 2, 2, 2, 5, 5, 5, 5, 7, 13]  # 910,000
carbide_factors = [1, 2, 2, 5, 5, 5, 5, 13]  # 32,500 (remove 2, 2, 7, add 1)


def make_base_rates(laser_factors):
    """
    Generate a dictionary of factors --> rates (TPG second)
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

    NOTE: AC baser rates are in terms of every timeslot. Sync instructions must specify
    a timeslot mask and a rate. There are 6 time slots each with sub 60H rate markers.

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

def make_base_sequence_nc():
    """
    Setup lase base rate sequences needed for running NC mode.

    Rates: Full AC Rate 71,428, 35,714 

    Notes:
    ----

    The AC power line is sampled at 1/14Mhz = 71428 Hz, 
    so all AC crossings (LCLS 1 fiducials) are guaranteed to line 
    up with the 71428 Hz markers.

    For carbide lasers we want to run at 35714 Hz subharmonic of 71428 Hz, 
    having only 50% overlap with the AC crossings.

    My understanding is that there is no guaranteed pattern that ensures 
    the 35714 Hz rates will line up with the AC crossings every time,
    because the grid voltage is not phase locked to 
    the master occilator.

    Therefore we need to "resync" to every AC marker available (360hz)
    such that the 35714 Hz shots always line up with the AC crossings.

    We might need to include more rates after disusing with lasers
 
    The simulator is unable to mix ac and fixed rate commands so
    this is not tested yet.
    ( simulations show either 910000 buckets or 
    360 buckets frames but not at the same time)

    """
    # initialize instruction set array
    instrset = []
    fiducial_marker = acRateHzToMarker["60Hz"] 
    timeslot_mask = (1<<0) | (1<<1) | (1<<2) | (1<<3) # all timeslots


    # No need to worry about starting offsets 
    branch_0 = len(instrset)
    # first sync to any of the AC crossings 
    # (we need to trigger the 35khz we don't miss xray shots)
    instrset.append(ACRateSync(timeslotm=timeslot_mask, marker=fiducial_marker, occ=1))
    instrset.append(ControlRequest([0, 1])) # 70kH + 35kH
    # after the synced AC crossing, there should be at least 198, 70KH markers
    # and 99 35kH markers before the next AC crossing
    #     (1/360) / (1/71,428) = ~198.4111    => / 2 = ~99.2055
    
    # 70k marker only 
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    instrset.append(ControlRequest([0])) # 70kH only

    # loop 98 times * 2 70 markerks = 196 + initial one = 197 markers)
    branch_1 = len(instrset)
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    instrset.append(ControlRequest([0, 1])) # 70kH + 35kH
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    instrset.append(ControlRequest([0])) # 70kH only
    instrset.append(Branch.conditional(line=branch_1, counter=0, value=98))

    # last 70k marker before next AC sync == 198th marker 
    # might need to take this line out (see next comment!)
    instrset.append(FixedRateSync(marker="70kH", occ=1))
    instrset.append(ControlRequest([0])) # 70kH only

    #NOTE I am forcing the next AC sync
    #  depending on how the AC is sampled by the sep down chassis
    #  the sequence is either 1 or 2 70khz periods from from the AC crossing trigger 
    #  This  will introduce jitter in the 35kHz rate but ensures we don't miss shots.
    # I don't know if there is a better way to handle this with the current hardware
    instrset.append(Branch.unconditional(line=branch_0))
    return instrset

def make_base_sequence_sc(offset=None):
    """
    Setup standard sequence of full rate, 32500, 100, and 5 Hz codes.
    """
    # Do some setup
    instrset = []

    # Insert bucket offset if it is present
    if offset is not None and offset != 0:
        instrset.append(FixedRateSync(marker="910kH", occ=offset))

    b0 = len(instrset)
    instrset.append(ControlRequest([0, 1, 2, 3]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    b1 = len(instrset)
    instrset.append(ControlRequest([0]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    instrset.append(Branch.conditional(line=b1, counter=0, value=26))
    b2 = len(instrset)
    instrset.append(ControlRequest([0, 1]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    b3 = len(instrset)
    instrset.append(ControlRequest([0]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    instrset.append(Branch.conditional(line=b3, counter=0, value=26))
    instrset.append(Branch.conditional(line=b2, counter=1, value=323))
    b4 = len(instrset)
    instrset.append(ControlRequest([0, 1, 2]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    b5 = len(instrset)
    instrset.append(ControlRequest([0]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    instrset.append(Branch.conditional(line=b5, counter=0, value=26))
    b6 = len(instrset)
    instrset.append(ControlRequest([0, 1]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    b7 = len(instrset)
    instrset.append(ControlRequest([0]))
    instrset.append(FixedRateSync(marker="910kH", occ=1))
    instrset.append(Branch.conditional(line=b7, counter=0, value=26))
    instrset.append(Branch.conditional(line=b6, counter=1, value=323))
    instrset.append(Branch.conditional(line=b4, counter=2, value=18))
    instrset.append(Branch.unconditional(line=b0))

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
    base_list = make_base_rates(sc_carbide_factors)

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
