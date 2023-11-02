import logging
from enum import Enum, auto
from typing import Union, List

import numpy as np
import gwpy as gw
from gwpy.table import GravitySpyTable

import gravyflow as gf

import numpy as np

if __name__ == "__main__":
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Initiate Test")
    
    glitch_times = gf.get_glitch_segments(
        gf.IFO.L1
    )
    print(glitch_times)
    print(len(glitch_times))
    
    glitch_times = gf.get_glitch_segments(
        gf.IFO.L1, glitch_types = gf.GlitchType.KOI_FISH
    )
    print(glitch_times)
    print(len(glitch_times))
        
    glitch_times = gf.get_glitch_segments(
        gf.IFO.L1, glitch_types = [gf.GlitchType.KOI_FISH, gf.GlitchType.HELIX]
    )
    print(glitch_times)
    print(len(glitch_times))