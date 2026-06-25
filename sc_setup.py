#-----------------------------------------------------------------------------#
#--------------------Quanser Interactive Labs Setup for-----------------------#
#---------------------------Mobile Robotics Lab-------------------------------#
#-----------------(Environment: QBot Platform / Warehouse)--------------------#
#-----------------------------------------------------------------------------#

from quanser.communications.exceptions import StreamError
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qbot_platform import QLabsQBotPlatform
from qvl.qbot_platform_flooring import QLabsQBotPlatformFlooring
from qvl.multi_agent import MultiAgent
import pal.resources.rtmodels as rtmodels
import time
import numpy as np
import sys
import subprocess

import numpy as np
from numpy.typing import NDArray

#------------------------------ Main program ----------------------------------
def our_floors_setup(qlabs: QuanserInteractiveLabs, top_left, row, col, verbose: bool) -> None:
    """
    Parameters
    ----------
    center: array of size (2,)\n
    row, col: int
    """

    rows, cols = 5, 5
    if verbose: print(f"Spawning grid of floors (r={rows}, c={cols})")
    floor = QLabsQBotPlatformFlooring(qlabs)

    count = 0
    for r in range(0, rows+1):
        for c in range(0, cols+1):
            floor.spawn_id_degrees(
                actorNumber=count,
                location=np.array([r, c, 0]) * 1.2 + top_left,
                rotation=np.zeros((3)),
                scale=np.ones((3)),
                configuration=count % 6,
                waitForConfirmation=False
            )
            count += 1
        

def our_setup(qlabs: QuanserInteractiveLabs, qbot_loc: list[float], verbose: bool) -> None:
    #---------------------------- QBot Platform ---------------------------
    if verbose: print("Spawning QBot Platform ...")
    our_floors_setup(qlabs, verbose)

    qbot = QLabsQBotPlatform(qlabs)
    pos = np.array(qbot_loc)
    qbot.spawn_degrees(
        location=pos, rotation=[0,0,90], scale=np.ones(3),
        configuration=0, waitForConfirmation=True
    )

    return qbot

def setup_multiagent(locs: NDArray[np.float64], verbose: bool) -> MultiAgent:
    assert locs.shape[1] >= 2, "Each position element must have at least 2 components (x, y)"

    if verbose: print("Spawning QBot Platform ...")
    robot_defs = [
        {
            "RobotType": "QBP",
            "Location": [loc[0], loc[1], 0.1],
            "Rotation": [0, 0, 90],
            "Radians": False,
        }
        for loc in locs
    ]

    return MultiAgent(robot_defs)


def _setup_boilerplate_helper(locations, verbose=True) -> MultiAgent:
    """
    Parameters
    ----------
    locations: a list of points (list of 2D points)
    verbose: flag that determines the amount of info printed
    """

    # subprocess.Popen(['quanser_host_peripheral_client.exe', '-q'])
    # time.sleep(2.0)
    # subprocess.Popen(['quanser_host_peripheral_client.exe', '-uri', 'tcpip://localhost:18444'])

    multiagent = setup_multiagent(locations, verbose)

    if verbose: print('QLabs setup completed')

    # mabay: add a bit of delay to make sure everything is init'd correctly
    time.sleep(1)

    return multiagent

def setup_boilerplate(locations, verbose=True) -> MultiAgent:
    try:
        return _setup_boilerplate_helper(locations, verbose=verbose)
    except StreamError as se:
        if verbose: print(f"Received a stream error with code {se.error_code}...")

        match se.error_code:
            case -64:
                print("Make sure that Quanser simulator is open and running; then try again.")
                sys.exit(1)

            case -41:
                if verbose: print(f"Retrying one more time...")

                # Receiving this error code most likely means that when the quanser sim was last used,
                # the sim was not closed correctly. So, opening and immediately closing might resolve
                # that issue. 
                qlabs = QuanserInteractiveLabs()
                qlabs.open("localhost")
                qlabs.close()
                time.sleep(1)
                return _setup_boilerplate_helper(locations, verbose=verbose)
            
            case _:
                print("Unsure what causes this error.")
                sys.exit(1)
                

if __name__ == '__main__':
    setup_boilerplate(locations=np.array([[0, 0], [1, 1], [-2, 2]]), verbose=True)
    print("\n\nNOTE: This is just the setup file; run the start file instead.\n\n")