import time

import numpy as np
import cv2
from qvl.qbot_platform import QLabsQBotPlatform

from sc_helper import QBotPlatformDriver, Keyboard
import sc_setup


def stop_qbot(qbot: QBotPlatformDriver, comm_dt: float = 0.1) -> None:
    # NOTE(mabay): This is hacky but these sequence of commands seem necessary
    # to fully stop the qbot from moving in sim.
    qbot.read_write_std(timestamp=elapsed_time(), arm=0, commands=[0, 0])
    time.sleep(comm_dt)
    qbot.read_write_std(timestamp=elapsed_time(), arm=1, commands=[0, 0])
    time.sleep(comm_dt)
    qbot.read_write_std(timestamp=elapsed_time(), arm=0, commands=[0, 0])

def compute_time_from_pos(
    qbot: QLabsQBotPlatform, target_pos: tuple[float, float],
    qb_v: float, qb_w: float
) -> tuple[float, float]:
    """
    Computes the amount of time required to execute a sequence
    of moves that will allow a qbot to fix its heading (step 1)
    and go to its target position (step 2)

    Returns:
        tuple[float, float]: time to move along heading (t_v) AND time to fix heading (t_v)
    """
    _, pos, rot, _ = qbot.get_world_transform()
    pos_xy = np.array(pos[:2])
    rot_z  = rot[2]
    target = np.array(target_pos)
    d_pos = target - pos_xy

    # Step 1: Fix heading
    target_heading = np.arctan2(d_pos[1], d_pos[0])
    print(target_heading)
    d_angle = target_heading - rot_z
    out_tw = d_angle / qb_w

    # Step 2: Travel along heading
    out_tv = np.linalg.norm(d_pos) / qb_v

    return (out_tv, out_tw)


def wait_until_taret_angle(target_heading, angle_eps, ask_dt=0.05):
    while True:
        _, _, rot, _ = qbot_obj.get_world_transform()
        if np.abs(target_heading - rot[2]) <= angle_eps:
            print("Done rotating.")
            break
            
        time.sleep(ask_dt)
        
def wait_until_target_pos(target_pos, dist_eps, ask_dt=0.05):
    while True:
        _, loc, _, _ = qbot_obj.get_world_transform()
        xy = loc[:2]
        if np.abs(np.linalg.norm(target_pos) - np.linalg.norm(xy)) <= dist_eps:
            print("Done moving.")
            break

        time.sleep(ask_dt)

qbot_obj = sc_setup.setup_boilerplate(locationQBotP=[0, 0, 0.05], verbose=True)
ipDriver = 'localhost'
startTime = time.time()
def elapsed_time():
    return time.time() - startTime


target = np.array((5, -2))
V, W = 0.3, -0.3
t_v, t_w = compute_time_from_pos(qbot_obj, (3, 3), V, W)
print(f"Assigned values: (v = {V}, w = {W})")
print(f"Computed values: (t_v = {t_v}, t_w = {t_w})")

try:
    # Initialization
    myQBot       = QBotPlatformDriver(mode=1, ip=ipDriver)
    keyboard     = Keyboard()

    startTime = time.time()
    time.sleep(0.5)

    print("Initial:", qbot_obj.get_world_transform())

    myQBot.read_write_std(timestamp=elapsed_time(), arm=1, commands=[0, W])
    angle_eps = np.deg2rad(0.5)
    target_heading = np.arctan2(target[1], target[0])
    wait_until_taret_angle(target_heading, angle_eps)

    print("After t_w:", qbot_obj.get_world_transform())
    stop_qbot(myQBot)

    myQBot.read_write_std(timestamp=elapsed_time(), arm=1, commands=[V, 0])
    dist_eps = 0.01
    wait_until_target_pos(target, dist_eps)

    print("After t_v:", qbot_obj.get_world_transform())
    stop_qbot(myQBot)
    print("Final:", qbot_obj.get_world_transform())


except KeyboardInterrupt:
    print('User interrupted.')

finally:
    # Termination
    myQBot.terminate()
    keyboard.terminate()