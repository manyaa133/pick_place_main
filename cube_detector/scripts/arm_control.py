"""
arm_control.py
Sends JSON commands to the RoArm M2 Pro over HTTP.
"""
import requests
import json

ARM_IP = "10.100.15.139"

def send_command(cmd_dict, timeout=3):
    url = f"http://{ARM_IP}/js?json={json.dumps(cmd_dict)}"
    try:
        r = requests.get(url, timeout=timeout)
        return r.text
    except requests.exceptions.RequestException as e:
        print(f"ERROR sending command: {e}")
        return None

def move_to(x, y, z, t=3.14):
    return send_command({"T": 1041, "x": x, "y": y, "z": z, "t": t})

def gripper(angle, spd=0, acc=10):
    return send_command({"T": 106, "cmd": angle, "spd": spd, "acc": acc})

def gripper_open():
    return gripper(1.08)

def gripper_close():
    try:
        return gripper(3.14)  # <-- set this to your actual "closed" angle
    except Exception as e:
        print(f"error: {e}")
        return None