import requests
import json
import time

ARM_IP = "10.100.14.12"

def send_command(cmd_dict):
    url = f"http://{ARM_IP}/js?json={json.dumps(cmd_dict)}"
    print(f"Sending: {url}")
    r = requests.get(url, timeout=5)
    print(f"Response: {r.text}")

send_command({"T": 405})
send_command({"T": 1041, "x": 235, "y": 0, "z": 234, "t": 3.14})
send_command({"T": 1041, "x": 150, "y": 100, "z": 300, "t": 3.14})
send_command({"T": 106, "cmd": 3.14, "spd": 0, "acc": 10})
send_command({"T": 106, "cmd": 1.08, "spd": 0, "acc": 10})
time.sleep(2)
send_command({"T": 106, "cmd": 3.14, "spd": 0, "acc": 10})
send_command({"T": 106, "cmd": 3.14, "spd": 0, "acc": 10})