import zmq
from jetracer.nvidia_racecar import NvidiaRacecar
import signal
import sys

# --- CONFIG ---
ZMQ_PORT = 5555
THRESHOLD_CIBLE = 0.70
THRESHOLD_NOCIBLE = 0.40
SPEED_NORMAL = 0.14

# Setup Car
car = NvidiaRacecar()

# Safety: Stop car on exit
def signal_handler(sig, frame):
    print("\n[Control] Stopping Robot...")
    car.throttle = 0.0
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topic

    print("[Control] Controller connected. Waiting for Vision data...")

    while True:
        # 1. Get Data (Blocking call - syncs logic with frame rate)
        data = socket.recv_json()

        prob = data['prob_target']

        # 2. Your Logic
        print(f"Target Probability: {prob:.4f}", end="\r")

        if prob > THRESHOLD_CIBLE:
            car.throttle = 0.0
        elif prob < THRESHOLD_NOCIBLE:
            car.throttle = SPEED_NORMAL

if __name__ == "__main__":
    main()
