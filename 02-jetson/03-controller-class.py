import zmq
from jetracer.nvidia_racecar import NvidiaRacecar
import signal
import sys

# --- CONFIG ---
ZMQ_PORT = 5555

# Probabilities thresholds
THRESHOLD_TARGET = 0.60
THRESHOLD_LOST = 0.30

# Speed settings
SPEED_FORWARD = 0.14
SPEED_TURN = 0.14

STEER_STRAIGHT = 0.0
STEER_SOFT = 0.3
STEER_HARD = 0.6

# Setup Car
car = NvidiaRacecar()

# Safety: Stop car on exit
def signal_handler(sig, frame):
    print("\n[Control] Stopping Robot...")
    car.throttle = 0.0
    car.steering = 0.0
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics

    print("[Control] Controller connected. Waiting for Vision data...")

    while True:
        # 1. Receive prediction data
        data = socket.recv_json()
        probs = data['probs']

        # Extract probabilities
        p_left = probs['left']
        p_center = probs['center']
        p_right = probs['right']

        # Determine boolean states (Is the target visible in this zone?)
        is_left = p_left > THRESHOLD_TARGET
        is_center = p_center > THRESHOLD_TARGET
        is_right = p_right > THRESHOLD_TARGET

        # Debug print to visualize the state logic
        state_str = f"L:{int(is_left)} C:{int(is_center)} R:{int(is_right)}"
        print(f"[{state_str}] Raw: L{p_left:.2f} C{p_center:.2f} R{p_right:.2f}", end="\r")

        # --- CONTROL LOGIC ---

        # CASE 1: Target is HUGE (Close) or PERFECTLY ALIGNED -> Go Straight
        # If all 3 trigger, or just Center, we drive forward.
        if (is_left and is_center and is_right) or (is_center and not is_left and not is_right):
            car.steering = STEER_STRAIGHT
            car.throttle = SPEED_FORWARD

        # CASE 2: Target is drifting (Center + Side) -> Soft Correction
        elif is_center and is_left:
            car.steering = -STEER_SOFT # Turn Left gently
            car.throttle = SPEED_TURN

        elif is_center and is_right:
            car.steering = STEER_SOFT  # Turn Right gently
            car.throttle = SPEED_TURN

        # CASE 3: Target is leaving the frame (Side Only) -> Hard Correction
        elif is_left:
            car.steering = -STEER_HARD # Hard Left
            car.throttle = SPEED_TURN

        elif is_right:
            car.steering = STEER_HARD  # Hard Right
            car.throttle = SPEED_TURN

        # CASE 4: Lost Target -> Stop
        else:
            car.throttle = 0.0
            car.steering = STEER_STRAIGHT

if __name__ == "__main__":
    main()
