#!/usr/bin/env python3
"""
Ambulance Rescue Client
Students modify ONLY the my_solution() function
"""

import socket
import argparse


def read_data(data_text):
    """Parse the problem data"""

    class Person:
        def __init__(self, x, y, st, pid):
            self.x = x
            self.y = y
            self.st = st
            self.pid = pid

    class Hospital:
        def __init__(self, num_amb, hid):
            self.num_amb = num_amb
            self.hid = hid

    persons = []
    hospitals = []
    mode = 0
    pid = 0
    hid = 0

    for line in data_text.strip().split("\n"):
        line = line.strip().lower()
        if "person" in line:
            mode = 1
        elif "hospital" in line:
            mode = 2
        elif line:
            if mode == 1:
                parts = line.split(",")
                if len(parts) == 3:
                    pid += 1
                    x, y, t = map(int, parts)
                    persons.append(Person(x, y, t, pid))
            elif mode == 2:
                hid += 1
                num = int(line)
                hospitals.append(Hospital(num, hid))

    return persons, hospitals


def my_solution(pers, hosps):
    ## Your solution goes here!
    """
    Parameters:
    -----------
    pers : list of Person objects
        Each person has:
        - .pid (person ID, 1-indexed)
        - .x (x coordinate)
        - .y (y coordinate)
        - .st (survival time - minutes until they must be at hospital)

    hosps : list of Hospital objects
        Each hospital has:
        - .hid (hospital ID, 1-indexed)
        - .num_amb (number of ambulances)

    Return:
    -------
    A string containing the solution in the same format as in validator_str.py
    (hospitals first, then ambulances), ending with a newline.
    Here is a very simple example that shows the format.
    """

    # This will store the final solution string
    solution_string = ""

    # Step 1: Place hospitals at simple, fixed locations.
    hospital_locations = [(50, 50), (100, 50), (50, 150), (100, 150), (75, 250)]
    for i, hosp in enumerate(hosps):
        # Use modulo to handle cases with more hospitals than predefined locations
        loc_index = i % len(hospital_locations)
        hosp.x = hospital_locations[loc_index][0]
        hosp.y = hospital_locations[loc_index][1]
        solution_string += f"Hospital:{hosp.x},{hosp.y},{hosp.num_amb}\n"

    solution_string += "\n"

    # Keep track of people who have been assigned a rescue.
    rescued_pids = set()

    # Sort people by survival time to rescue the most urgent cases first.
    sorted_pers = sorted(pers, key=lambda p: p.st)

    # Step 2: For each hospital, use one ambulance to rescue the first possible person.
    for hosp in hosps:
        # Check if the hospital has any ambulances.
        if hosp.num_amb > 0:
            # Find the first person this ambulance can rescue.
            for person in sorted_pers:
                if person.pid not in rescued_pids:
                    # Calculate time for a simple round trip: hospital -> person -> hospital.
                    distance = abs(hosp.x - person.x) + abs(hosp.y - person.y)
                    # Time includes travel to person, loading (1 min), travel back, and unloading (1 min).
                    time_needed = distance + 1 + distance + 1

                    # If the person can be saved in time, create the route.
                    if time_needed <= person.st:
                        route = f"Ambulance: {hosp.hid}: ({hosp.x},{hosp.y}), {person.pid}: ({person.x},{person.y},{person.st}), {hosp.hid}: ({hosp.x},{hosp.y})\n"
                        solution_string += route
                        rescued_pids.add(person.pid)

                        # This hospital's ambulance is now assigned, so move to the next hospital.
                        break

    return solution_string


def run_play(name, host, port):
    """Connect to server and play one game"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to server
        sock.connect((host, port))

        # Read greeting and timeout
        greeting = sock.recv(1024).decode("utf-8")
        if not "CONNECTED" in greeting:
            raise RuntimeError(f"Unexpected greeting: {greeting}")

        # Extract timeout value from greeting (format: "CONNECTED Ambulance Server\nTIMEOUT <value>\n")
        lines = greeting.strip().split("\n")
        timeout = 60  # Default fallback
        for line in lines:
            if line.startswith("TIMEOUT"):
                timeout = int(line.split()[1])
                break

        # Set socket timeout
        sock.settimeout(timeout + 5)  # Add small buffer for network delays

        # Send PLAY command
        sock.send(f"PLAY {name}\n".encode("utf-8"))

        # Read problem data until SEND_SOLUTION
        data = b""
        while b"SEND_SOLUTION" not in data:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk

        # Extract problem data (everything before SEND_SOLUTION)
        problem_text = data.decode("utf-8").split("SEND_SOLUTION")[0].strip()

        # Parse problem
        persons, hospitals = read_data(problem_text)
        print(f"Problem loaded: {len(persons)} people, {len(hospitals)} hospitals")
        print(f"Time limit: {timeout} seconds")

        # Run student's solution with timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Solution timed out")

        # Set up timeout handler (Unix-based systems)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                solution = my_solution(persons, hospitals)
                signal.alarm(0)  # Cancel the alarm
            except TimeoutError:
                print(f"\n{'='*50}")
                print(f"Timeout! Score: 0")
                print(f"{'='*50}")
                sock.close()
                return False
        else:
            # For Windows or systems without SIGALRM, use threading
            import threading

            result = [None]
            exception = [None]

            def run_solution():
                try:
                    result[0] = my_solution(persons, hospitals)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_solution)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                print(f"\n{'='*50}")
                print(f"Timeout! Score: 0")
                print(f"{'='*50}")
                sock.close()
                return False
            elif exception[0]:
                raise exception[0]
            else:
                solution = result[0]

        # Send solution
        sock.send(solution.encode("utf-8"))
        sock.send(b"END_SOLUTION\n")

        # Read verdict
        verdict = sock.recv(1024).decode("utf-8")

        # Parse and display result
        for line in verdict.split("\n"):
            if line.startswith("OK"):
                parts = line.split()
                score = parts[1] if len(parts) > 1 else "?"
                print(f"\n{'='*50}")
                print(f"SUCCESS! Score: {score}")
                print(f"{'='*50}")
            elif line.startswith("NOT_OK"):
                print(f"\n{'='*50}")
                print("FAILED: Solution was invalid")
                print(f"{'='*50}")
            elif line.strip():
                print(line)

        sock.close()
        return True

    except socket.timeout:
        print(f"\n{'='*50}")
        print(f"Timeout! Score: 0")
        print(f"{'='*50}")
        sock.close()
        return False
    except Exception as e:
        print(f"Error: {e}")
        sock.close()
        return False


def run_results(host, port):
    """Get leaderboard from server"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)

    try:
        sock.connect((host, port))

        # Read greeting and timeout info
        greeting = sock.recv(1024).decode("utf-8")

        # Send RESULTS command
        sock.send(b"RESULTS\n")

        # Read results
        data = b""
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk
            if b"DISCONNECTED" in data:
                break

        # Display results
        result_text = data.decode("utf-8")
        print("\n" + "=" * 50)
        print("LEADERBOARD")
        print("=" * 50)

        in_results = False
        for line in result_text.split("\n"):
            if "RESULTS_BEGIN" in line:
                in_results = True
            elif "END" in line:
                in_results = False
            elif in_results and line.strip():
                print(line)

        print("=" * 50)
        sock.close()
        return True

    except Exception as e:
        print(f"Error: {e}")
        sock.close()
        return False


def test_local(data_file="data.txt"):
    """Test solution locally"""
    try:
        # Read data file
        with open(data_file, "r") as f:
            data_text = f.read()

        # Parse problem
        persons, hospitals = read_data(data_text)
        print(f"Problem: {len(persons)} people, {len(hospitals)} hospitals")

        # Run solution
        solution = my_solution(persons, hospitals)

        # Display solution
        print("\nGenerated solution:")
        print("-" * 40)
        lines = solution.split("\n")
        for i, line in enumerate(lines[:20]):  # Show first 20 lines
            print(line)
        if len(lines) > 20:
            print(f"... ({len(lines)-20} more lines)")
        print("-" * 40)

        # Save to file for inspection
        with open("test_solution.txt", "w") as f:
            f.write(solution)
        print("\nSolution saved to test_solution.txt")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ambulance Rescue Client")
    parser.add_argument("--name", help="Player name for submission")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--results", action="store_true", help="Get leaderboard")
    parser.add_argument("--test", action="store_true", help="Test locally")

    args = parser.parse_args()

    if args.test:
        test_local()
    elif args.results:
        run_results(args.host, args.port)
    elif args.name:
        run_play(args.name, args.host, args.port)
    else:
        print("Usage:")
        print("  Test locally:  python client.py --test")
        print("  Submit:        python client.py --name YourName --port 5555")
        print("  Leaderboard:   python client.py --results --port 5555")
