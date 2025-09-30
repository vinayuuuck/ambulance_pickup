#!/usr/bin/env python3
"""
Ambulance Rescue Server
Usage: python server.py <port> <data_file> [timeout_seconds]
"""

import socket
import threading
import os
import sys
import tempfile
import shutil
from datetime import datetime


class AmbulanceServer:
    def __init__(self, port, data_file, timeout=30):
        self.port = port
        self.data_file = data_file
        self.timeout = timeout
        self.results = {}
        self.seq_counter = 0
        self.lock = threading.Lock()

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file '{data_file}' not found")

    def send_all(self, sock, data):
        """Send all data to socket"""
        if isinstance(data, str):
            data = data.encode("utf-8")
        sock.sendall(data)

    def recv_line(self, sock):
        """Receive one line from socket"""
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(1)
            if not chunk:
                return None
            data += chunk
        return data.decode("utf-8").strip()

    def recv_until(self, sock, marker):
        """Receive data until marker is found"""
        data = b""
        marker_bytes = marker.encode("utf-8")
        while marker_bytes not in data:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk
        # Split at marker and return the part before it
        if marker_bytes in data:
            return data.split(marker_bytes)[0].decode("utf-8")
        return data.decode("utf-8")

    def load_problem_data(self):
        """Load and return the problem data as string"""
        with open(self.data_file, "r") as f:
            return f.read()

    def validate_solution(self, solution_text):
        """Validate a solution and return score"""
        # Create temp directory for validation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy data file
            shutil.copy(self.data_file, os.path.join(tmpdir, "data.txt"))

            # Write solution
            with open(os.path.join(tmpdir, "result.txt"), "w") as f:
                f.write(solution_text)

            # Create validator script
            # FIX: Replaced the original flawed validator with the robust one from validator.py,
            # with plotting and unnecessary functions removed for server-side execution.
            validator_script = """
import re
import pandas as pd
import sys

# Exception classes
class ValidationError(ValueError):
    pass

class FormatSyntaxError(ValidationError):
    pass

class DataMismatchError(ValidationError):
    pass

class IllegalPlanError(ValidationError):
    pass

#  Person object
PID = 0
class Person:
    def __init__(self, x, y, st):
        global PID
        PID += 1
        self.pid = PID
        self.x = x
        self.y = y
        self.st = st
        self.rescued = False
    def __repr__(self):
        return '%d: (%d,%d,%d)' % (self.pid, self.x, self.y, self.st)

#  Hospital object
HID = 0
class Hospital:
    def __init__(self, x, y, num_amb):
        global HID
        HID += 1
        self.hid = HID
        self.x = x
        self.y = y
        self.num_amb = num_amb
        self.amb_time = [0] * num_amb
    def __repr__(self):
        return '%d: [(%d,%d), %d]' % (self.hid, self.x, self.y, self.num_amb)
    def take_time(self, a, b):
        return abs(a.x - b.x) + abs(a.y - b.y)
    def rescue(self, pers, end_hospital):
        if self.num_amb == 0:
            raise IllegalPlanError('No ambulance left at the hospital: %s' % self)
        if 4 < len(pers):
            raise IllegalPlanError('Cannot rescue more than four people at once: %s' % pers)
        already_rescued = list(filter(lambda p: p.rescued, pers))
        if already_rescued:
            raise IllegalPlanError('Person already rescued: %s' % already_rescued)
        t = 1
        start = self
        for p in pers:
            t += self.take_time(start, p)
            start = p
        t += len(pers)
        t += self.take_time(start, end_hospital)
        self.amb_time.sort(reverse=True)
        for (i, t0) in enumerate(self.amb_time):
            if not list(filter(lambda p: p.st < t0 + t, pers)): break
        else:
            raise IllegalPlanError('Either person cannot make it: %s' % pers)
        self.amb_time[i] += t
        end_hospital.num_amb += 1
        end_hospital.amb_time.append(self.amb_time[i])
        self.num_amb -= 1
        self.amb_time.pop(i)
        for p in pers:
            p.rescued = True
        # print('Rescued:', ' and '.join(map(str, pers)), 'taking', t, '|ended at hospital', end_hospital)
        return pers

def read_data(fname="data.txt"):
    persons = []
    hospitals = []
    mode = 0
    with open(fname, 'r') as fil:
        data = fil.readlines()
    for line in data:
        line = line.strip().lower()
        if line.startswith("person") or line.startswith("people"):
            mode = 1
        elif line.startswith("hospital"):
            mode = 2
        elif line:
            if mode == 1:
                (a, b, c) = map(int, line.split(","))
                persons.append(Person(a, b, c))
            elif mode == 2:
                (c,) = map(int, line.split(","))
                hospitals.append(Hospital(-1, -1, c))
    return persons, hospitals

def readresults(persons, hospitals, fname='result.txt'):
    p1 = re.compile(r'(\\d+\\s*:\\s*\\(\\s*\\d+\\s*,\\s*\\d+(\\s*,\\s*\\d+)?\\s*\\))')
    p2 = re.compile(r'(\\d+)\\s*:\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)')
    p3 = re.compile(r'(\\d+)\\s*:\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)')
    score = 0
    hospital_index = 0
    with open(fname, 'r') as fil:
        data = fil.readlines()
    for line in data:
        line = line.strip().lower()
        if not line:
            continue
        if line.startswith('hospital'):
            (x, y, z) = map(int, line.replace('hospital:', '').split(','))
            if not (x and y):
                raise ValidationError("Hospital coordinates not set: line".format(line))
            if z != hospitals[hospital_index].num_amb:
                raise ValidationError("Hospital's ambulance # does not match input (input={0}, results={1})".format(
                    hospitals[hospital_index].num_amb, z))
            hospitals[hospital_index].x = x
            hospitals[hospital_index].y = y
            hospital_index += 1
            continue
        if not line.startswith('ambulance'):
            continue
        try:
            hos = None
            end_hos = None
            rescue_persons = []
            groups = p1.findall(line)
            groups_length = len(groups)
            for (i, (w, z)) in enumerate(groups):
                m = p2.match(w)
                if m:
                    if i != 0 and i != groups_length - 1:
                        raise FormatSyntaxError('Specify a person now: %r' % line)
                    (a, b, c) = map(int, m.groups())
                    if a <= 0 or len(hospitals) < a:
                        raise FormatSyntaxError('Illegal hospital id: %d' % a)
                    if i == 0:
                        hos = hospitals[a - 1]
                        if hos.x != b or hos.y != c:
                            raise DataMismatchError('Starting Hospital mismatch: %s != %d:%s' % (hos, a, (b, c)))
                    else:
                        end_hos = hospitals[a - 1]
                        if end_hos.x != b or end_hos.y != c:
                            raise DataMismatchError('Ending Hospital mismatch: %s != %d:%s' % (end_hos, a, (b, c)))
                    continue
                m = p3.match(w)
                if m:
                    if i == 0:
                        raise FormatSyntaxError('Specify a hospital first: %r' % line)
                    (a, b, c, d) = map(int, m.groups())
                    if a <= 0 or len(persons) < a:
                        raise FormatSyntaxError('Illegal person id: %d' % a)
                    per = persons[a - 1]
                    if per.x != b or per.y != c or per.st != d:
                        raise DataMismatchError('Person mismatch: %s != %d:%s' % (per, a, (b, c, d)))
                    rescue_persons.append(per)
                    continue
                raise FormatSyntaxError('Expected "n:(x,y)" or "n:(x,y,t)": %r' % line)
            if not hos or not rescue_persons:
                continue
            if not hos or not end_hos:
                raise FormatSyntaxError('Either start hospital or end hospital is not defined: %s' % line)
            pers = hos.rescue(rescue_persons, end_hos)
            score += len(rescue_persons)
        except ValidationError as x:
            # print('!!!', x)
            pass
    print('Total score:', score)
    return score

if __name__ == '__main__':
    try:
        (persons, hospitals) = read_data("data.txt")
        readresults(persons, hospitals)
    except Exception as e:
        print(f"Validator Error: {e}", file=sys.stderr)
        print("Total score: 0")
"""

            validator_path = os.path.join(tmpdir, "validator.py")
            with open(validator_path, "w") as f:
                f.write(validator_script)

            # Run validator
            import subprocess

            try:
                result = subprocess.run(
                    [sys.executable, validator_path],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                # Parse score
                score = 0
                for line in result.stdout.split("\n"):
                    if line.startswith("Total score:"):
                        try:
                            score = int(line.split(":")[1].strip())
                        except:
                            pass

                return True, score

            except Exception as e:
                # print(f"Validator execution failed: {e}")
                return False, 0

    def handle_client(self, client_socket, addr):
        """Handle a client connection"""
        try:
            # Send greeting with timeout info
            self.send_all(
                client_socket, f"CONNECTED Ambulance Server\nTIMEOUT {self.timeout}\n"
            )

            # Read command
            command = self.recv_line(client_socket)
            if not command:
                return

            if command.startswith("PLAY "):
                # Extract player name
                name = command[5:].strip()
                if not name:
                    name = "Anonymous"

                # Send problem data
                problem_data = self.load_problem_data()
                self.send_all(client_socket, problem_data)
                self.send_all(client_socket, "\nSEND_SOLUTION\n")

                # Set timeout for receiving solution
                client_socket.settimeout(
                    self.timeout + 10
                )  # Add buffer for network delays

                try:
                    # Receive solution
                    solution = self.recv_until(client_socket, "END_SOLUTION")

                    # Validate solution
                    success, score = self.validate_solution(solution)

                    # Record attempt
                    with self.lock:
                        if name not in self.results:
                            self.results[name] = []
                        self.results[name].append(
                            {
                                "score": score,
                                "timestamp": datetime.now(),
                                "seq": self.seq_counter,
                            }
                        )
                        self.seq_counter += 1

                    # Send result
                    if success:
                        self.send_all(client_socket, f"OK {score}\n")
                    else:
                        self.send_all(client_socket, "NOT_OK\n")

                    # Log attempt
                    self.log_attempt(name, score, solution)

                except socket.timeout:
                    # Client timed out
                    print(f"Client {name} timed out")
                    with self.lock:
                        if name not in self.results:
                            self.results[name] = []
                        self.results[name].append(
                            {
                                "score": 0,
                                "timestamp": datetime.now(),
                                "seq": self.seq_counter,
                            }
                        )
                        self.seq_counter += 1
                    self.send_all(client_socket, "TIMEOUT 0\n")
                    self.log_attempt(name, 0, "TIMEOUT")

            elif command == "RESULTS":
                # Send leaderboard
                leaderboard = self.format_leaderboard()
                self.send_all(client_socket, leaderboard)

            else:
                self.send_all(client_socket, "ERROR Unknown command\n")

            # Send disconnection message
            self.send_all(client_socket, "DISCONNECTED\n")

        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()

    def log_attempt(self, name, score, solution):
        """Log an attempt to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attempt_{timestamp}_{name.replace(' ', '_')}.txt"

        with open(filename, "w") as f:
            f.write(f"Player: {name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Score: {score}\n")
            f.write("\n--- Solution ---\n")
            f.write(solution)
            f.write("\n")

    def format_leaderboard(self):
        """Format the leaderboard"""
        with self.lock:
            # Get best score for each player
            best_scores = {}
            for name, attempts in self.results.items():
                if attempts:
                    best = max(attempts, key=lambda x: x["score"])
                    best_scores[name] = best

            # Sort by score
            sorted_players = sorted(
                best_scores.items(), key=lambda x: (-x[1]["score"], x[1]["seq"])
            )

            result = "RESULTS_BEGIN\n"
            for name, attempt in sorted_players:
                result += f"{name} {attempt['score']}\n"
            if not sorted_players:
                result += "No submissions yet\n"
            result += "END\n"

            return result

    def start(self):
        """Start the server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("", self.port))
        server_socket.listen(5)

        print(f"Ambulance Server listening on port {self.port}")
        print(f"Data file: {self.data_file}")
        print(f"Time restriction: {self.timeout} seconds")
        print("Press Ctrl+C to stop")

        try:
            while True:
                client_socket, addr = server_socket.accept()
                print(f"Connection from {addr}")

                # Handle each client in a thread
                thread = threading.Thread(
                    target=self.handle_client, args=(client_socket, addr)
                )
                thread.daemon = True
                thread.start()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            server_socket.close()
            self.save_results()

    def save_results(self):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(self.format_leaderboard())

        print(f"Results saved to {filename}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <port> <data_file> [timeout]")
        sys.exit(1)

    port = int(sys.argv[1])
    data_file = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    server = AmbulanceServer(port, data_file, timeout)
    server.start()


if __name__ == "__main__":
    main()
