#!/usr/bin/env python3

import socket
import argparse
import random
import heapq


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


def read_data(data_text):
    """Parse the problem data"""

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


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def k_medians(points, k, max_iter=100):
    if k <= 0:
        return []
    n = len(points)
    seeds = [(points[0].x, points[0].y)]
    while len(seeds) < k and len(seeds) < n:
        farthest = Person(0, 0, 0, 0)
        best_dist = -1
        for p in points:
            d = min(manhattan((p.x, p.y), (s[0], s[1])) for s in seeds)
            if d > best_dist:
                best_dist = d
                farthest = p
        seeds.append((farthest.x, farthest.y))
    medians = [(s[0], s[1]) for s in seeds[:k]]
    clusters = []
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for p in points:
            idx = min(range(k), key=lambda i: manhattan((p.x, p.y), medians[i]))
            clusters[idx].append(p)
        newmedians = []
        for cl in clusters:
            if not cl:
                newmedians.append((random.choice(points).x, random.choice(points).y))
            else:
                xs = sorted([q.x for q in cl])
                ys = sorted([q.y for q in cl])
                midx = len(xs) // 2
                midy = len(ys) // 2
                newmedians.append((xs[midx], ys[midy]))
        if newmedians == medians:
            break
        medians = newmedians
    return medians, clusters


def compute_distance_matrix(coords):
    n = len(coords)
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                d[i][j] = 0
            else:
                xj, yj = coords[j]
                d[i][j] = abs(xi - xj) + abs(yi - yj)
    return d


def route_finish_time(indices, dist, start_time):
    if not indices:
        return start_time
    t = start_time
    pos = 0
    for idx in indices:
        t += dist[pos][idx]
        t += 1
        pos = idx
    t += dist[pos][0]
    t += 1
    return t


def aco_find_routes(
    pers,
    hospital_xy,
    capacity=4,
    num_ants=20,
    iterations=100,
    alpha=1.0,
    beta=2.0,
    evaporation=0.1,
    candidate_k=8,
    Q=1.0,
    tau_min=0.01,
    tau_max=10.0,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
    n = len(pers)
    if n == 0:
        return []
    coords = [hospital_xy] + [(p.x, p.y) for p in pers]
    dist = compute_distance_matrix(coords)
    survival_times = [None] + [p.st for p in pers]
    min_possible = []
    for i in range(1, n + 1):
        min_possible.append(dist[i][0] + 1 + 1)
    feasible_idx = [
        i
        for i in range(1, n + 1)
        if route_finish_time([i], dist, 0) <= survival_times[i]
    ]
    if not feasible_idx:
        return []
    tau = [[1.0 for _ in range(n + 1)] for _ in range(n + 1)]
    urgency = [0.0] * (n + 1)
    for i in range(1, n + 1):
        slack = survival_times[i] - (dist[i][0] + 2)
        if slack <= 0:
            urgency[i] = 1.0
        else:
            urgency[i] = 1.0 / (slack + 1.0)
    eta = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(1, n + 1):
            eta[i][j] = urgency[j] / (dist[i][j] + 1.0)
    best_routes_pool = []
    all_generated = []
    for it in range(iterations):
        population_routes = []
        for ant in range(num_ants):
            available = set(range(1, n + 1))
            ant_routes = []
            while True:
                route_idx = []
                pos = 0
                while len(route_idx) < capacity and available:
                    cand_list = sorted(list(available), key=lambda j: dist[pos][j])[
                        :candidate_k
                    ]
                    feasible_cand = []
                    for j in cand_list:
                        t_finish = route_finish_time(route_idx + [j], dist, 0)
                        ok = all(
                            t_finish <= survival_times[k] for k in (route_idx + [j])
                        )
                        if ok:
                            feasible_cand.append(j)
                    if not feasible_cand:
                        break
                    weights = []
                    for j in feasible_cand:
                        weights.append(
                            (tau[pos][j] ** alpha) * (eta[pos][j] ** beta) + 1e-12
                        )
                    total = sum(weights)
                    r = random.random() * total
                    cum = 0.0
                    chosen = feasible_cand[-1]
                    for w, j in zip(weights, feasible_cand):
                        cum += w
                        if r <= cum:
                            chosen = j
                            break
                    route_idx.append(chosen)
                    available.remove(chosen)
                    pos = chosen
                if not route_idx:
                    break
                t_finish = route_finish_time(route_idx, dist, 0)
                if any(t_finish > survival_times[k] for k in route_idx):
                    continue
                route_persons = [pers[i - 1] for i in route_idx]
                score = len(route_idx) / (t_finish + 1.0)
                population_routes.append(
                    {
                        "route_idx": route_idx,
                        "route": route_persons,
                        "finish": t_finish,
                        "score": score,
                    }
                )
                ant_routes.append(route_idx)
            all_generated.extend(population_routes)
        for i in range(n + 1):
            for j in range(n + 1):
                tau[i][j] = (1.0 - evaporation) * tau[i][j]
                if tau[i][j] < tau_min:
                    tau[i][j] = tau_min
        population_routes.sort(key=lambda r: (r["score"], -r["finish"]), reverse=True)
        for rdict in population_routes:
            indices = rdict["route_idx"]
            L = rdict["finish"]
            m = len(indices)
            if m == 0:
                continue
            delta = Q * (m / (L + 1.0))
            prev = 0
            for idx in indices:
                tau[prev][idx] += delta
                prev = idx
            tau[prev][0] += delta
        for i in range(n + 1):
            for j in range(n + 1):
                if tau[i][j] > tau_max:
                    tau[i][j] = tau_max
    unique_routes = []
    seen_sets = set()
    for r in sorted(all_generated, key=lambda x: x["score"], reverse=True):
        s = tuple(sorted(r["route_idx"]))
        if s in seen_sets:
            continue
        seen_sets.add(s)
        unique_routes.append(r)
    selected = []
    used = set()
    for r in unique_routes:
        if any(idx in used for idx in r["route_idx"]):
            continue
        selected.append(r)
        used.update(r["route_idx"])
    return selected


def algorithm(pers, hosps):
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

    solution_string = ""
    k = len(hosps)
    hid_to_hosp = {}
    medians, clusters = k_medians(pers, k)

    for i, h in enumerate(hosps):
        hx, hy = medians[i]
        h.x = hx
        h.y = hy
        solution_string += f"Hospital:{h.x},{h.y},{h.num_amb}\n"
        hid_to_hosp[h.hid] = h
    solution_string += "\n"
    hosp_to_people = {}
    for i, h in enumerate(hosps):
        hosp_to_people[h.hid] = clusters[i][:] if i < len(clusters) else []

    hid_to_routes = {}
    for h in hosps:
        cluster_pers = hosp_to_people[h.hid]
        routes = aco_find_routes(
            cluster_pers,
            (h.x, h.y),
            capacity=4,
            num_ants=30,
            iterations=120,
            alpha=1.5,
            beta=2.0,
            evaporation=0.15,
            candidate_k=8,
            Q=1.0,
            seed=0,
        )
        routes_sorted = sorted(routes, key=lambda r: r["score"], reverse=True)
        hid_to_routes[h.hid] = routes_sorted[:]
    rescued = set()
    heap = []
    for h in hosps:
        for amb_idx in range(h.num_amb):
            heapq.heappush(heap, (0, h.hid, amb_idx))

    while heap:
        avail_time, hid, amb_idx = heapq.heappop(heap)
        h = hid_to_hosp[hid]
        routes = hid_to_routes.get(hid, [])
        assigned = False
        new_routes = []
        for r in routes:
            person_objs = r["route"]
            route_duration = r["finish"]
            if any(p.pid in rescued for p in person_objs):
                new_routes.append(r)
                continue
            finish_time = avail_time + route_duration
            if all(finish_time <= p.st for p in person_objs):
                line = f"Ambulance: {h.hid}: ({h.x},{h.y})"
                for p in person_objs:
                    line += f", {p.pid}: ({p.x},{p.y},{p.st})"
                line += f", {h.hid}: ({h.x},{h.y})\n"
                solution_string += line
                for p in person_objs:
                    rescued.add(p.pid)
                heapq.heappush(heap, (finish_time, hid, amb_idx))
                assigned = True
                break
            else:
                new_routes.append(r)
        if assigned:
            remaining = [rr for rr in routes if rr not in (r,)]
            hid_to_routes[hid] = remaining
        else:
            hid_to_routes[hid] = new_routes

    solution_string += "\n"
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
                solution = algorithm(persons, hospitals)
                signal.alarm(0)
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
                    result[0] = algorithm(persons, hospitals)
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
        solution = algorithm(persons, hospitals)

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
