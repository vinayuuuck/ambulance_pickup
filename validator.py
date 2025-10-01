#!/usr/bin/env python

import re
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from client import algorithm


def take_time(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


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
        return

    def __repr__(self):
        return "%d: (%d,%d,%d)" % (self.pid, self.x, self.y, self.st)

    def prettify(self):
        return {self.pid: (self.x, self.y, self.st)}


#  Hospital object
HID = 0


class Hospital:

    def __init__(self, x, y, num_amb):
        global HID
        HID += 1
        self.hid = HID
        self.x = x
        self.y = y
        # amb_time array represents the time each ambulance have already spent.
        # NOTE: this should be sorted in a decreasing order (larger value first).

        # TODO: Handle for ambulance count after dropping
        self.num_amb = num_amb
        self.amb_time = [0] * num_amb
        return

    def __repr__(self):
        return "%d: [(%d,%d), %d]" % (self.hid, self.x, self.y, self.num_amb)

    def prettify(self):
        return {self.hid: (self.x, self.y)}

    def rescue(self, pers, end_hospital):
        if self.num_amb == 0:
            raise IllegalPlanError("No ambulance left at the hospital: %s" % self)
        if 4 < len(pers):
            raise IllegalPlanError(
                "Cannot rescue more than four people at once: %s" % pers
            )
        already_rescued = list(filter(lambda p: p.rescued, pers))
        if already_rescued:
            raise IllegalPlanError("Person already rescued: %s" % already_rescued)
        # t: time to take
        # note: we don't have to go back to the starting hospital
        # so go to the closest from the last student
        t = 1
        start = self
        for p in pers:
            t += take_time(start, p)
            start = p

        t += len(pers)

        # TODO: Change to user-input end hospital

        """"
        # look for closest hospital
        # min_hosp = 0
        # min_hosp_distance = None
        # print(hospitals)
        # for hosp in hospitals:
        #     tmp_dist = take_time(start, hosp)
        #     if not min_hosp_distance:
        #         min_hosp_distance = tmp_dist
        #         min_hosp = hosp
        #     elif tmp_dist < min_hosp_distance:
        #         min_hosp_distance = tmp_dist
        #         min_hosp = hosp
        # t += min_hosp_distance
        """

        t += take_time(start, end_hospital)

        self.amb_time.sort(reverse=True)
        # try to schedule from the busiest ambulance at the hospital.
        for i, t0 in enumerate(self.amb_time):
            if not list(filter(lambda p: p.st < t0 + t, pers)):
                break
        else:
            raise IllegalPlanError("Either person cannot make it: %s" % pers)
        # proceed the time.
        self.amb_time[i] += t

        # TODO: Send the ambulance to next hospital

        end_hospital.num_amb += 1
        end_hospital.amb_time.append(self.amb_time[i])

        self.num_amb -= 1
        self.amb_time.pop(i)

        # keep it sorted.
        for p in pers:
            p.rescued = True
        print(
            "Rescued:",
            " and ".join(map(str, pers)),
            "taking",
            t,
            "|ended at hospital",
            end_hospital,
        )
        return pers


# read_data
def read_data(fname="data.txt"):
    print("Reading data:", fname)
    persons = []
    hospitals = []
    mode = 0

    with open(fname, "r") as fil:
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

    print(persons)
    print(hospitals)
    return persons, hospitals


# read_results
def readresults(persons, hospitals, fname="result.txt"):

    # Regex to extract the patterns
    p1 = re.compile(r"(\d+\s*:\s*\(\s*\d+\s*,\s*\d+(\s*,\s*\d+)?\s*\))")
    p2 = re.compile(r"(\d+)\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    p3 = re.compile(r"(\d+)\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

    res = {}
    score = 0
    hospital_index = 0
    with open(fname, "r") as fil:
        data = fil.readlines()

    for line in data:
        line = line.strip().lower()
        if not line:
            continue

        # check for hospital coordinates
        if line.startswith("hospital"):

            # read in hospital coordinates and set on hospital object
            (x, y, z) = map(int, line.replace("hospital:", "").split(","))
            print(
                "Hospital #{idx}: coordinates ({x},{y})".format(
                    x=x, y=y, idx=hospital_index + 1
                )
            )
            if not (x and y):
                raise ValidationError("Hospital coordinates not set: line".format(line))
            if z != hospitals[hospital_index].num_amb:
                raise ValidationError(
                    "Hospital's ambulance # does not match input (input={0}, results={1})".format(
                        hospitals[hospital_index].num_amb, z
                    )
                )
            hospitals[hospital_index].x = x
            hospitals[hospital_index].y = y
            hospital_index += 1
            continue

        # check for ambulance records
        if not line.startswith("ambulance"):
            print("!!! Ignored: %r" % line)
            continue
        try:
            hos = None
            end_hos = None
            rescue_persons = []
            groups = p1.findall(line)
            groups_length = len(groups)

            for i, (w, z) in enumerate(groups):
                m = p2.match(w)
                if m:
                    # Hospital n:(x,y)
                    if i != 0 and i != groups_length - 1:
                        raise FormatSyntaxError("Specify a person now: %r" % line)
                    (a, b, c) = map(int, m.groups())
                    if a <= 0 or len(hospitals) < a:
                        raise FormatSyntaxError("Illegal hospital id: %d" % a)
                    if i == 0:
                        hos = hospitals[a - 1]
                        if hos.x != b or hos.y != c:
                            raise DataMismatchError(
                                "Starting Hospital mismatch: %s != %d:%s"
                                % (hos, a, (b, c))
                            )
                    else:
                        end_hos = hospitals[a - 1]
                        if end_hos.x != b or end_hos.y != c:
                            raise DataMismatchError(
                                "Ending Hospital mismatch: %s != %d:%s"
                                % (end_hos, a, (b, c))
                            )

                    continue
                m = p3.match(w)
                if m:
                    # Person n:(x,y,t)
                    if i == 0:
                        raise FormatSyntaxError("Specify a hospital first: %r" % line)
                    (a, b, c, d) = map(int, m.groups())
                    if a <= 0 or len(persons) < a:
                        raise FormatSyntaxError("Illegal person id: %d" % a)
                    per = persons[a - 1]
                    if per.x != b or per.y != c or per.st != d:
                        raise DataMismatchError(
                            "Person mismatch: %s != %d:%s" % (per, a, (b, c, d))
                        )
                    rescue_persons.append(per)
                    continue
                # error
                raise FormatSyntaxError('Expected "n:(x,y)" or "n:(x,y,t)": %r' % line)

            if not hos or not rescue_persons:
                print("!!! Insufficient data: %r" % line)
                continue
            if not hos or not end_hos:
                raise FormatSyntaxError(
                    "Either start hospital or end hospital is not defined: %s" % line
                )
            pers = hos.rescue(rescue_persons, end_hos)
            if hos.hid in res:
                res[hos.hid].extend(pers)
            else:
                res[hos.hid] = pers
            score += len(rescue_persons)
        except ValidationError as x:
            print("!!!", x)

    print("Total score:", score)
    return res


def process_list(hosp, row):
    global hospitals
    x, y = hospitals.iloc[hosp - 1]["x"], hospitals.iloc[hosp - 1]["y"]
    values = [(x, y)] + row

    temp = [values[0]]
    for i in range(len(values) - 1):
        temp.append((values[i][0], values[i + 1][1]))
        temp.append(values[i + 1])
    return temp


def prettify(self):
    try:
        return {self.pid: (self.x, self.y, self.st)}
    except AttributeError:
        return {self.hid: (self.x, self.y)}


def plot(_persons, _hospitals, fname="result.txt"):
    # TODO: Plot bug
    global persons, hospitals
    persons = _persons
    persons = {key: val for person in persons for key, val in person.prettify().items()}

    persons = pd.DataFrame.from_dict(persons, orient="index")
    persons["person"] = persons.index
    persons = persons.rename(columns={0: "x", 1: "y", 2: "time"})
    persons = persons[["person", "x", "y", "time"]]

    hospitals = _hospitals
    hospitals = {
        key: val for hospital in hospitals for key, val in hospital.prettify().items()
    }

    hospitals = pd.DataFrame.from_dict(hospitals, orient="index")
    hospitals["hospital"] = hospitals.index
    hospitals = hospitals.rename(columns={0: "x", 1: "y"})
    hospitals = hospitals[["hospital", "x", "y"]]

    curr_time = 100

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")

    line = ax.plot([], [], "o-", lw=2)

    max_, min_ = persons["time"].max(), persons["time"].min()
    bins = (max_ - min_) / 4

    colors = ["r", "r", "r", "r"]
    labels = ["<= " + str(min_ + (i * bins)) for i in range(4)]

    dead = persons[persons["time"] > curr_time]

    for i, color in enumerate(colors):
        temp = persons[
            (min_ + (i * bins) <= persons["time"])
            & (persons["time"] <= min_ + ((i + 1) * bins))
            & (persons["time"] < curr_time)
        ]
        ax.scatter(temp["x"], temp["y"], color=color, s=20, label=labels[i])

    ax.scatter(dead["x"], dead["y"], color="black", marker="x", s=50, label="Dead")

    colors = ["r", "g", "b", "orange", "y"]

    p1 = re.compile(r"(\d+\s*:\s*\(\s*\d+\s*,\s*\d+(\s*,\s*\d+)?\s*\))")
    p2 = re.compile(r"(\d+)\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*.*\)")

    with open(fname, "r") as fil:
        data = fil.read().split("\n\n")[1].strip().split("\n")
    splitted = [[cor[0] for cor in p1.findall(val)] for val in data]

    vals = {}
    for row in splitted:
        hospital = int(p2.findall(row[0])[0][0])
        temp = []
        for col in row:
            find = list(map(int, p2.findall(col)[0]))
            temp.append((find[1], find[2]))
        if hospital in vals:
            vals[hospital].append(temp)
        else:
            vals[hospital] = [temp]

    for i, (hosp, val) in enumerate(vals.items()):
        for pers in val:
            ax.plot(*zip(*process_list(hosp, pers)), label=hosp, color=colors[i])
            ax.scatter(*zip(*pers), color=colors[i])
        ax.scatter(
            x=hospitals.iloc[hosp - 1]["x"],
            y=hospitals.iloc[hosp - 1]["y"],
            marker="s",
            s=100,
            label="Hospital" + str(hosp),
            color=colors[i],
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


# Main
if __name__ == "__main__":
    print("Using data.txt as input")
    (persons, hospitals) = read_data("data.txt")
    solution = algorithm(persons, hospitals)
    with open("result.txt", "w") as rf:
        rf.write(solution)
    readresults(persons, hospitals)

    # Comment the below line to disable plot
    plot(persons, hospitals)
