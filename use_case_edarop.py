"""This file solves a use case using alibaba traces. It has 8 regions (4 edge
and its associated 4 cloud regions) and 3 applications."""

import pickle
from typing import Optional

import pandas as pd
from pulp import PULP_CBC_CMD  # type: ignore
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich import print

from edarop.model import (
    Region,
    TimeUnit,
    TimeValue,
    InstanceClass,
    TimeRatioValue,
    App,
    Performance,
    Workload,
    System,
    Problem,
    Latency,
    Status,
    Solution,
)
from edarop.edarop import (
    EdaropCAllocator,
    EdaropRAllocator,
    EdaropCRAllocator,
    EdaropRCAllocator,
)
from edarop.simple_allocator import SimpleCostAllocator
from edarop.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
from edarop.analysis import SolutionAnalyzer

TRACE_FILE = "edge_1h.csv"
DIR_OUT = "."


def get_workload(
    wdf: pd.DataFrame, app: int, region: int, time_slot: Optional[int] = None
) -> tuple[int, ...]:
    """Returns the workload values for an app in a region. It gets them from the
    dataframe."""
    wl = tuple(wdf[(wdf.app == app) & (wdf.reg == region)].reqs)
    if time_slot is not None:
        return (wl[time_slot],)
    return wl


def set_up(
    time_slot: Optional[int] = None,
) -> tuple[System, dict[tuple[App, Region], Workload]]:
    """Prepares the data. If timeslot is None, it uses all the timeslots. If
    it is an integer, only the timeslot with that index."""
    user_region_list = (
        Region("eu-central-1-ham-1a-u"),  # Hamburg
        Region("us-east-1-atl-1a-u"),  # Atlanta
        Region("us-west-2-lax-1a-u"),  # Los Angeles
        Region("ap-south-1-del-1a-u"),  # Delhi
    )
    edge_region_list = (
        Region("eu-central-1-ham-1a"),  # Hamburg
        Region("us-east-1-atl-1a"),  # Atlanta
        Region("us-west-2-lax-1a"),  # Los Angeles
        Region("ap-south-1-del-1a"),  # Delhi
    )
    cloud_region_list = (
        Region("eu-central-1"),  # Frankfurt
        Region("us-east-1"),  # N. Virginia
        Region("us-west-2"),  # Oregon
        Region("ap-south-1"),  # Mumbai
    )
    region_list = (
        *user_region_list,
        *cloud_region_list,
        *edge_region_list,
    )

    regions = {i.name: i for i in region_list}

    latencies = {
        # eu-central-1-ham-1a
        (regions["eu-central-1-ham-1a-u"], regions["eu-central-1-ham-1a"]): Latency(
            value=TimeValue(0.00872, TimeUnit("s")),
        ),
        (regions["eu-central-1-ham-1a-u"], regions["eu-central-1"]): Latency(
            value=TimeValue(0.0162, TimeUnit("s")),
        ),
        (regions["eu-central-1-ham-1a-u"], regions["us-east-1"]): Latency(
            value=TimeValue(0.1069, TimeUnit("s")),
        ),
        (regions["eu-central-1-ham-1a-u"], regions["us-west-2"]): Latency(
            value=TimeValue(0.2143, TimeUnit("s")),
        ),
        (regions["eu-central-1-ham-1a-u"], regions["ap-south-1"]): Latency(
            value=TimeValue(0.1258, TimeUnit("s")),
        ),
        # us-east-1-atl-1a
        (regions["us-east-1-atl-1a-u"], regions["us-east-1-atl-1a"]): Latency(
            value=TimeValue(0.0011, TimeUnit("s")),
        ),
        (regions["us-east-1-atl-1a-u"], regions["eu-central-1"]): Latency(
            value=TimeValue(0.1058, TimeUnit("s")),
        ),
        (regions["us-east-1-atl-1a-u"], regions["us-east-1"]): Latency(
            value=TimeValue(0.0151, TimeUnit("s")),
        ),
        (regions["us-east-1-atl-1a-u"], regions["us-west-2"]): Latency(
            value=TimeValue(0.0662, TimeUnit("s")),
        ),
        (regions["us-east-1-atl-1a-u"], regions["ap-south-1"]): Latency(
            value=TimeValue(0.2033, TimeUnit("s")),
        ),
        # us-west-2-lax-1a
        (regions["us-west-2-lax-1a-u"], regions["us-west-2-lax-1a"]): Latency(
            value=TimeValue(0.0009, TimeUnit("s")),
        ),
        (regions["us-west-2-lax-1a-u"], regions["eu-central-1"]): Latency(
            value=TimeValue(0.1479, TimeUnit("s")),
        ),
        (regions["us-west-2-lax-1a-u"], regions["us-east-1"]): Latency(
            value=TimeValue(0.0599, TimeUnit("s")),
        ),
        (regions["us-west-2-lax-1a-u"], regions["us-west-2"]): Latency(
            value=TimeValue(0.0236, TimeUnit("s")),
        ),
        (regions["us-west-2-lax-1a-u"], regions["ap-south-1"]): Latency(
            value=TimeValue(0.2472, TimeUnit("s")),
        ),
        # ap-south-1-del-1a
        (regions["ap-south-1-del-1a-u"], regions["ap-south-1-del-1a"]): Latency(
            value=TimeValue(0.0085, TimeUnit("s")),
        ),
        (regions["ap-south-1-del-1a-u"], regions["eu-central-1"]): Latency(
            value=TimeValue(0.1794, TimeUnit("s")),
        ),
        (regions["ap-south-1-del-1a-u"], regions["us-east-1"]): Latency(
            value=TimeValue(0.2346, TimeUnit("s")),
        ),
        (regions["ap-south-1-del-1a-u"], regions["us-west-2"]): Latency(
            value=TimeValue(0.2949, TimeUnit("s")),
        ),
        (regions["ap-south-1-del-1a-u"], regions["ap-south-1"]): Latency(
            value=TimeValue(0.0271, TimeUnit("s")),
        ),
    }

    # Instance classes
    ic_list = (
        # eu-central-1
        InstanceClass(
            name="c5.2xlarge-eu-central-1",
            price=TimeRatioValue(0.388, TimeUnit("h")),
            region=regions["eu-central-1"],
        ),
        InstanceClass(
            name="c5.4xlarge-eu-central-1",
            price=TimeRatioValue(0.776, TimeUnit("h")),
            region=regions["eu-central-1"],
        ),
        # eu-central-1-ham-1a
        InstanceClass(
            name="c5.2xlarge-eu-central-1-ham-1a",
            price=TimeRatioValue(0.524, TimeUnit("h")),
            region=regions["eu-central-1-ham-1a"],
        ),
        # us-east-1
        InstanceClass(
            name="c5.2xlarge-us-east-1",
            price=TimeRatioValue(0.34, TimeUnit("h")),
            region=regions["us-east-1"],
        ),
        InstanceClass(
            name="c5.4xlarge-us-east-1",
            price=TimeRatioValue(0.68, TimeUnit("h")),
            region=regions["us-east-1"],
        ),
        # us-east-1-atl-1a
        InstanceClass(
            name="c5d.2xlarge-us-east-1-atl-1a",
            price=TimeRatioValue(0.48, TimeUnit("h")),
            region=regions["us-east-1-atl-1a"],
        ),
        # us-west-2
        InstanceClass(
            name="c5.2xlarge-us-west-2",
            price=TimeRatioValue(0.34, TimeUnit("h")),
            region=regions["us-west-2"],
        ),
        InstanceClass(
            name="c5.4xlarge-us-west-2",
            price=TimeRatioValue(0.68, TimeUnit("h")),
            region=regions["us-west-2"],
        ),
        # us-east-1-atl-1a
        InstanceClass(
            name="c5.2xlarge-us-west-2-lax-1a",
            price=TimeRatioValue(0.408, TimeUnit("h")),
            region=regions["us-west-2-lax-1a"],
        ),
        InstanceClass(
            name="c5.4xlarge-us-west-2-lax-1a",
            price=TimeRatioValue(0.816, TimeUnit("h")),
            region=regions["us-west-2-lax-1a"],
        ),
        # ap-south-1
        InstanceClass(
            name="c5.2xlarge-ap-south-1",
            price=TimeRatioValue(0.34, TimeUnit("h")),
            region=regions["ap-south-1"],
        ),
        InstanceClass(
            name="c5.4xlarge-ap-south-1",
            price=TimeRatioValue(0.68, TimeUnit("h")),
            region=regions["ap-south-1"],
        ),
        # ap-south-1-del-1
        InstanceClass(
            name="c5.2xlarge-ap-south-1-del-1a",
            price=TimeRatioValue(0.459, TimeUnit("h")),
            region=regions["ap-south-1-del-1a"],
        ),
    )

    ics = {i.name: i for i in ic_list}

    app_list = (
        App(name="a0", max_resp_time=TimeValue(0.2, TimeUnit("s"))),
        App(name="a1", max_resp_time=TimeValue(0.325, TimeUnit("s"))),
        App(name="a2", max_resp_time=TimeValue(0.050, TimeUnit("s"))),
    )

    apps = {a.name: a for a in app_list}

    # The values are the performance (in rps) and the S_ia (in seconds).
    # This is a short cut for not having to repeat all units.
    #
    # The data is obtained with flask bench with these iterations:
    # - a0: 60_000
    # - a1: 352_000
    # - a2: 100_000
    perf_dict = {
        # a0
        (apps["a0"], ics["c5.2xlarge-eu-central-1"]): (461.125, 0.025),
        (apps["a0"], ics["c5.4xlarge-eu-central-1"]): (919.905, 0.025),
        (apps["a0"], ics["c5.2xlarge-eu-central-1-ham-1a"]): (461.125, 0.025),
        (apps["a0"], ics["c5.2xlarge-us-east-1"]): (461.125, 0.025),
        (apps["a0"], ics["c5.4xlarge-us-east-1"]): (919.905, 0.025),
        (apps["a0"], ics["c5d.2xlarge-us-east-1-atl-1a"]): (449.114, 0.025),
        (apps["a0"], ics["c5.2xlarge-us-west-2"]): (461.125, 0.025),
        (apps["a0"], ics["c5.4xlarge-us-west-2"]): (919.905, 0.025),
        (apps["a0"], ics["c5.2xlarge-us-west-2-lax-1a"]): (461.125, 0.025),
        (apps["a0"], ics["c5.4xlarge-us-west-2-lax-1a"]): (919.905, 0.025),
        (apps["a0"], ics["c5.2xlarge-ap-south-1"]): (461.125, 0.025),
        (apps["a0"], ics["c5.4xlarge-ap-south-1"]): (919.905, 0.025),
        (apps["a0"], ics["c5.2xlarge-ap-south-1-del-1a"]): (461.125, 0.025),
        # a1
        (apps["a1"], ics["c5.2xlarge-eu-central-1"]): (76.796, 0.0125),
        (apps["a1"], ics["c5.4xlarge-eu-central-1"]): (153.005, 0.0125),
        (apps["a1"], ics["c5.2xlarge-eu-central-1-ham-1a"]): (76.796, 0.0125),
        (apps["a1"], ics["c5.2xlarge-us-east-1"]): (76.796, 0.0125),
        (apps["a1"], ics["c5.4xlarge-us-east-1"]): (153.005, 0.0125),
        (apps["a1"], ics["c5d.2xlarge-us-east-1-atl-1a"]): (80.296, 0.0125),
        (apps["a1"], ics["c5.2xlarge-us-west-2"]): (76.796, 0.0125),
        (apps["a1"], ics["c5.4xlarge-us-west-2"]): (153.005, 0.0125),
        (apps["a1"], ics["c5.2xlarge-us-west-2-lax-1a"]): (76.796, 0.0125),
        (apps["a1"], ics["c5.4xlarge-us-west-2-lax-1a"]): (153.005, 0.0125),
        (apps["a1"], ics["c5.2xlarge-ap-south-1"]): (76.796, 0.0125),
        (apps["a1"], ics["c5.4xlarge-ap-south-1"]): (153.005, 0.0125),
        (apps["a1"], ics["c5.2xlarge-ap-south-1-del-1a"]): (76.796, 0.0125),
        # a2
        (apps["a2"], ics["c5.2xlarge-eu-central-1"]): (263.76, 0.040),
        (apps["a2"], ics["c5.4xlarge-eu-central-1"]): (556.68, 0.040),
        (apps["a2"], ics["c5.2xlarge-eu-central-1-ham-1a"]): (263.76, 0.040),
        (apps["a2"], ics["c5.2xlarge-us-east-1"]): (263.76, 0.040),
        (apps["a2"], ics["c5.4xlarge-us-east-1"]): (556.68, 0.040),
        (apps["a2"], ics["c5d.2xlarge-us-east-1-atl-1a"]): (275.09, 0.040),
        (apps["a2"], ics["c5.2xlarge-us-west-2"]): (263.76, 0.040),
        (apps["a2"], ics["c5.4xlarge-us-west-2"]): (556.68, 0.040),
        (apps["a2"], ics["c5.2xlarge-us-west-2-lax-1a"]): (263.76, 0.040),
        (apps["a2"], ics["c5.4xlarge-us-west-2-lax-1a"]): (556.68, 0.040),
        (apps["a2"], ics["c5.2xlarge-ap-south-1"]): (263.76, 0.040),
        (apps["a2"], ics["c5.4xlarge-ap-south-1"]): (556.68, 0.040),
        (apps["a2"], ics["c5.2xlarge-ap-south-1-del-1a"]): (263.76, 0.040),
    }

    perfs = {}
    for p, v in perf_dict.items():
        perfs[p] = Performance(
            value=TimeRatioValue(v[0], TimeUnit("s")),
            slo=TimeValue(v[1], TimeUnit("s")),
        )

    system = System(apps=app_list, ics=ic_list, perfs=perfs, latencies=latencies)

    wdf = pd.read_csv(TRACE_FILE)

    workloads = {}
    for r in range(len(user_region_list)):
        for a in range(len(app_list)):
            workloads[(app_list[a], user_region_list[r])] = Workload(
                values=get_workload(wdf, app=a, region=r, time_slot=time_slot),
                time_unit=TimeUnit("h"),
            )

    return (system, workloads)


def solve_for_one_timeslot(
    time_slot: int,
) -> tuple[list[dict[str, object]], list[Solution]]:
    """Solve for one time slot. Returns the summaries and the cost."""
    console = Console()

    system, workloads = set_up(time_slot)
    problem = Problem(system=system, workloads=workloads, max_cost=-1)
    problem_edarop_r = Problem(system=system, workloads=workloads, max_cost=200)

    ProblemPrettyPrinter(problem).print()

    msg = True

    solver = PULP_CBC_CMD(msg=msg, timeLimit=120, options=["preprocess off"])

    if msg:
        console.print(Markdown("# EdaropC"))
    sol_c = EdaropCAllocator(problem).solve(solver)

    if msg:
        console.print(Markdown("# EdaropCR"))
    sol_cr = EdaropCRAllocator(problem).solve(solver)

    if msg:
        console.print(Markdown("# EdaropR"))
    sol_r = EdaropRAllocator(problem_edarop_r).solve(solver)

    if msg:
        console.print(Markdown("# EdaropRC"))
    sol_rc = EdaropRCAllocator(problem_edarop_r).solve(solver)

    if msg:
        console.print(Markdown("# SimpleCostAllocator"))
    sol_simple = SimpleCostAllocator(problem).solve()

    # Print in three tables the five solutions
    # Gather the information
    info_allocs = []
    summaries = []  # Each element is for a time slot
    sols = [sol_c, sol_cr, sol_r, sol_rc, sol_simple]
    for sol in sols:
        sol_table = Table.grid()

        analyzer = SolutionAnalyzer(sol)

        if sol.solving_stats.status in [Status.OPTIMAL, Status.INTEGER_FEASIBLE]:
            summaries.append(
                {
                    "cost": analyzer.cost(),
                    "avg_resp_time": analyzer.avg_resp_time(),
                    "status": sol.solving_stats.status.name,
                    "deadline_miss_rate": analyzer.deadline_miss_rate(),
                    "total_reqs_per_app": analyzer.total_reqs_per_app(),
                    "total_missed_reqs_per_app": analyzer.total_missed_reqs_per_app(),
                }
            )
        else:
            summaries.append(
                {
                    "cost": "-",
                    "avg_resp_time": "-",
                    "status": sol.solving_stats.status.name,
                    "deadline_miss_rate": "-",
                    "total_reqs_per_app": "-",
                    "total_missed_reqs_per_app": "-",
                }
            )

        sol_table.add_row(SolutionPrettyPrinter(sol).get_summary())

        for t in SolutionPrettyPrinter(sol).get_tables():
            sol_table.add_row(t)

        info_allocs.append(sol_table)

    # Table 1: EdaropC and EdaropCR
    table = Table(box=None)
    for i in ["EdaropC", "EdaropCR"]:
        table.add_column(i, justify="center")

    table.add_row(*info_allocs[:2])

    print(table)

    # Table 2: EdaropR and EdaropRC
    table = Table(box=None)
    for i in ["EdaropR", "EdaropRC"]:
        table.add_column(i, justify="center")

    table.add_row(*info_allocs[2:4])

    print(table)

    # Table 3: SimpleCostAllocator
    table = Table(box=None)
    table.add_column("SimpleCostAllocator", justify="center")
    table.add_row(info_allocs[4])

    print(table)

    return summaries, sols


def get_trace_len(trace_file: str) -> int:
    """Returns the number of time slots in a CSV trace file"""
    wdf = pd.read_csv(trace_file)
    return len(wdf.timestamp.unique())


def summary_cell_to_str(cell) -> str:
    """Converts a cell in the summary table to a string."""
    if cell["cost"] == "-":
        return f"-\n\n{cell['status']}\n"  # Not optimal solution

    return (
        f"${cell['cost']:.2f}\n"
        f"{cell['avg_resp_time'].value*1000:.2f} ms\n"
        f"{cell['status']}\n"
        f"{cell['deadline_miss_rate']*100:.2f} %"
    )


def solve_all_time_slots_individually() -> None:
    """Solve for all time slots and print the summary table."""
    console = Console()
    trace_len = get_trace_len(TRACE_FILE)

    table = Table(
        "Time slot",
        "EdaropC",
        "EdaropCR",
        "EdaropR",
        "EdaropRC",
        "SimpleCostAllocator",
        title="Summary of all time slots (cost/avg. resp. time/status/deadline miss rate)",
    )

    print(f"There are {trace_len} time slots")
    totals = {
        "cost": [
            0,
            0,
            0,
            0,
            0,
        ],  # EdaropC, EdaropCR, EdaropR, EdaropRC, SimpleCostAllocator
        "resp_time": [0, 0, 0, 0, 0],  # Idem
        "deadline_miss_rate": [0, 0, 0, 0, 0],  # Idem
        "reqs_per_app": {app_index: [0, 0, 0, 0, 0] for app_index in range(3)},
        "miss_reqs_per_app": {app_index: [0, 0, 0, 0, 0] for app_index in range(3)},
    }
    all_sols = []
    for i in range(trace_len):
        console.rule(f":clock10: Time slot {i}")
        summaries, sols = solve_for_one_timeslot(i)

        all_sols.append(sols)

        for j in range(5):
            if summaries[j]["cost"] == "-":
                continue  # The solution is not optimal
            totals["cost"][j] += summaries[j]["cost"]
            totals["resp_time"][j] += summaries[j]["avg_resp_time"].value
            totals["deadline_miss_rate"][j] += summaries[j]["deadline_miss_rate"]

            apps = list(summaries[j]["total_reqs_per_app"].keys())

            for app_index in range(3):
                app = apps[app_index]
                totals["reqs_per_app"][app_index][j] += summaries[j][
                    "total_reqs_per_app"
                ][app]
                totals["miss_reqs_per_app"][app_index][j] += summaries[j][
                    "total_missed_reqs_per_app"
                ][app]

        table.add_row(
            str(i),
            *(summary_cell_to_str(s) for s in summaries),
        )
        table.add_section()

    table.add_row("Total cost", *(f"${i:.2f}" for i in totals["cost"]))
    table.add_row(
        "Avg. resp. time", *(f"{i*1000/trace_len:.2f} ms" for i in totals["resp_time"])
    )
    table.add_row(
        "Avg. miss rate",
        *(f"{i*100/trace_len:.2f} %" for i in totals["deadline_miss_rate"]),
    )
    print(table)

    print(
        "Notice that the avg. resp. time and the avg. miss rate in the bottom "
        "is an approximation, because it's the average of the aggregated "
        "response times and miss rates, respectively."
    )

    with open(f"{DIR_OUT}/sols_edarop.p", "wb") as f:
        pickle.dump(all_sols, f)


if __name__ == "__main__":
    solve_all_time_slots_individually()
