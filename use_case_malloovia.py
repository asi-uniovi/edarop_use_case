"""This module takes the Alibaba use case in edarop and converts it into a
malloovia problem. Then, it solves it with malloovia and converts the solution
back to edarop."""
import math
import pickle
from typing import Dict, Tuple

from rich.console import Console
from rich.table import Table
from rich import print

from cloudmodel.unified.units import Time, Requests, RequestsPerTime, Currency

import malloovia
import edarop
import edarop.model
from use_case_edarop import set_up, get_trace_len, TRACE_FILE, DIR_OUT


class Edarop2Malloovia:
    """This class converts an edarop problem into a malloovia problem. It
    doesn't take into account the latencies or the response times."""

    def __init__(self, problem: edarop.model.Problem):
        self.edarop_problem = problem
        self.m2e_apps: Dict[
            malloovia.App, edarop.model.App
        ] = {}  # malloovia app -> edarop app
        self.m2e_ics: Dict[
            malloovia.InstanceClass, edarop.model.InstanceClass
        ] = {}  # malloovia ic -> edarop ic
        self.m2e_ic_regions: Dict[
            malloovia.InstanceClass, edarop.model.Region
        ] = {}  # malloovia ic -> edarop region

        self.malloovia_problem = self.__convert_problem()

    def __convert_problem(self) -> malloovia.model.Problem:
        units_h_str = "h"

        # Malloovia apps
        malloovia_apps = []
        e2m_apps = {}  # edarop app -> malloovia app
        for edarop_app in self.edarop_problem.system.apps:
            malloovia_app = malloovia.App(
                id=edarop_app.name,
                name=edarop_app.name,
            )
            malloovia_apps.append(malloovia_app)

            e2m_apps[edarop_app] = malloovia_app
            self.m2e_apps[malloovia_app] = edarop_app

        # Malloovia workloads. They aggregate the values of the edarop workloads
        # that have the same in any region.
        malloovia_wls: Dict[
            malloovia.App, list[float]
        ] = {}  # for each app, the list of workloads for each time slot
        num_time_slots = self.edarop_problem.workload_len
        for edarop_app in self.edarop_problem.system.apps:
            malloovia_app = e2m_apps[edarop_app]
            malloovia_wls[malloovia_app] = [0.0] * num_time_slots

            for region in self.edarop_problem.regions:
                if (edarop_app, region) not in self.edarop_problem.workloads:
                    continue

                for ts in range(num_time_slots):
                    malloovia_wls[malloovia_app][ts] += (
                        self.edarop_problem.workloads[edarop_app, region]
                        .values[ts]
                        .magnitude
                    )

        malloovia_wl_list = []
        for malloovia_app in malloovia_apps:
            wl = malloovia.Workload(
                id=malloovia_app.id,
                app=malloovia_app,
                description="",
                time_unit=units_h_str,
                values=tuple(malloovia_wls[malloovia_app]),
            )
            malloovia_wl_list.append(wl)

        # Malloovia instance classes
        malloovia_ics = []
        e2m_ics = {}  # edarop ic -> malloovia ic
        for edarop_ic in self.edarop_problem.system.ics:
            malloovia_ic = malloovia.InstanceClass(
                id=edarop_ic.name,
                name=edarop_ic.name,
                price=edarop_ic.price.to("usd/h").magnitude,
                time_unit=units_h_str,
                # Should be set(), but it generates "TypeError: unhashable type: 'set'"
                limiting_sets=(),
                max_vms=0,  # No limit
            )
            malloovia_ics.append(malloovia_ic)

            e2m_ics[edarop_ic] = malloovia_ic
            self.m2e_ics[malloovia_ic] = edarop_ic
            self.m2e_ic_regions[malloovia_ic] = edarop_ic.region

        # Malloovia performances
        edarop_perfs = self.edarop_problem.system.perfs
        perf_dict: Dict[malloovia.InstanceClass, Dict[malloovia.App, float]] = {}
        for edarop_ic in self.edarop_problem.system.ics:
            malloovia_ic = e2m_ics[edarop_ic]
            perf_dict[malloovia_ic] = {}
            for edarop_app in self.edarop_problem.system.apps:
                malloovia_app = e2m_apps[edarop_app]
                perf_hours = (
                    edarop_perfs[edarop_app, edarop_ic].value.to("req/h").magnitude
                )
                perf_dict[malloovia_ic][malloovia_app] = perf_hours

        malloovia_perfs = malloovia.PerformanceSet(
            id="perf_set",
            time_unit=units_h_str,
            values=malloovia.PerformanceValues(perf_dict),
        )

        return malloovia.Problem(
            id="Alibaba",
            name="Alibaba",
            workloads=tuple(malloovia_wl_list),
            instance_classes=tuple(malloovia_ics),
            performances=malloovia_perfs,
        )

    def malloovia_sol_2_edarop(
        self, malloovia_sol: malloovia.SolutionI
    ) -> edarop.model.Solution:
        """Converts a malloovia solution into an edarop solution. It is assumed
        that the malloovia solution comes from self.malloovia_problem."""
        # Convert the malloovia allocation into an edarop allocation
        edarop_alloc = self.edarop_alloc_from_malloovia_sol(malloovia_sol)
        edarop_solving_stats = self.edarop_stats_from_malloovia_sol(malloovia_sol)

        edarop_sol = edarop.model.Solution(
            problem=self.edarop_problem,
            alloc=edarop_alloc,
            solving_stats=edarop_solving_stats,
        )

        return edarop_sol

    def edarop_stats_from_malloovia_sol(self, malloovia_sol):
        """Converts the solving stats of a malloovia solution into edarop solving
        stats."""
        malloovia_stats = malloovia_sol.solving_stats.algorithm
        solving_stats_edarop = edarop.model.SolvingStats(
            frac_gap=malloovia_stats.frac_gap,
            max_seconds=malloovia_stats.max_seconds,
            lower_bound=malloovia_stats.lower_bound,
            creation_time=malloovia_sol.solving_stats.creation_time,
            solving_time=malloovia_sol.solving_stats.solving_time,
            status=Edarop2Malloovia.m2e_status(malloovia_stats.status),
        )

        return solving_stats_edarop

    def edarop_alloc_from_malloovia_sol(
        self, malloovia_sol: malloovia.SolutionI
    ) -> edarop.model.Allocation:
        """Converts a malloovia solution into an edarop allocation. It is
        assumed that the malloovia solution comes from
        self.malloovia_problem."""
        time_slot_allocs = []
        m_alloc = malloovia_sol.allocation
        for ts_num, m_ts_alloc in enumerate(m_alloc.values):
            e_ts_alloc = self.edarop_ts_alloc_from_malloovia(
                m_alloc, ts_num, m_ts_alloc
            )
            time_slot_allocs.append(e_ts_alloc)

        return edarop.model.Allocation(time_slot_allocs=time_slot_allocs)

    def edarop_ts_alloc_from_malloovia(
        self,
        malloovia_alloc: malloovia.AllocationInfo,
        ts_num: int,
        malloovia_ts_alloc: Tuple[Tuple[float, ...], ...],  # Apps x time slots
    ) -> edarop.model.TimeSlotAllocation:
        """Converts a time slot allocation from malloovia into an edarop
        allocation."""
        ts_ics: edarop.model.AllocationIcs = {}
        ts_reqs: edarop.model.AllocationReqs = {}
        for app_index, app_alloc in enumerate(malloovia_ts_alloc):
            m_app = malloovia_alloc.apps[app_index]
            self.edarop_ts_app_alloc_from_malloovia(
                malloovia_alloc, m_app, ts_num, ts_ics, ts_reqs, app_alloc
            )

        edarop_ts_alloc = edarop.model.TimeSlotAllocation(ics=ts_ics, reqs=ts_reqs)
        return edarop_ts_alloc

    def total_allocated_perf_rph(
        self,
        malloovia_alloc: malloovia.AllocationInfo,
        m_app: malloovia.App,
        app_alloc: Tuple[float, ...],
    ) -> float:
        """Returns the total allocated performance for an app in a time slot."""
        total_perf = 0.0
        for ic_index, num_vms in enumerate(app_alloc):
            m_ic = malloovia_alloc.instance_classes[ic_index]
            total_perf += num_vms * self.get_malloovia_perf_rph(m_app, m_ic)

        return total_perf

    def get_malloovia_perf_rph(
        self, m_app: malloovia.App, m_ic: malloovia.InstanceClass
    ) -> float:
        """Returns the performance of a malloovia app on an instance class in
        requests per hour."""

        perf_units = malloovia.TimeUnit(self.malloovia_problem.performances.time_unit)
        return self.malloovia_problem.performances.values[m_ic, m_app] * perf_units.to(
            "h"
        )

    def edarop_ts_app_alloc_from_malloovia(
        self,
        malloovia_alloc: malloovia.AllocationInfo,
        m_app: malloovia.App,
        ts_num: int,
        ts_ics: edarop.model.AllocationIcs,
        ts_reqs: edarop.model.AllocationReqs,
        app_alloc: Tuple[float, ...],
    ):
        """Converts a time slot allocation for an app from malloovia into an
        edarop allocation."""
        e_app = self.m2e_apps[m_app]

        allocated_perf = self.total_allocated_perf_rph(
            malloovia_alloc, m_app, app_alloc
        )

        for ic_index, num_vms in enumerate(app_alloc):
            if num_vms == 0:
                continue

            m_ic = malloovia_alloc.instance_classes[ic_index]
            e_ic = self.m2e_ics[m_ic]

            # num_vms can be in theory a float in malloovia, but it is an int
            # when the allocation represents a the number of VMs. It can be a
            # float when the allocation represents cost or performance.
            ts_ics[e_app, e_ic] = int(num_vms)

            # Get proportion of requests that are allocated to this IC
            ic_perf = num_vms * self.get_malloovia_perf_rph(m_app, m_ic)
            ic_share = ic_perf / allocated_perf

            for e_region in self.edarop_problem.regions:
                e_wls = self.edarop_problem.workloads
                if (e_app, e_region) not in e_wls:
                    continue

                reqs = e_wls[e_app, e_region].values[ts_num].magnitude

                ts_reqs[e_app, e_region, e_ic] = math.ceil(reqs * ic_share)

    @staticmethod
    def m2e_status(status: malloovia.Status) -> edarop.model.Status:
        """Converts a malloovia solving status into an edarop solving status."""
        translation = {
            malloovia.Status.unsolved: edarop.model.Status.UNSOLVED,
            malloovia.Status.optimal: edarop.model.Status.OPTIMAL,
            malloovia.Status.infeasible: edarop.model.Status.INFEASIBLE,
            malloovia.Status.integer_infeasible: edarop.model.Status.INTEGER_INFEASIBLE,
            malloovia.Status.overfull: edarop.model.Status.OVERFULL,
            malloovia.Status.trivial: edarop.model.Status.TRIVIAL,
            malloovia.Status.aborted: edarop.model.Status.ABORTED,
            malloovia.Status.cbc_error: edarop.model.Status.CBC_ERROR,
            malloovia.Status.unknown: edarop.model.Status.UNKNOWN,
        }
        return translation[status]


def solve_with_malloovia(problem: edarop.model.Problem) -> edarop.model.Solution:
    """Converts an edarop problem into a malloovia problem and solves it."""
    e2m = Edarop2Malloovia(problem)
    sol_malloovia = malloovia.PhaseI(e2m.malloovia_problem).solve()
    sol_edarop = e2m.malloovia_sol_2_edarop(sol_malloovia)

    return sol_edarop


def is_cloud_ic(ic: edarop.model.InstanceClass) -> bool:
    """Returns true if the instance class is a cloud instance class. It assumes
    that edge instance classes end with '-1a'"""
    return not ic.name.endswith("-1a")


def remove_edge_ics(system: edarop.model.System) -> edarop.model.System:
    """Removes the edge instance classes from the system."""
    cloud_ics = tuple(ic for ic in system.ics if is_cloud_ic(ic))

    # Remove the edge instance classes from the performance matrix
    perf_cloud_ics = {}
    for app, ic in system.perfs:
        if is_cloud_ic(ic):
            perf_cloud_ics[app, ic] = system.perfs[app, ic]

    return edarop.model.System(
        apps=system.apps,
        ics=cloud_ics,
        perfs=perf_cloud_ics,
        latencies=system.latencies,
    )


def solve_for_one_timeslot(time_slot: int):
    """Solves the problem for one time slot."""
    edarop_system, edarop_workloads = set_up(time_slot)

    # Since malloovia doesn't know about the difference between cloud and edge
    # regions, we remove the edge instance classes from the system because if
    # malloovia selected an edge instance class, there would be problems in
    # sending requests from other edge regions to it because we are assuming
    # that there's no edge-to-edge communication and, thus, we don't have
    # latency information.
    edarop_system_without_edge_ics = remove_edge_ics(edarop_system)
    edarop_problem = edarop.model.Problem(
        system=edarop_system_without_edge_ics, workloads=edarop_workloads, max_cost=-1
    )

    edarop.visualization.ProblemPrettyPrinter(edarop_problem).print()

    # solver = PULP_CBC_CMD(msg=True, timeLimit=120, options=["preprocess off"])

    sol_edarop = solve_with_malloovia(edarop_problem)
    edarop.visualization.SolutionPrettyPrinter(sol_edarop).print()

    # Gather the information
    summaries = []
    sols = [sol_edarop]
    for sol in sols:
        sol_table = Table("Cost", "Avg. resp. time", "Status")

        analyzer = edarop.visualization.SolutionAnalyzer(sol)

        if sol.solving_stats.status in [
            edarop.model.Status.OPTIMAL,
            edarop.model.Status.INTEGER_FEASIBLE,
        ]:
            summaries.append(
                {
                    "cost": analyzer.cost(),
                    "avg_resp_time": analyzer.avg_resp_time(),
                    "status": sol.solving_stats.status.name,
                }
            )
        else:
            summaries.append(
                {
                    "cost": "-",
                    "avg_resp_time": "-",
                    "status": sol.solving_stats.status.name,
                }
            )

        for summary in summaries:
            sol_table.add_row(
                str(summary["cost"]),
                str(summary["avg_resp_time"]),
                str(summary["status"]),
            )
        print(sol_table)
    return summaries, sols


def solve_all_time_slots_individually():
    """Solves the problem for all time slots."""
    console = Console()
    trace_len = get_trace_len(TRACE_FILE)
    print(f"There are {trace_len} time slots")
    all_sols = []
    for i in range(trace_len):
        console.rule(f":clock10: Time slot {i}")
        _, sols = solve_for_one_timeslot(i)

        all_sols.append(sols)

    with open(f"{DIR_OUT}/sols_malloovia.p", "wb") as f:
        pickle.dump(all_sols, f)


if __name__ == "__main__":
    solve_all_time_slots_individually()
