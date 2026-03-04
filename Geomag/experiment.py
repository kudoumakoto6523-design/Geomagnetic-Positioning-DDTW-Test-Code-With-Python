import math
import sys

from Geomag.algorithms import (
    PF,
    get_heading_angle,
    get_mag,
    get_sensor,
    get_step_len,
    get_test_len,
    get_true_route,
    judge_step,
    visualize,
)
from Geomag.models import PFState


class Experiment:
    def __init__(self, context):
        self.context = context

    def run(self):
        geomag_map = self.context.geomag_map
        num_runs = self.context.num_runs
        window_size = self.context.window_size
        route_source = self.context.route_source
        sensor_source = self.context.sensor_source
        data_root = self.context.data_root
        uji_test_file = self.context.uji_test_file
        own_data_dir = self.context.own_data_dir

        for _ in range(num_runs):
            templist = []
            geomag_list = []
            route = get_true_route(
                source=route_source,
                data_root=data_root,
                uji_test_file=uji_test_file,
                own_data_dir=own_data_dir,
            )
            pf_state = PFState(init_pos=route[0], mag_map=geomag_map)  # or PF_init(...)
            pos_list = [pf_state.get_pos()]
            pdr_list = [pf_state.get_pos()]
            particle_counts = [len(pf_state.particles)]
            test_len = get_test_len(
                source=sensor_source,
                data_root=data_root,
                uji_test_file=uji_test_file,
                own_data_dir=own_data_dir,
            )

            def _print_progress(current, total, width=36):
                total = max(int(total), 1)
                current = min(max(int(current), 0), total)
                ratio = current / total
                filled = int(width * ratio)
                bar = "#" * filled + "-" * (width - filled)
                sys.stdout.write(f"\rProgress [{bar}] {current}/{total} ({ratio * 100:5.1f}%)")
                sys.stdout.flush()

            _print_progress(0, test_len)
            for i in range(test_len):
                mag, acc, gyro = get_sensor(
                    source=sensor_source,
                    data_root=data_root,
                    uji_test_file=uji_test_file,
                    own_data_dir=own_data_dir,
                )
                templist.append([acc, gyro, mag])
                judge = judge_step(templist)
                if judge:
                    stplen = get_step_len(templist)
                    heading_angle = get_heading_angle(templist)
                    geomag = get_mag()
                    geomag_list.append(geomag)
                    geomag_list_using = geomag_list[-window_size:]
                    pdr_x, pdr_y = pdr_list[-1]
                    pdr_list.append(
                        (
                            float(pdr_x + stplen * math.cos(heading_angle)),
                            float(pdr_y + stplen * math.sin(heading_angle)),
                        )
                    )
                    pos = PF(stplen, heading_angle, geomag_list_using, pf_state)
                    pos_list.append(pos)
                    particle_counts.append(len(pf_state.particles))
                    templist.clear()
                else:
                    _print_progress(i + 1, test_len)
                    continue
                _print_progress(i + 1, test_len)

            sys.stdout.write("\n")
            sys.stdout.flush()

            visualize(
                pos_list=pos_list,
                pdr_list=pdr_list,
                route=route,
                particle_counts=particle_counts,
                geomag_map=geomag_map,
                mode="ujimap",
                show=True,
            )
