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

        for _ in range(num_runs):
            templist = []
            geomag_list = []
            route = get_true_route()
            pf_state = PFState(init_pos=route[0], mag_map=geomag_map)  # or PF_init(...)
            pos_list = [pf_state.get_pos()]
            test_len = get_test_len()

            for _ in range(test_len):
                mag, acc, gyro = get_sensor()
                templist.append([acc, gyro, mag])
                judge = judge_step(templist)
                if judge:
                    stplen = get_step_len(templist)
                    heading_angle = get_heading_angle(templist)
                    geomag = get_mag()
                    geomag_list.append(geomag)
                    geomag_list_using = geomag_list[-window_size:]
                    pos = PF(stplen, heading_angle, geomag_list_using, pf_state)
                    pos_list.append(pos)
                    templist.clear()
                else:
                    continue

            visualize(pos_list, route, geomag_map)
