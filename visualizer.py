import time

import meshcat
import numpy as onp

class Visualizer(meshcat.Visualizer):

    def add_kite(self, params):
        self["kite"].set_object(
            meshcat.geometry.Box(
                onp.array(
                    [
                        params.chord,
                        params.wingspan,
                        onp.abs(params.tether_attachments[0][2]),
                    ]
                )
            )
        )

    def draw_state(self, state, rate=None):
        self["kite"].set_transform(onp.array(state.kite.pose().as_matrix(), dtype=onp.float64))

        if rate is not None:
            t = time.time()
            if hasattr(self, "last_draw_time"):
                t_remaining = self.last_draw_time + rate - t
                if t_remaining > 0:
                    time.sleep(t_remaining)

            self.last_draw_time = t

