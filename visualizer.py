import time

import meshcat
from meshcat import geometry as g
import numpy as onp


class Visualizer(meshcat.Visualizer):
    def add_kite(self, params):
        self["kite"].set_object(
            g.Box(
                onp.array(
                    [
                        params.chord,
                        params.wingspan,
                        onp.abs(params.tether_attachments[0][2]),
                    ]
                )
            )
        )

        # Keep a copy of the params for subsequent drawing
        self.params = params

    def draw_state(self, state, rate=None):
        self["kite"].set_transform(
            onp.array(state.kite.pose().as_matrix(), dtype=onp.float64)
        )

        for i in range(len(self.params.tether_attachments)):
            points = onp.stack(
                [
                    state.kite.pose() @ self.params.tether_attachments[i],
                    self.params.anchor_positions[i],
                ]
            )
            self["tethers"][str(i)].set_object(
                g.LineSegments(g.PointsGeometry(points.T))
            )

        if rate is not None:
            t = time.time()
            if hasattr(self, "last_draw_time"):
                t_remaining = self.last_draw_time + rate - t
                if t_remaining > 0:
                    time.sleep(t_remaining)

            self.last_draw_time = t
