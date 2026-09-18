"""The browser's physics must be the same physics.

`src/visualization/assistant/sim.js` is a hand port of `src/physics/spin.py`
and `simulate_with_spin`. Two copies of a model drift, and when the JS one
drifts the only symptom is a line on the advice page that looks plausible and
is wrong. So run both on the same shots and make them agree.

Skipped where node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.physics.simulator import simulate_with_spin

SIM_JS = Path(__file__).resolve().parent.parent / "src" / "visualization" / "assistant" / "sim.js"
LAYOUT = {"white": [700.0, 400.0], "yellow": [1500.0, 900.0], "red": [2300.0, 500.0]}
# Straight down the table, and a cut across it; with right side, none, left side.
SHOTS = [(0.0, 2699.0, 0.0), (0.0, 2699.0, 3.0), (0.0, 2699.0, -3.0),
         (35.0, 2400.0, 1.5), (150.0, 3000.0, -1.5)]


def _node_says():
    script = f"""
    const fs = require('fs');
    const SIM = eval(fs.readFileSync({json.dumps(str(SIM_JS))}, 'utf8') + '\\nSIM;');
    const layout = {json.dumps(LAYOUT)};
    const out = [];
    for (const [deg, speed, tips] of {json.dumps(SHOTS)}) {{
      const r = deg * Math.PI / 180;
      const shot = SIM.play(layout, 'white', [Math.cos(r) * speed, Math.sin(r) * speed], tips, 0);
      out.push({{ rest: shot.rest.white,
                 rails: shot.events.filter((e) => e.kind === 'cushion').map((e) => e.detail),
                 balls: shot.events.filter((e) => e.kind === 'ball').map((e) => e.detail) }});
    }}
    console.log(JSON.stringify(out));
    """
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_js_port_matches_the_python_model():
    theirs = _node_says()
    for (deg, speed, tips), js in zip(SHOTS, theirs):
        heading = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        mine = simulate_with_spin({k: np.array(v) for k, v in LAYOUT.items()}, "white",
                                  heading * speed, tips_side=tips, tips_vertical=0.0)
        rails = [detail for _f, kind, detail in mine.events if kind == "cushion"]
        balls = [detail for _f, kind, detail in mine.events if kind == "ball"]
        where = f"{deg}° {speed} mm/s {tips}팁"

        assert rails == js["rails"], f"{where}: 쿠션 차례가 다릅니다"
        assert balls == js["balls"], f"{where}: 맞은 공이 다릅니다"
        # Within a ball's width: the two integrate in the same steps, so any
        # real gap here is a difference in the model, not in the arithmetic.
        apart = float(np.hypot(*(np.asarray(js["rest"]) - mine.paths["white"][-1])))
        assert apart < 61.5, f"{where}: 멈춘 자리가 {apart:.0f} mm 떨어졌습니다"
