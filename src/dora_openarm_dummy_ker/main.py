# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node that mimics OpenArm KER for testing."""

import dora
import numpy as np
import pyarrow as pa

ARM_HOME = np.array(
    [0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)

RIGHT_ARM_TARGET = np.array(
    [-0.8, 0.8, 0.3, 2.5, 0.5, 0.5, 0.5, -0.7], dtype=np.float32
)
LEFT_ARM_TARGET = np.array(
    [0.8, -0.8, -0.3, 2.5, -0.5, -0.5, -0.5, 0.7], dtype=np.float32
)

SEQUENCE_STEPS = 300


def _lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start + alpha * (end - start)


def main():
    """Act OpenArm KER as a leader of OpenArm."""
    rng = np.random.default_rng()
    step = 0
    node = dora.Node()

    for event in node:
        if event["type"] != "INPUT":
            continue

        t = step % (2 * SEQUENCE_STEPS)
        alpha = np.float32(1.0 - abs(t / SEQUENCE_STEPS - 1.0))
        step += 1

        right_position = _lerp(ARM_HOME, RIGHT_ARM_TARGET, alpha)
        left_position = _lerp(ARM_HOME, LEFT_ARM_TARGET, alpha)

        joystick_x = np.float32(2.0 * alpha - 1.0)
        joystick_y = np.float32(1.0 - 2.0 * alpha)
        joystick_button = np.int32(rng.integers(0, 2))

        pa_right = pa.array(right_position)
        pa_left = pa.array(left_position)

        node.send_output("right_position", pa_right)
        node.send_output("left_position", pa_left)
        node.send_output("right_follower_position", pa_right)
        node.send_output("left_follower_position", pa_left)

        node.send_output("joystick_x", pa.array([joystick_x], type=pa.float32()))
        node.send_output("joystick_y", pa.array([joystick_y], type=pa.float32()))
        node.send_output("joystick_button", pa.array([joystick_button], type=pa.int32()))


if __name__ == "__main__":
    main()
