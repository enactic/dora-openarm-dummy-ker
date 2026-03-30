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


def main():
    """Act OpenArm KER as a leader of OpenArm."""
    right_position = np.array(
        [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    left_position = np.array(
        [
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    node = dora.Node()
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        def perturb(position):
            noise = np.random.uniform(-0.3, 0.3, size=position.shape).astype(np.float32)
            return position + noise

        right_position = perturb(right_position)
        left_position = perturb(left_position)

        pa_right_position = pa.array(right_position)
        pa_left_position = pa.array(left_position)

        pa_right_follower_position = pa_right_position
        pa_left_follower_position = pa_left_position

        # (-1.0, 1.0] but real value range is (-1.0, 1.0)
        joystick_x = np.random.rand() * 2 - 1.0
        joystick_y = np.random.rand() * 2 - 1.0
        # 0 or 1
        joystick_button = np.random.randint(0, 2)

        node.send_output("right_position", pa_right_position)
        node.send_output("left_position", pa_left_position)

        node.send_output("right_follower_position", pa_right_follower_position)
        node.send_output("left_follower_position", pa_left_follower_position)

        node.send_output("joystick_x", pa.array([joystick_x], type=pa.float32()))
        node.send_output("joystick_y", pa.array([joystick_y], type=pa.float32()))
        node.send_output(
            "joystick_button", pa.array([joystick_button], type=pa.int32())
        )


if __name__ == "__main__":
    main()
