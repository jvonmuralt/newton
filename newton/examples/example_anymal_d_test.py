# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Anymal D
#
# Loads Anymal D model from mjcf file and compares dynamics with mujoco warp.
#Comment/uncomment these lineas dependent on which model needs to be simulated
#model = example.solver.mj_model
#data = example.solver.mj_data
#
###########################################################################

import os

import torch
import warp as wp
import newton
import newton.utils
import mujoco.viewer
import mujoco_warp as mjwarp


use_usd = False # loading usd or mjcf model in newton
show_viewer = False # show sim the viewer or print masses and inertias only
show_newton_model_in_viewer = False # show newton or mujoco model in viewer

class Example:
    def __init__(self, stage_path=None, headless=False):
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if use_usd:
            stage_path = os.path.join(script_dir, "assets", "anymal_d_simple_description", "anymal_d.usd")
            newton.utils.parse_usd(
            stage_path,
                builder,
                #floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                #ignore_inertial_definitions=False,
            )
        else:
            path = os.path.join(script_dir, "assets", "mjmodel.xml")
            newton.utils.parse_mjcf(
                path,
                builder,
                #floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                #ignore_inertial_definitions=False,
            )

        builder.add_ground_plane()


        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.7, 0.7]

   
        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        self.model= builder.finalize()  

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None if headless else newton.utils.SimRendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            torch_tensor = torch.zeros(18, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):

            with torch.no_grad():
                a = torch.zeros_like(self.joint_pos_initial) #+ 0.5 * self.rearranged_act
                a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
                a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
                wp.copy(
                    self.control.joint_target, a_wp
                )
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        help="Path to the output URDF file.",
    )
    parser.add_argument("--num-frames", type=int, default=1000000, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]


    def _compile_step(m, d):
        mjwarp.step(m, d)
        mjwarp.step(m, d)
        with wp.ScopedCapture() as capture:
            mjwarp.step(m, d)
        return capture.graph
    
    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, headless=args.headless)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        device = example.torch_device

        example.joint_pos_initial = torch.tensor(
            example.state_0.joint_q[7:], device=example.torch_device, dtype=torch.float32
        ).unsqueeze(0)
        example.joint_vel_initial = torch.tensor(
            example.state_0.joint_qd[6:], device=example.torch_device, dtype=torch.float32
        )
        example.act = torch.zeros(1, 12, device=example.torch_device, dtype=torch.float32)
        example.rearranged_act = torch.zeros(1, 12, device=example.torch_device, dtype=torch.float32)


        path = os.path.join(script_dir, "assets", "mjmodel_with_ground_plane.xml")
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for i in range(model.nbody):
            print("MASS of ", i, model.body_mass[i], example.solver.mj_model.body_mass[i])
            print("INERTIA of ", i, model.body_inertia[i], example.solver.mj_model.body_inertia[i])
        if not show_viewer:
            exit()
        
        if show_newton_model_in_viewer:
            model = example.solver.mj_model
            data = example.solver.mj_data
        m = mjwarp.put_model(model)
        d = mjwarp.put_data(model, data)
        i = 0

        viewer = mujoco.viewer.launch_passive(model, data)
        joint_pos_initial =  torch.tensor(data.qpos[7:].copy(), device=device).unsqueeze(0)
        joint_vel_initial =  torch.tensor(data.qvel[6:].copy(), device=device)
        act = torch.zeros(1, 12, device=device)
        rearranged_act = torch.zeros(1, 12, device=device)
        graph = _compile_step(m, d)

        with viewer:
            while i < 8000:
                wp.capture_launch(graph)
                wp.synchronize()
                i+=1
                mjwarp.get_data_into(data, model, d)
                viewer.sync()

