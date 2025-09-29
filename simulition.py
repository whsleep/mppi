import numpy as np
from irsim.env import EnvBase
from MppiSolver import MppiplanSolver



class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):

        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)
        # 环境参数
        self.robot_goal = self.env.get_robot_info(0).goal
        self.lidar_r = 1.5
        # 从csv提取参考路径
        self.ref_path = np.genfromtxt('./ovalpath.csv', delimiter=',', skip_header=1)
        xyz_matrix = self.ref_path[:, 0:3]
        formatted_path_list = []
        for waypoint in xyz_matrix:
            column_vector = waypoint.reshape((3, 1))
            formatted_path_list.append(column_vector)
        self.env.draw_trajectory(formatted_path_list, traj_type='-k') 
        self.prev_waypoints_idx = 0

        # 局部求解器
        self.solver = MppiplanSolver(ref_path=self.ref_path)


    def step(self,):
        # 环境单步仿真
        # self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        # 环境可视化
        if self.env.display:
            self.env.render()

        self.robot_state = self.env.get_robot_state()
        opt_traj ,samp_traj, action_input_list = self.solver.calc_control_input(self.robot_state)
        # print(samp_traj.shape[0])
        for i in range(samp_traj.shape[0]):
            list_of_arrays = [np.array(row).reshape(-1, 1) for row in samp_traj[i]]
            self.env.draw_trajectory(list_of_arrays, traj_type='-g', refresh=True)
        # 将二维数组转换为列表
        list_of_arrays = [np.array(row).reshape(-1, 1) for row in opt_traj]
        self.env.draw_trajectory(list_of_arrays, traj_type='-r', refresh=True)
        # 执行动作
        # [v, steer]
        self.env.step(action_id=0, action=action_input_list)

        # 是否抵达
        if self.env.robot.arrive:
            print("Goal reached")
            return True

        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True

        return False



