import numpy as np
from irsim.env import EnvBase
from MppiSolver import MppiplanSolver
from irsim.lib.path_planners.a_star import AStarPlanner
from matplotlib import pyplot as plt



class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):

        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)
        # 环境参数
        self.robot_goal = self.env.get_robot_info(0).goal
        self.lidar_r = 1.5
        data = self.env.get_map()
        # # 全局规划器
        start = self.env.get_robot_state().T
        start = start[0, :2].squeeze()
        end = self.robot_goal.T
        end = end[0, :2].squeeze()
        self.planner = AStarPlanner(data, data.resolution)
        self.global_path = self.planner.planning(start, end, show_animation=False)
        self.global_path = self.global_path[:, ::-1].T
        self.path_index = 0

       



        # 局部求解器

        self.solver = MppiplanSolver(np.array([0.0,0.0,0.0,0.0]),np.array([0.0,0.0,0.0,0.0]),[8.5, 1.5])

        # 速度指令
        self.v = 0.0
        self.w = 0.0
        self.robot_state = [[1.5],[8.5],[0.],[0.]]
        self.robot_state = np.array(self.robot_state)
        self.env.draw_trajectory(traj=self.global_path.T, traj_type="--y")

        # for obs in obs_list:
        #     self.env.draw_box(obs, refresh=True, color="-b")

    def step(self,):
        # 环境单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        # 环境可视化
        if self.env.display:
            self.env.render()

        # TODO
        self.robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()
        obs_list, center_list = self.scan_box(self.robot_state, scan_data)

        # 绘制障碍
        # for obs in obs_list:
        #     self.env.draw_box(obs, refresh=True, color="-b")
        # 计算临时目标点
        current_goal = self.compute_currentGoal(self.robot_state)
        self.env.draw_points(current_goal[:2], c="r", refresh=True)

        # 求解局部最优轨迹
        optimal_input, _,xy, sampled_traj_list= self.solver.calc_control_input(self.robot_state.squeeze(),current_goal.squeeze(),[8.5, 1.5])
        self.v = optimal_input[0]
        self.w = optimal_input[1]
        print(self.v,self.w)
        self.update(self.robot_state,optimal_input)
        # traj_xy = optimal_traj[:, :2]
        # x = xy[:, 0]  # x坐标
        # y = xy[:, 1]  # y坐标
        #
        # plt.figure(figsize=(6, 4))
        # plt.plot(x, y, '-o')  # 连线+标点
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('20-points XY curve')
        # plt.axis('equal')  # 可选：x/y等比例
        # plt.grid(True)
        # plt.show()
        # 轨迹可视化
        traj_list = [np.array([[x], [y]]) for x, y in xy]
        self.env.draw_trajectory(traj_list, 'b-', refresh=True)






        # 是否抵达
        if self.env.robot.arrive:
            print("Goal reached")
            return True

        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True

        return False
    def scan_box(self, state, scan_data):

        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

        point_list = []
        obstacle_list = []
        center_list = []

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan_data['range_max'] - 0.1):
                point = np.array([[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]])
                point_list.append(point)

        if len(point_list) < 4:
            return obstacle_list, center_list

        else:
            point_array = np.hstack(point_list).T
            labels = DBSCAN(eps=0.2, min_samples=2).fit_predict(point_array)

            for label in np.unique(labels):
                if label == -1:
                    continue
                else:
                    point_array2 = point_array[labels == label]
                    rect = cv2.minAreaRect(point_array2.astype(np.float32))
                    box = cv2.boxPoints(rect)
                    center_local = np.array(rect[0]).reshape(2, 1)

                    vertices = box.T

                    trans = state[0:2]
                    rot = state[2, 0]
                    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                    global_vertices = trans + R @ vertices
                    center_global = trans + R @ center_local

                    obstacle_list.append(global_vertices)
                    center_list.append(center_global)

            return obstacle_list, center_list
    def compute_currentGoal(self, robot_state):
        rx, ry = robot_state[0], robot_state[1]
        path = self.global_path
        goal_index = 0

        # 1. 计算所有点到机器人的距离
        robot_xy = robot_state.reshape(-1)  # 将(2,1)重塑为(2,)

        # 计算路径上每个点到机器人位置的距离
        dists = np.linalg.norm(path- robot_xy[:2], axis=1)

        # 2. 更新 path_index 为最近点索引（防止倒退）
        nearest_idx = int(np.argmin(dists))
        self.path_index = max(self.path_index, nearest_idx)

        # 3. 从 path_index 开始找第一个距离 > lidar_r 的点
        found = False
        for i in range(self.path_index, len(path)):
            if dists[i] > self.lidar_r:
                goal_index = i
                found = True
                break

        # 4. 确定最终目标点
        if not found:
            # 如果没找到，使用全局目标点
            goal_index = len(path) - 1
            target_x, target_y = path[goal_index]
            target_theta = self.robot_goal[-1]
        else:
            # 如果找到，使用路径上的点
            target_x, target_y = path[goal_index]
            target_theta = np.arctan2(target_y - ry, target_x - rx)

        # 返回目标点和朝向
        return np.array([[target_x], [target_y], target_theta])

    # def update(self, u,robot_state):
    #     """计算下一时刻状态（运动学模型）"""
    #     delta_t = 0.1

    #     # steer = np.clip(u[0], -np.deg2rad(30.0),np.deg2rad(30.0) )
    #     # accel = np.clip(u[1], -2, 2)
    #     steer = u[0]
    #     accel = u[1]

    #     robot_state[0] += robot_state[3] * np.cos(robot_state[2]) * delta_t
    #     robot_state[1] += robot_state[3] * np.sin(robot_state[2]) * delta_t
    #     robot_state[2] += robot_state[3] / 0.25 * np.tan(steer) * delta_t
    #     robot_state[3] += accel * delta_t

    def update(self, robot_state: np.ndarray, u: list) -> np.ndarray:
        """
        对外接口保持第二种形式：
        robot_state = [x, y, yaw, v]        # 4×1 或 4 维向量
        u = [steer, accel]        # 2 维列表/数组
        返回 4×1 或 1 维向量（与输入形状一致）
        """
        # 统一转 1-D 数组，最后再 reshape 回去
        robot_state   = np.asarray(robot_state).ravel()
        steer, accel = u[0], u[1]
    
        x, y, yaw, v = robot_state

        # 内部状态扩展：把当前真实转角 δ 存在最后一个元素
        # 第一次调用时若长度=4，自动补 δ=0
        if len(robot_state) == 4:
            delta = 0.0
        else:
            delta = robot_state[4]

        # 1. 转向一阶动态  δ̇ = (steer - δ)/τ
        delta_dot = (steer - delta) / 0.05
        new_delta = delta + delta_dot * 0.1

        # 2. 速度积分
        new_v = v + accel * 0.1

        # 3. 运动学
        new_x   = x + v * np.cos(yaw) * 0.1
        new_y   = y + v * np.sin(yaw) * 0.1
        new_yaw = yaw + v / 0.25 * np.tan(new_delta) * 0.1

        # 4. 角度归一化
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))
        robot_state[0] = new_x
        robot_state[1] = new_y
        robot_state[2] = new_yaw
        robot_state[3] = new_v




