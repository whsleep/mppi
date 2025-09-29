
import numpy as np
import math

class MppiplanSolver:
    def __init__(self, 
                delta_t: float = 0.1,
                wheel_base: float = 3.0, # [m]
                max_steer_abs: float = 1.0, # [rad]
                max_vel_abs: float = 5.0, # [m/s]
                ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
                horizon_step_T: int = 20,
                number_of_samples_K: int = 50,
                param_exploration: float = 0.0,
                param_lambda: float = 50.0,
                param_alpha: float = 1.0,
                sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
                stage_cost_weight: np.ndarray = np.array([20.0, 20.0, 20.0, 1.0]), # weight for [x, y, theta, beta]
                terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 50.0, 1.0]), # weight for [x, y, theta, beta]
                visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
                visualze_sampled_trajs = True, # if True, sampled trajectories are visualized
                 ):
  
        """初始化mppi参数"""
        self.dim_x = 4 # 状态维度
        self.dim_u = 2 # 控制维度
        self.T = horizon_step_T # 预测视野
        self.K = number_of_samples_K # 采样轨迹数量
        self.param_exploration = param_exploration  # mppi常量参数
        self.param_lambda = param_lambda  # mppi常量参数
        self.param_alpha = param_alpha # mppi常量参数
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # mppi常量参数
        self.Sigma = sigma # 噪声偏差
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # 车辆参数
        self.delta_t = delta_t # 仿真步长[s]
        self.wheel_base = wheel_base # 车辆轴距[m]
        self.max_steer_abs = max_steer_abs # 最大转向角[rad]
        self.max_vel_abs = max_vel_abs # 最大速度[m/s]
        self.ref_path = ref_path # 参考路径[n x 4]，每一行是[x, y, yaw, v]

        # 上次控制序列
        self.u_prev = np.zeros((self.T, self.dim_u))

        # 上次最近点索引
        self.prev_waypoints_idx = 0

    """
    计算最优控制
    """
    def calc_control_input(self, observed_x: np.ndarray):
        """calculate optimal control input"""
        # 加载上次计算的控制序列
        u = self.u_prev
        # 获取观测状态
        x0 = observed_x.squeeze()
        # 计算参考路径基于当前位置的最近索引
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError
        # 状态成本列表
        S = np.zeros((self.K))
        # 生成噪声
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)    
        # 准备控制输入序列
        v = np.zeros((self.K, self.T, self.dim_u)) 

        # loop for 0 ~ K-1 samples
        for k in range(self.K):         
            # 设定采样初始值
            x = x0
            current_vel = 0.0
            # 单条轨迹前向推进
            for t in range(1, self.T+1):
                # 添加噪声的比例
                if k < (1.0-self.param_exploration)*self.K:
                    # 在上次最优控制序列上添加噪声
                    v[k, t-1] = u[t-1] + epsilon[k, t-1]
                else:
                    # 仅添加噪声
                    v[k, t-1] = epsilon[k, t-1]
                # 前向推进
                u_clamped = self._g(v[k, t-1]) 
                x = self._F(x, u_clamped)
                current_vel = u_clamped[0] 
                # 添加阶段代价  
                S[k] += self._c(x, current_vel) + self.param_gamma * u[t-1].T @ np.linalg.inv(self.Sigma) @ v[k, t-1]
            # 添加终端代价
            S[k] += self._phi(x, current_vel)

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)
        # calculate w_k * epsilon_k
        w_epsilon = np.zeros((self.T, self.dim_u))
        for t in range(0, self.T): # loop for time step t = 0 ~ T-1
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)
        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x = x0
            for t in range(0, self.T): # loop for time step t = 0 ~ T-1
                x = self._F(x, self._g(u[t]))
                optimal_traj[t] = x

        # # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        sorted_idx = np.argsort(S) # sort samples by state cost, 0th is the best sample
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(0, self.T): # loop for time step t = 0 ~ T-1
                    x = self._F(x, self._g(v[k, t]))
                    sampled_traj_list[k, t] = x

        # update privious control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        
        return optimal_traj,sampled_traj_list,np.array([float(u[0][0]), float(u[0][1])]) # return first control input [v, steer]

    """
    生成噪声序列
    """
    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        # 检查协方差矩阵是否与控制维度匹配
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError

        # 零均值高斯噪声 1x(size_dim_u)
        mu = np.zeros((size_dim_u))
        # 生成噪声
        # size_sample条轨迹，每个轨迹size_time_step个采样点，每个采样点的噪声维度为2
        # epsilon为 (size_sample) x (size_time_step) x (2) 维度
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    """
    截断超出限制部分
    """
    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        # limit control inputs
        v[0] = np.clip(v[0], -self.max_vel_abs, self.max_vel_abs) # limit steering input
        v[1] = np.clip(v[1], -self.max_steer_abs, self.max_steer_abs) # limit acceleraiton input
        return v
    
    """
    状态转移过程
    """
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        x, y, theta, beta= x_t
        vel, steer = v_t

        # prepare params
        l = self.wheel_base
        dt = self.delta_t

        # update state variables
        new_x = x + vel * np.cos(theta) * dt
        new_y = y + vel * np.sin(theta) * dt
        new_theta = theta + vel * np.tan(steer)/l * dt
        new_theta = ((new_theta + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]
        new_beta = steer

        # return updated state
        x_t_plus_1 = np.array([new_x, new_y, new_theta, new_beta])
        return x_t_plus_1

    """
    根据状态计算阶段成本
    """
    def _c(self, x_t: np.ndarray, current_vel: float) -> float:
        """calculate stage cost"""
        # parse x_t
        x, y, theta, beta = x_t
        theta = ((theta + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_theta, ref_v = self._get_nearest_waypoint(x, y)
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                     self.stage_cost_weight[2]*(theta-ref_theta)**2 + self.stage_cost_weight[3]*(current_vel - ref_v)**2
        return stage_cost
    
    """
    根据状态计算终端成本
    """
    def _phi(self, x_T: np.ndarray, current_vel: float) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, y, theta, beta = x_T
        theta = ((theta + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate terminal cost
        _, ref_x, ref_y, ref_theta, ref_v = self._get_nearest_waypoint(x, y)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(theta-ref_theta)**2 + self.terminal_cost_weight[3]*(current_vel - ref_v)**2
        return terminal_cost

    """
    计算权重信息
    """
    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )

        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w
    
    """
    滑动窗口滤波
    """
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control inputs.
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean
    

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""
        # 仅仅检索前方一定范围的点以节省计算时间
        SEARCH_IDX_LEN = 200 # [points] forward search range
        # 记录上次最近点的索引以加速搜索
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        # 在已有检索上递增索引
        nearest_idx = d.index(min_d) + prev_idx

        # 获取参考点信息
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]

        # 更新最近点索引
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v