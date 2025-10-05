import torch
import numpy as np
import math

class MppiplanSolver:
    def __init__(self, 
                delta_t: float = 0.1,
                wheel_base: float = 3.0,
                max_steer_abs: float = 1.0,
                max_vel_abs: float = 5.0,
                ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
                horizon_step_T: int = 20,
                number_of_samples_K: int = 200,
                param_exploration: float = 0.0,
                param_lambda: float = 50.0,
                param_alpha: float = 1.0,
                sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
                stage_cost_weight: np.ndarray = np.array([20.0, 20.0, 20.0, 1.0]),
                terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 50.0, 1.0]),
                visualize_optimal_traj: bool = True,
                visualze_sampled_trajs: bool = True,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """初始化GPU加速的MPPI参数"""
        self.dim_x = 4
        self.dim_u = 2
        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.device = device  # 设备选择（GPU或CPU）

        # 算法参数（转换为GPU张量）
        self.param_exploration = param_exploration
        self.param_lambda = torch.tensor(param_lambda, device=device)
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)
        
        # 协方差矩阵及其逆（转换为GPU张量）
        self.Sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        self.inv_Sigma = torch.inverse(self.Sigma)
        
        # 成本权重（转换为GPU张量）
        self.stage_cost_weight = torch.tensor(stage_cost_weight, device=device, dtype=torch.float32)
        self.terminal_cost_weight = torch.tensor(terminal_cost_weight, device=device, dtype=torch.float32)
        
        # 车辆参数
        self.delta_t = delta_t
        self.wheel_base = wheel_base
        self.max_steer_abs = max_steer_abs
        self.max_vel_abs = max_vel_abs
        
        # 参考路径（转换为GPU张量加速最近点搜索）
        self.ref_path = torch.tensor(ref_path, device=device, dtype=torch.float32)
        
        # 控制序列（GPU张量）
        self.u_prev = torch.zeros((self.T, self.dim_u), device=device, dtype=torch.float32)
        
        # 最近点索引
        self.prev_waypoints_idx = 0
        
        # 可视化开关
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

    def calc_control_input(self, observed_x: np.ndarray):
        """GPU加速的最优控制计算"""
        # 观测状态转换为GPU张量（[4] -> [1,4]）
        x0 = torch.tensor(observed_x, device=self.device, dtype=torch.float32).squeeze().unsqueeze(0)
        
        # 查找最近参考点
        self._get_nearest_waypoint(x0[0, 0], x0[0, 1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0] - 1:
            raise IndexError("[ERROR] Reached end of reference path")
        
        # 生成噪声（GPU上批量生成 K x T x 2 的噪声）
        epsilon = self._calc_epsilon()  # 形状: [K, T, 2]
        
        # 生成带噪声的控制输入（批量操作）
        u = self.u_prev.clone()  # 形状: [T, 2]
        v = torch.zeros((self.K, self.T, self.dim_u), device=self.device, dtype=torch.float32)
        
        # 区分探索性采样和利用性采样（向量化操作）
        explore_mask = torch.arange(self.K, device=self.device) >= (1.0 - self.param_exploration) * self.K
        v[~explore_mask] = u + epsilon[~explore_mask]  # 利用性采样（大部分）
        v[explore_mask] = epsilon[explore_mask]  # 探索性采样（小部分）
        
        # 批量计算所有轨迹的成本（GPU并行核心）
        S = self._batch_compute_costs(x0, v, u)  # 形状: [K]
        
        # 计算权重（向量化操作）
        w = self._compute_weights(S)  # 形状: [K]
        
        # 加权融合噪声（批量矩阵运算）
        w_epsilon = torch.sum(w.view(-1, 1, 1) * epsilon, dim=0)  # 形状: [T, 2]
        
        # 平滑控制输入
        w_epsilon = self._moving_average_filter(w_epsilon, window_size=10)
        
        # 更新控制序列
        u += w_epsilon
        
        # 计算最优轨迹（可选）
        optimal_traj = self._compute_optimal_trajectory(x0, u) if self.visualize_optimal_traj else None
        
        # 计算采样轨迹（可选）
        sampled_traj_list = self._compute_sampled_trajectories(x0, v) if self.visualze_sampled_trajs else None
        
        # 更新历史控制序列
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        
        # 返回CPU numpy数组（便于后续处理）
        return (
            optimal_traj.cpu().numpy() if optimal_traj is not None else None,
            sampled_traj_list.cpu().numpy() if sampled_traj_list is not None else None,
            u[0].cpu().numpy()
        )

    def _calc_epsilon(self):
        """在GPU上批量生成多元正态噪声"""
        # 生成 K*T 个噪声向量，再重塑为 [K, T, 2]
        mean = torch.zeros(self.dim_u, device=self.device)
        epsilon = torch.distributions.MultivariateNormal(mean, covariance_matrix=self.Sigma).sample(
            (self.K, self.T)
        )
        return epsilon  # 形状: [K, T, 2]

    def _batch_compute_costs(self, x0, v, u):
        """
        批量计算所有K条轨迹的总成本（GPU并行）
        x0: 初始状态 [1, 4]
        v: 带噪声的控制输入 [K, T, 2]
        u: 基础控制输入 [T, 2]
        """
        K, T = self.K, self.T
        total_cost = torch.zeros(K, device=self.device)
        
        # 初始化所有轨迹的状态 [K, 4]
        x = x0.repeat(K, 1)  # 复制初始状态到K条轨迹
        
        for t in range(T):
            # 批量限幅控制输入 [K, 2]
            u_clamped = self._g(v[:, t, :])  # 每条轨迹的t时刻控制输入
            
            # 批量更新状态 [K, 4] -> [K, 4]
            x = self._F(x, u_clamped)
            
            # 批量计算阶段成本 [K]
            current_vel = u_clamped[:, 0]  # 每条轨迹的当前速度 [K]
            stage_cost = self._c(x, current_vel)  # 阶段成本 [K]
            
            # 批量计算控制成本（向量化矩阵运算）
            control_cost = self.param_gamma * torch.sum(
                u[t].unsqueeze(0) @ self.inv_Sigma @ v[:, t, :].unsqueeze(2), 
                dim=(1, 2)
            )  # 形状: [K]
            
            # 累加成本
            total_cost += stage_cost + control_cost
        
        # 添加终端成本
        total_cost += self._phi(x, current_vel)
        
        return total_cost

    def _F(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        批量状态转移（GPU并行）
        x: [K, 4] 状态 (x, y, theta, beta)
        v: [K, 2] 控制输入 (vel, steer)
        返回: [K, 4] 下一状态
        """
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        theta = x[:, 2]
        vel = v[:, 0]
        steer = v[:, 1]
        
        # 批量计算运动学模型（所有操作均为向量化）
        dt = self.delta_t
        l = self.wheel_base
        
        new_x = x_pos + vel * torch.cos(theta) * dt
        new_y = y_pos + vel * torch.sin(theta) * dt
        new_theta = theta + vel * torch.tan(steer) / l * dt
        new_theta = (new_theta + 2 * math.pi) % (2 * math.pi)  # 归一化航向角
        new_beta = steer
        
        return torch.stack([new_x, new_y, new_theta, new_beta], dim=1)

    def _g(self, v: torch.Tensor) -> torch.Tensor:
        """批量控制输入限幅 [K, 2] -> [K, 2]"""
        v_clamped = v.clone()
        v_clamped[:, 0] = torch.clamp(v_clamped[:, 0], -self.max_vel_abs, self.max_vel_abs)
        v_clamped[:, 1] = torch.clamp(v_clamped[:, 1], -self.max_steer_abs, self.max_steer_abs)
        return v_clamped

    def _c(self, x: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        """批量计算阶段成本 [K, 4] -> [K]"""
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        theta = x[:, 2]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        # 批量查找每条轨迹当前位置的最近参考点
        _, ref_x, ref_y, ref_theta, ref_v = self._batch_get_nearest_waypoint(x_pos, y_pos)
        
        # 向量化计算成本
        dx = x_pos - ref_x
        dy = y_pos - ref_y
        d_theta = theta - ref_theta
        d_vel = current_vel - ref_v
        
        return (
            self.stage_cost_weight[0] * dx**2 +
            self.stage_cost_weight[1] * dy**2 +
            self.stage_cost_weight[2] * d_theta**2 +
            self.stage_cost_weight[3] * d_vel**2
        )

    def _batch_get_nearest_waypoint(self, x: torch.Tensor, y: torch.Tensor):
        """批量查找K条轨迹位置对应的最近参考点（GPU加速）"""
        K = x.shape[0]
        prev_idx = self.prev_waypoints_idx
        end_idx = min(prev_idx + 200, self.ref_path.shape[0])
        ref_segment = self.ref_path[prev_idx:end_idx]  # 参考路径片段
        
        # 计算所有轨迹到参考点的距离（向量化）
        dx = x.view(-1, 1) - ref_segment[:, 0].view(1, -1)  # [K, N]
        dy = y.view(-1, 1) - ref_segment[:, 1].view(1, -1)  # [K, N]
        dist_sq = dx**2 + dy**2  # [K, N]
        
        # 找到每条轨迹的最近点索引
        min_idx = torch.argmin(dist_sq, dim=1)  # [K]
        nearest_idx = prev_idx + min_idx  # [K]
        
        # 提取参考点信息
        ref_x = ref_segment[min_idx, 0]
        ref_y = ref_segment[min_idx, 1]
        ref_theta = ref_segment[min_idx, 2]
        ref_v = ref_segment[min_idx, 3]
        
        return nearest_idx, ref_x, ref_y, ref_theta, ref_v

    # 以下为其他辅助函数（与CPU版本逻辑一致，但基于PyTorch实现）
    def _phi(self, x: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        """批量计算终端成本"""
        x_pos = x[:, 0]
        y_pos = x[:, 1]
        theta = x[:, 2]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        _, ref_x, ref_y, ref_theta, ref_v = self._batch_get_nearest_waypoint(x_pos, y_pos)
        
        dx = x_pos - ref_x
        dy = y_pos - ref_y
        d_theta = theta - ref_theta
        d_vel = current_vel - ref_v
        
        return (
            self.terminal_cost_weight[0] * dx**2 +
            self.terminal_cost_weight[1] * dy**2 +
            self.terminal_cost_weight[2] * d_theta**2 +
            self.terminal_cost_weight[3] * d_vel**2
        )

    def _compute_weights(self, S: torch.Tensor) -> torch.Tensor:
        """向量化计算权重"""
        rho = torch.min(S)
        exp_terms = torch.exp((-1.0 / self.param_lambda) * (S - rho))
        eta = torch.sum(exp_terms)
        return exp_terms / eta

    def _moving_average_filter(self, xx: torch.Tensor, window_size: int = 10) -> torch.Tensor:
        """彻底解决尺寸匹配的GPU滑动平均滤波：有效卷积+手动补值"""
        # 1. 初始化输出（与输入形状完全一致：[T, dim]）
        T, dim = xx.shape
        xx_mean = torch.zeros_like(xx, device=self.device)
        
        # 2. 定义滑动平均卷积核（权重归一化）
        kernel = torch.ones(window_size, device=self.device, dtype=torch.float32) / window_size
        # 重塑卷积核为conv1d要求的格式：[out_channel, in_channel, kernel_size]
        kernel = kernel.view(1, 1, -1)
        
        for d in range(dim):
            # --------------------------
            # 步骤1：提取当前维度的控制序列（[T]）
            # --------------------------
            x = xx[:, d].view(1, 1, T)  # 重塑为conv1d输入格式：[batch=1, channel=1, length=T]
            
            # --------------------------
            # 步骤2：执行有效卷积（无padding，输出长度会缩短）
            # --------------------------
            # 有效卷积输出长度 = T - window_size + 1（如T=20、window_size=10时，输出长度=11）
            conv_out = torch.conv1d(input=x, weight=kernel, padding=0, stride=1)
            conv_out = conv_out.squeeze()  # 压缩为1D张量：[T - window_size + 1]
            
            # --------------------------
            # 步骤3：手动补值，将输出长度补到T（关键！）
            # --------------------------
            # 计算需要补的长度：前补n_left个，后补n_right个
            n_left = window_size // 2 - 1  # 前补4个（window_size=10时）
            n_right = window_size // 2      # 后补5个（window_size=10时）
            # 补值策略：复制边界值（避免引入突变）
            left_pad = torch.full((n_left,), conv_out[0], device=self.device)  # 前补：用第一个元素
            right_pad = torch.full((n_right,), conv_out[-1], device=self.device)  # 后补：用最后一个元素
            # 拼接得到最终长度=T的序列
            padded_out = torch.cat([left_pad, conv_out, right_pad], dim=0)
            
            # --------------------------
            # 步骤4：原有边界修正（保持逻辑不变）
            # --------------------------
            n_conv = math.ceil(window_size / 2)
            padded_out[0] *= window_size / n_conv
            for i in range(1, n_conv):
                padded_out[i] *= window_size / (i + n_conv)
                padded_out[-i] *= window_size / (i + n_conv - (window_size % 2))
            
            # --------------------------
            # 步骤5：赋值到输出（尺寸完全匹配）
            # --------------------------
            xx_mean[:, d] = padded_out

        return xx_mean

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """单一点的最近参考点查找（用于初始化）"""
        prev_idx = self.prev_waypoints_idx
        end_idx = min(prev_idx + 100, self.ref_path.shape[0])
        ref_segment = self.ref_path[prev_idx:end_idx]
        
        dx = x - ref_segment[:, 0]
        dy = y - ref_segment[:, 1]
        dist_sq = dx**2 + dy**2
        min_idx = torch.argmin(dist_sq)
        nearest_idx = prev_idx + min_idx
        
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx.item()  # 转为Python标量
        
        return (
            nearest_idx.item(),
            ref_segment[min_idx, 0].item(),
            ref_segment[min_idx, 1].item(),
            ref_segment[min_idx, 2].item(),
            ref_segment[min_idx, 3].item()
        )

    def _compute_optimal_trajectory(self, x0, u):
        """计算最优轨迹（批量单轨迹）"""
        traj = torch.zeros(self.T, self.dim_x, device=self.device)
        x = x0[0].clone()  # 初始状态 [4]
        for t in range(self.T):
            x = self._F(x.unsqueeze(0), u[t].unsqueeze(0))[0]  # 单步更新
            traj[t] = x
        return traj

    def _compute_sampled_trajectories(self, x0, v):
        """批量计算所有采样轨迹"""
        K, T = self.K, self.T
        traj = torch.zeros(K, T, self.dim_x, device=self.device)
        x = x0.repeat(K, 1)  # 初始化所有轨迹 [K, 4]
        
        for t in range(T):
            u_clamped = self._g(v[:, t, :])
            x = self._F(x, u_clamped)
            traj[:, t, :] = x
        
        return traj
    