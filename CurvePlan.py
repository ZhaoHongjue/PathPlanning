import numpy as np
import matplotlib.pyplot as plt

def TrapezoidPlan(init: float or np.ndarray, final: float or np.ndarray, 
                  acc: float or np.ndarray, t_f: float, t: float) -> float or np.ndarray:
    
    # 求对应时刻的位置
    if type(init) != np.ndarray:
        # 求解加速度
        acc = acc if final >= init else -acc
        tmp = acc**2 * t_f**2 - 4*acc*(final - init)
        assert tmp >= 0
        
        # 求解加速减速的时间
        if acc > 0:
            t_acc = 0.5 * (t_f - np.sqrt(tmp) / acc)
        else:
            t_acc = 0.5 * (t_f + np.sqrt(tmp) / acc)
        v_max = acc * t_acc
        
        # 根据不同时间段返回值
        if 0 <= t <= t_acc:
            Delta = 0.5 * acc * t**2
        elif t_acc < t <= t_f - t_acc:
            Delta = 0.5 * acc * t_acc**2 + v_max * (t - t_acc)
        elif t_f - t_acc < t <= t_f:
            Delta = final - init - 0.5 * acc * (t_f - t)**2
        else:
            raise ValueError
        return init + Delta
    
    else:
        pos = np.zeros_like(init)
        for i in range(len(init)):
            pos[i] = TrapezoidPlan(init[i], final[i], acc[i], t_f, t)
        return pos
    
def TrapezoidPlanMid(points: list or np.ndarray, ts: list or np.ndarray, 
                     acc: float or int, t: float or int):
    assert len(points) == len(ts)
    
    deltas = [points[i] - points[i-1] for i in range(1, len(points))]
    delta_ts = [ts[i] - ts[i-1] for i in range(1, len(points))]
    
    vels = [deltas[i] / delta_ts[i] for i in range(len(deltas))]
    acc_0 = acc if deltas[0] > 0 else -acc
    t_accs_0 = delta_ts[0] - np.sqrt(delta_ts[0]**2 - 2 * deltas[0] / acc_0)
    vels[0] = deltas[0] / (delta_ts[0] - 0.5 * t_accs_0)
    
    acc_n = acc if deltas[-1] < 0 else -acc
    t_accs_n = delta_ts[-1] - np.sqrt(delta_ts[-1]**2 + 2 * deltas[-1] / acc_n)
    vels[-1] = deltas[-1] / (delta_ts[-1] - 0.5*t_accs_n)
    
    accs = [acc if vels[i] > vels[i-1] else -acc for i in range(len(vels))] + [acc_n]
    t_accs = [t_accs_0] + [(vels[i] - vels[i-1]) / accs[i] for i in range(1, len(vels))] + [t_accs_n]
    t_vels = [delta_ts[i] - 0.5 * (t_accs[i] + t_accs[i+1]) for i in range(len(t_accs)-1)]
    
    t_vels[0] = delta_ts[0] - t_accs[0] - 0.5 * t_accs[1]
    t_vels[-1] = delta_ts[-1] - t_accs[-1] - 0.5 * t_accs[-2]
    
    mean_speed_delta = [vels[i] * t_vels[i] for i in range(len(vels))]
    acc_delta = [0.5 * accs[0] * t_accs[0]**2] + [vels[i-1] * t_accs[i] + 0.5 * accs[i] * t_accs[i]**2 for i in range(1, len(accs))]
    
    # print('accs:    ', accs)
    # print('vels:    ', vels)
    # print('t_accs:  ', t_accs)
    # print('t_vels:  ',t_vels)
    # print(acc_delta)
    
    ts_flag, delta_flag = [0], [0]
    for i in range(len(t_vels) + len(t_accs)):
        if i % 2 == 0:
            idx = i // 2
            ts_flag.append(t_accs[idx])
            delta_flag.append(acc_delta[idx])
            
        else:
            idx = (i - 1) // 2
            ts_flag.append(t_vels[idx])
            delta_flag.append(mean_speed_delta[idx])
            
    ts_point = [sum(ts_flag[:i]) for i in range(1, len(ts_flag) + 1)]
    delta_point = [points[0] + sum(delta_flag[:i]) for i in range(1, len(delta_flag) + 1)]
    assert len(ts_point) == len(delta_point)
    
    for i in range(len(ts_point)-1):
        if ts_point[i] <= t <= ts_point[i+1]:
            t_delta = t - ts_point[i]
            if i % 2 == 0:
                idx = i // 2
                if idx == 0:
                    delta = 0.5 * accs[idx] * t_delta**2
                else:
                    delta = vels[idx-1] * t_delta + 0.5 * accs[idx] * t_delta**2
            else:
                idx = (i - 1) // 2
                delta = vels[idx] * t_delta
            return delta + delta_point[i]

def Polynomial3Plan(init: float or np.ndarray, final: float or np.ndarray, 
             v_i: float or np.ndarray, v_f: float or np.ndarray,
             t_f: float, t: float) -> float or np.ndarray: 
    assert type(init) == type(final) == type(v_i) == type(v_f)
    # 时间构成的矩阵，分别对应初位置、末位置、初速度、末速度
    tfMat = np.array([
        [0,             0,              0,              1],
        [t_f**3,        t_f**2,         t_f,            1],
        [0,             0,              1,              0],
        [3*t_f**2,      2*t_f,          1,              0],
    ])
    
    inv_tMatrix = np.linalg.inv(tfMat)
    if type(init) != np.ndarray:
        rhs = np.array([[init, final, v_i, v_f]]).T
        kArray = inv_tMatrix @ rhs
    else:
        kArray = []
        for i in range(len(init)):
            rhs = np.array([[init[i], final[i], v_i[i], v_f[i]]]).T
            kArray.append(inv_tMatrix @ rhs)
        kArray = np.concatenate(kArray, axis = 1)
    
    tvec = np.array([[t**3, t**2, t, 1]])
    
    if type(init) != np.ndarray:
        return (tvec @ kArray)[0, 0]
    else:
        return (tvec @ kArray).reshape(-1)

def Polynomial5Plan(init: float or np.ndarray, final: float or np.ndarray, 
             v_i: float or np.ndarray, v_f: float or np.ndarray,
             a_i: float or np.ndarray, a_f: float or np.ndarray,
             t_f: float, t: float) -> float or np.ndarray: 
    assert type(init) == type(final) == type(v_i) == type(v_f) == type(a_i) == type(a_f)
    # 时间构成的矩阵，分别对应初位置、末位置、初速度、末速度、初加速度、末加速度
    tfMat = np.array([
        [0,             0,              0,              0,              0,          1],
        [t_f**5,        t_f**4,         t_f**3,         t_f**2,         t_f,        1],
        [0,             0,              0,              0,              1,          0],
        [5*t_f**4,      4*t_f**3,       3*t_f**2,       2*t_f,          1,          0],
        [0,             0,              0,              2,              0,          0],
        [20*t_f**3,     12*t_f**2,      6*t_f,          2,              0,          0]
    ])
    
    inv_tMatrix = np.linalg.inv(tfMat)
    if type(init) != np.ndarray:
        rhs = np.array([[init, final, v_i, v_f, a_i, a_f]]).T
        kArray = inv_tMatrix @ rhs
    else:
        kArray = []
        for i in range(len(init)):
            rhs = np.array([[init[i], final[i], v_i[i], v_f[i], a_i[i], a_f[i]]]).T
            kArray.append(inv_tMatrix @ rhs)
        kArray = np.concatenate(kArray, axis = 1)
    
    tvec = np.array([[t**5, t**4, t**3, t**2, t, 1]])
    
    if type(init) != np.ndarray:
        return (tvec @ kArray)[0, 0]
    else:
        return (tvec @ kArray).reshape(-1)

def PathGenerate(init: np.ndarray, final: np.ndarray, acc: np.ndarray,
                 t_f: float, t: float) -> np.ndarray:
    # 检查：初末位置和加速度的形状应该一样
    assert init.shape == final.shape == acc.shape
    # 改变加速度的方向
    acc_x = (final[0] - init[0]) / abs(final[0] - init[0]) * abs(acc[0])
    Delta_x = acc_x**2 * t_f**2 - 4 * acc_x * (final[0] - init[0])
    
    # 求解加速减速的时间
    tx_acc1 = 0.5 * (t_f - np.sqrt(Delta_x) / acc_x)
    tx_acc2 = 0.5 * (t_f  + np.sqrt(Delta_x) / acc_x)
    tx_acc = min(tx_acc1, tx_acc2)
    vx_max = acc_x * tx_acc
    
    # 进染色槽和出染色槽的位置：(0.1, 0.35, 0.15), (-0.1, 0.35, 0.15)
    mid_x = np.array([0.1, -0.1])
    mid_y, mid_z = 0.35, 0.15
    # 进染色槽和出染色槽的时间
    mid_t = (mid_x - init[0] - 0.5*acc_x*tx_acc**2) / vx_max + tx_acc
    
    # print(f'tx_acc1: {tx_acc1},\ntx_acc2: {tx_acc2},\ntx_acc: {tx_acc}')
    # print('vx_max           ', vx_max)
    # print('mid_delta        ', mid_x - init[0])
    # print('acc_delta        ', 0.5*acc_x*tx_acc**2)
    # print('mid_t-tx_acc     ', mid_t - tx_acc)
    # print('mid_t            ', mid_t)
    
    pos = np.zeros(3)
    pos[0] = TrapezoidPlan(init[0], final[0], acc_x, t_f, t)
    if 0 <= t <= mid_t[0]:
        pos[1] = TrapezoidPlan(init[1], mid_y, acc[1], mid_t[0], t)
        pos[2] = TrapezoidPlan(init[2], mid_z, acc[2], mid_t[0], t)
    elif mid_t[0] < t <= mid_t[1]:
        pos[1:] = [mid_y, mid_z]
    elif mid_t[1] < t <= t_f:
        pos[1] = TrapezoidPlan(mid_y, final[1], acc[1], t_f - mid_t[1], t - mid_t[1])
        pos[2] = TrapezoidPlan(mid_z, final[2], acc[2], t_f - mid_t[1], t - mid_t[1])
    else:
        raise ValueError
    return pos
    
if __name__ == '__main__':
    init = np.array([0.4, -0.12, 0.40])
    final = np.array([-0.35, 0.0, 0.40])
    acc = np.array([0.5, 0.5, 0.5])
    t_f = 10
    
    # print(PathGenerate(init, final, acc, t_f, 20))
    
    ts = np.linspace(0, t_f, 200)
    pose = np.zeros((200, 6))
    for i in range(200):
        pose[i, :3] = PathGenerate(init, final, acc, t_f, ts[i])
        pose[i, 3:] = np.array([np.pi, 0, 0])
    
    # plt.plot(ts, pose[:, 0], label = 'x')
    # plt.plot(ts, pose[:, 1], label = 'y')
    # plt.plot(ts, pose[:, 2], label = 'z')
    # plt.plot(ts, 0.1 * np.ones_like(ts), label = '0.1')
    # plt.plot(ts, -0.1 * np.ones_like(ts), label = '-0.1')
    # plt.legend()
    # plt.show()
    
    import Kinematics
    joints = np.zeros((200, 6))
    fail_cnt, one_cnt, two_cnt, three_cnt, four_cnt = 0, 0, 0, 0, 0
    for i in range(200):
        res = Kinematics.IKSolver(pose[i])
        if len(res) == 0:
            fail_cnt += 1
        elif len(res) == 1:
            one_cnt += 1
            joints[i] = res.reshape(-1)
        elif len(res) == 2:
            two_cnt += 1
            joints[i] = res[0]
        elif len(res) == 3:
            three_cnt += 1
            joints[i] = res[0]
        elif len(res) == 4:
            four_cnt += 1
            joints[i] = res[0]
        else:
            print('fuck!')
    print(fail_cnt, one_cnt, two_cnt, three_cnt, four_cnt)
        
    
    