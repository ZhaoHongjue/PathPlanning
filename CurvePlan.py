from msilib.schema import Error
import numpy as np
import matplotlib.pyplot as plt

def TrapezoidPlan(init: float or np.ndarray, final: float or np.ndarray, 
                  acc: float or np.ndarray, t_f: float, t: float) -> float or np.ndarray:
    acc = (final - init) / abs(final - init) * abs(acc)
    tmp = acc**2 * t_f**2 - 4*acc*(final - init)
    
    if type(tmp)!=np.ndarray:
        assert tmp >= 0
    else:
        assert tmp.all() >=0
        
    # 求对应时刻的位置
    if type(init) != np.ndarray:
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
        elif t_f - t_acc < t < t_f:
            Delta = final - init - 0.5 * acc * (t_f - t)**2
        else:
            raise Error
        return init + Delta
    
    else:
        pos = np.zeros_like(init)
        for i in range(len(init)):
            pos[i] = TrapezoidPlan(init[i], final[i], acc[i], t_f, t)
        return pos
    

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
    # 类型检查，初末位置和加速度的形状应该一样
    assert init.shape == final.shape == acc.shape
    # 改变加速度的方向
    acc_x = (final[0] - init[0]) / abs(final[0] - init[0]) * abs(acc[0])
    Delta_x = acc_x**2 * t_f**2 - 4 * acc_x * (final[0] - init[0])
    
    # 求解加速减速的时间
    if acc[0] > 0:
        tx_acc = 0.5 * (t_f - np.sqrt(Delta_x) / acc_x)
    else:
        tx_acc = 0.5 * (t_f  + np.sqrt(Delta_x) / acc_x)
    vx_max = acc_x * tx_acc
    
    # 进染色槽和出染色槽的位置：(0.1, 0.35, 0.15), (-0.1, 0.35, 0.15)
    mid_x = np.array([0.1, -0.1])
    mid_y, mid_z = 0.35, 0.15
    # 进染色槽和出染色槽的时间
    mid_t = (mid_x - init[0] - 0.5*acc_x*tx_acc**2) / vx_max + tx_acc
    
    print('mid_delta', mid_x - init[0])
    print('acc_delta', 0.5*acc_x*tx_acc**2)
    print('mid_t', mid_t)
    
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
        raise Error
    return pos
    

if __name__ == '__main__':
    init = np.array([0.4, -0.12, 0.40])
    final = np.array([-0.35, 0.0, 0.40])
    acc = np.array([0.1, 0.1, 0.1])
    
    v_i, a_i = np.zeros(3), np.zeros(3)
    v_f, a_f = 0.1 * np.ones(3), 1 * np.ones(3)
    t_f = 20
    
    t_f = 9
    print(PathGenerate(init, final, acc, t_f, 4.40707142))
    
    # ts = np.linspace(0, t_f, 200)
    # pos = np.zeros((200, 3))
    # for i in range(200):
    #     pos[i] = PathGenerate(init, final, acc, t_f, ts[i])
    #     # pos[i] = Poly3Plan(init, final, v_i, v_f, t_f, ts[i])
    
    # plt.plot(ts, pos[:, 0], label = 'x')
    # # plt.plot(ts, pos[:, 1], label = 'y')
    # # plt.plot(ts, pos[:, 2], label = 'z')
    # plt.plot(ts, 0.1 * np.ones_like(ts), label = '0.1')
    # plt.plot(ts, -0.1 * np.ones_like(ts), label = '-0.1')
    # plt.legend()
    # plt.show()
            
        
    
    