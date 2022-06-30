import copy
import numpy as np
from math import atan2, sqrt, sin, cos, acos

class Quaternion:
    '''
    定义四元数
    '''
    def __init__(self, array: list or np.ndarray) -> None:
        assert len(array) == 4
        self.num = np.asarray(array, dtype = np.float32)
        
    @property
    def conj(self):
        tmp = copy.deepcopy(self.num)
        tmp[1:] = -1 * tmp[1:]
        return Quaternion(tmp)

    @property
    def norm(self):
        return np.linalg.norm(self.num)
    
    @property
    def inverse(self):
        return self.conj / self.norm ** 2
        
    def __add__(self, q):
        return Quaternion(self.num + q.num)

    def __sub__(self, q):
        return Quaternion(self.num - q.num)
    
    def __mul__(self, a: float or int):
        return Quaternion(a * self.num)
    
    def __rmul__(self, a: float or int):
        return Quaternion(a * self.num)
    
    def __truediv__(self, a: float or int):
        return Quaternion(self.num / a)
    
    def __rtruediv__(self, a: float or int):
        return Quaternion(a / self.num)
    
    def __neg__(self):
        self.num = -1 * self.num
    
    def __eq__(self, q) -> bool:
        return (self.num == q.num).all()
    
    def __len__(self):
        return 4
    
    def __getitem__(self, idx: int):
        return self.num[idx]
    
    def __setitem__(self, idx: int, value: float or int):
        self.num[idx] = float(value)

    def __repr__(self) -> str:
        return f'Quaternion({self.num.tolist()})'  
        
    def __str__(self) -> str:
        s = f'{self.num[0]:.3f}'
        tmp = 'ijk'
        for i in range(3):
            if self.num[i + 1] < 0:
                s += f'{self.num[i+1]:.3f}' + tmp[i]
            else: 
                s += '+' + f'{self.num[i+1]:.3f}' + tmp[i]
        return s
    
def QuatMul(q: Quaternion, p: Quaternion) -> Quaternion:
    '''
    四元数乘法
    '''
    c = np.zeros(4)
    c[0] = q[0] * p[0] - np.dot(q[1:], p[1:])
    c[1:] = q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])
    return Quaternion(c)

def Quat2RotMat(q: Quaternion) -> np.ndarray:
    return np.array([
        [2*(q[0]**2 + q[1]**2) - 1,     2*(q[1]*q[2] - q[0]*q[3]),      2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]),     2*(q[0]**2 + q[2]**2) - 1,      2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]),     2*(q[2]*q[3] + q[0]*q[1]),      2*(q[0]**2 + q[3]**2) - 1],
    ])

def RotMat2EulerXYZ(R: np.ndarray) -> list:
    assert R.shape == (3, 3)
    Euler_beta = atan2(R[0, 2], sqrt(R[0, 0]**2 + R[0, 1]**2))
    Euler_alpha = atan2(-R[1, 2]/cos(Euler_beta), R[2, 2]/cos(Euler_beta))
    Euler_gamma = atan2(-R[0, 1]/cos(Euler_beta), R[0, 0]/cos(Euler_beta))
    return [Euler_alpha, Euler_beta, Euler_gamma]

def EulerXYZ2RotMat(Euler_XYZ: list or tuple or np.ndarray) -> np.ndarray:
    assert len(Euler_XYZ) == 3
    alpha, beta, gamma = Euler_XYZ
    
    c, s = cos, sin

    R00 = c(beta) * c(gamma)
    R01 = -c(beta) * s(gamma)
    R02 = s(beta)
    
    R10 = s(alpha) * s(beta) * c(gamma) + c(alpha) * s(gamma)
    R11 = -s(alpha) * s(beta) * s(gamma) + c(alpha) * c(gamma)
    R12 = -s(alpha) * c(beta)
    
    R20 = -c(alpha) * s(beta) * c(gamma) + s(alpha) * s(gamma)
    R21 = c(alpha) * s(beta) * s(gamma) + s(alpha) * c(gamma)
    R22 = c(alpha) * c(beta)
    
    return np.array([
        [R00,    R01,    R02],
        [R10,    R11,    R12],
        [R20,    R21,    R22],
    ])

def RotMat2Quat(R: np.array) -> Quaternion:
    assert R.shape == (3, 3)
    q = Quaternion([0.0] * 4)
    q[0] = 0.5 * sqrt(np.einsum('ii -> i', R).sum() + 1)
    q[1] = (R[2, 1] - R[1, 2]) / 4 / q[0]
    q[2] = (R[0, 2] - R[2, 0]) / 4 / q[0]
    q[3] = (R[1, 0] - R[0, 1]) / 4 / q[0]
    return q

def Quat2EulerXYZ(q: Quaternion) -> list:
    return RotMat2EulerXYZ(Quat2RotMat(q))

def EulerXYZ2Quat(Euler_XYZ: list or tuple or np.ndarray) -> Quaternion:
    return RotMat2Quat(EulerXYZ2Quat(Euler_XYZ))

def Slerp(q0, q1, t) -> Quaternion:
    assert 0 < t < 1
    theta = acos(np.dot(q0, q1))
    alpha = sin((1-t) * theta) / sin(theta)
    beta = sin(t * theta) / sin(theta)
    qt = alpha * q0 + beta * q1
    return qt

def find_nearest_angles(q, q_list):
    assert(len(q) == q_list.shape[1])
    delta = abs(q - q_list)
    delta = np.max(delta, axis=1)
    index = delta.argmin()
    return q_list[index]

if __name__ == '__main__':
    a = Quaternion([1.0, 2.0, 3.0, 4.0]) 
    b = Quaternion([2.0, 3.0, 4.0, 5.0])
    a /= a.norm
    b /= b.norm
    print(RotMat2Quat(np.eye(3)))