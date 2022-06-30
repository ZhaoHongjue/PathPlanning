from math import cos, acos, sin, atan2, sqrt, pi
import numpy as np
from pose import EulerXYZ2RotMat, Quat2RotMat


def DHs(theta):
    return np.array([
        [0,         0,         0.23,      theta[0]        ],
        [0,         -pi/2,     -0.054,    -pi/2 + theta[1]],
        [0.185,     0,         0,         theta[2]        ],
        [0.170,     0,         0.077,     pi/2 + theta[3] ],
        [0,         pi/2,      0.077,     pi/2 + theta[4] ],
        [0,         pi/2,      0,         theta[5]        ],
        [0,         0,         0.0855,    0               ]
    ], dtype = np.float32)

def TransMat(DH: np.ndarray):
    '''求解单步的T'''
    a, alpha, d, theta = DH
    return np.array([
        [cos(theta),                -sin(theta),                0,              a               ],
        [sin(theta)*cos(alpha),     cos(theta)*cos(alpha),      -sin(alpha),    -d*sin(alpha)   ],
        [sin(theta)*sin(alpha),     cos(theta)*sin(alpha),      cos(alpha),     d*cos(alpha)    ],
        [0,                         0,                          0,              1               ]
    ])
    
def IKSolver(pose, check = True):
    c, s = cos, sin
    theta = []
    x, y, z = pose[0:3]
    rot = pose[3:]
    # 求取旋转矩阵
    if len(rot) == 3:
        R = EulerXYZ2RotMat(rot)
    elif len(rot) == 4:
        R = Quat2RotMat(rot)
    else:
        raise
    
    # 求theta1(有两解)
    px = x - 0.0855 * R[0, 2]
    py = y - 0.0855 * R[1, 2]
    rho = sqrt(px**2 + py**2)
    theta11 = atan2(py, px) - atan2(0.023, sqrt(rho**2 - 0.023**2))
    theta12 = atan2(py, px) - atan2(0.023, -sqrt(rho**2 - 0.023**2))
    
    for theta1 in [theta11, theta12]:
        if (theta1 < -200/180*pi or theta1 > 200/180*pi) and check:
            continue
        
        LS21 = -R[0, 0] * s(theta1) + R[1, 0] * c(theta1)
        LS22 = -R[0, 1] * s(theta1) + R[1, 1] * c(theta1)
        LS23 = -R[0, 2] * s(theta1) + R[1, 2] * c(theta1)
        LS33 = R[2, 2]
        LS13 = R[0, 2] * c(theta1) + R[1, 2] * s(theta1)
        
        theta51 = atan2(LS23, sqrt(LS21**2 + LS22**2))
        theta52 = atan2(LS23, -sqrt(LS21**2 + LS22**2))
        
        for theta5 in [theta51, theta52]:
            if (theta5 < -5*pi/6 or theta5 > 5*pi/6) and check:
                continue
            
            theta6 = atan2(-LS22/c(theta5), LS21/c(theta5))
            
            if (theta6 < -pi or theta6 > pi) and check:
                continue
            
            theta234 = atan2(-LS33/c(theta5), LS13/c(theta5))

            LS14 = px * c(theta1) + py * s(theta1)
            LS34 = z - 0.0855*R[2, 2] - 0.23
            a, b = LS14 - 0.077 * s(theta234), LS34 - 0.077 * c(theta234)
            
            c3 = (a**2 + b**2 - 0.185**2 - 0.17**2) / (2 * 0.185 * 0.17)
            if c3 > 1 or c3 < -1:
                continue
            theta31 = acos(c3)
            theta32 = -theta31
            
            for theta3 in [theta31, theta32]:
                if (theta3 < -2*pi/3 or theta3 > 2*pi/3) and check:
                    continue
                p = 0.185 + 0.17 * c(theta3)
                q = 0.17 * s(theta3)
                
                theta2 = atan2((a*p - b*q)/(p**2 + q**2), (a*q + b*p)/(p**2 + q**2))
                if (theta2 < -pi/2 or theta2 > pi/2) and check:
                    continue
                
                theta4 = theta234 - theta2 - theta3
                if (theta4 < -5*pi/6 or theta4 > 5*pi/6) and check:
                    continue
                theta.append([theta1, theta2, theta3, theta4, theta5, theta6])
    
    return np.asarray(theta)

if __name__ == '__main__':
    pose = [0.4, -0.04, 0.125, pi, 0, 0]
    print(IKSolver(pose))