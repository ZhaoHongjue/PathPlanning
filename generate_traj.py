import pose
from Kinematics import *
from CurvePlan import *
from matplotlib import pyplot as plt

import csv

# 1. Prism : suckPoints[1] -> [0.1,0.35,0.15, pi, 0, 0] -> [-0.1, 0.35, 0.15, pi, 0, 0]
# -> [-0.35, -0.05, 0.175, pi / 4, 0, 0]
# 2. Cuboid : suckPoints[0] -> [0.1, 0.35, 0.15, pi, 0, 0] -> [-0.1, 0.35, 0.15, pi, 0, 0]
# -> [-0.35, 0, 0.15, 0.2, pi, 0, 0]
traj_path = "/home/luke/Documents/Robot_Summer/traj.csv"

"""_summary
1. Prism : suckPoints[1] -> [0.1,0.35,0.15, pi, 0, 0] -> [-0.1, 0.35, 0.15, pi, 0, 0]
-> [-0.35, -0.05, 0.175, pi / 4, 0, 0]
2. Cuboid : suckPoints[0] -> [0.1, 0.35, 0.15, pi, 0, 0] -> [-0.1, 0.35, 0.15, pi, 0, 0]
-> [-0.35, 0, 0.15, 0.2, pi, 0, 0]
3. 
"""

def generate_traj(suck_points):
    traj_file = open(traj_path, "w")
    writer = csv.writer(traj_file)

    traj = []
    
    # From initial point to first point
    q_init = np.zeros(6)
    q_suck0 = IKSolver(suck_points[0])
    q_suck0 = q_suck0[0] # Randomly choose one. 
    
    delta_t = 0.05
    t_init = 2
    
    joint_acc = 500 / 180 * np.pi
    joint_acc = np.ones_like(q_init) * joint_acc
    
    for i in range(int(t_init / delta_t)):
        one_ang = TrapezoidPlan(q_init, q_suck0, joint_acc, t_f=2, t=i * delta_t)
        traj.append(one_ang)
    
    # ---------------------------------------
    traj.append('True')
    
    last_q = q_suck0
    for point in suck_points:
        """_summary_
        prism->cuboid->cuboid->prism
        """
        points = []
        p0 = []
        p0.append(point)
        # p0.append(np.array([0.4, 0.12, 0.2, pi, 0, 0]))
        p0.append(np.array([0.1,0.35,0.15, pi, 0, 0]))
        p0.append(np.array([-0.1, 0.35, 0.15, pi, 0, 0]))
        p0.append(np.array([-0.35, -0.05, 0.175, pi / 4, 0, 0]))
        points.append(p0)
        
        q0_0 = IKSolver(p0[0])
        q0_0 = pose.find_nearest_angles(last_q, q0_0)
        q0_1 = IKSolver(p0[1])
        q0_1 = pose.find_nearest_angles(q0_0, q0_1)
        q0_2 = IKSolver(p0[2])
        q0_2 = pose.find_nearest_angles(q0_1, q0_2)
        q0_3 = IKSolver(p0[3])
        q0_3 = pose.find_nearest_angles(q0_2, q0_3)
        q0 = [q0_0, q0_1, q0_2, q0_3]
        last_q = q0_3
        
        # for q in q0:
        #     print(q)
        
        # From platform to pond
        
        delta_t = 0.05
        t_final1 = 2
        
        joint_acc = 500 / 180 * np.pi
        joint_acc = np.ones_like(q0_0) * joint_acc
        
        for i in range(int(t_final1 / delta_t)):
            one_ang = TrapezoidPlan(q0[0], q0[1], joint_acc, t_final1, i * delta_t)
            traj.append(one_ang)
        
        # In the pond
        t_final2 = 2
        
        for i in range(int(t_final2 / delta_t)):
            pos = TrapezoidPlan(p0[1], p0[2], np.ones(6), t_final2, i * delta_t)
            q = IKSolver(pos)
            traj.append(q)
        
        
        
        # traj = np.array(traj)
    
    
    writer.writerows(traj)
    traj_file.close()
    
    # figure, axs = plt.subplots(joint_num)
    # for i in range(len(axs)):
    #     axs[i].plot(traj[:, i])
    # plt.show()

if __name__ == '__main__':
    generate_traj([np.array([0.3999999463558197, -0.1199999675154686, 0.1499999761581421, 0.8703556656837463, 0.49242353439331055, -2.152451727965854e-08, -3.804445825039693e-08])])