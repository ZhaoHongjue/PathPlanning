#python
from IK.IKSolver import IKSolver
import numpy as np

####################################
### You Can Write Your Code Here ###
####################################



def sysCall_init_back():
    # initialization the simulation
    doSomeInit()    # must have    
    
    #------------------------------------------------------------------------
    # using the codes, you can obtain the poses and positions of four blocks
    pointHandles = []
    for i in range(2):
        pointHandles.append(sim.getObject('::/Platform1/Cuboid' + str(i+1) + '/SuckPoint'))
    for i in range(2):
        pointHandles.append(sim.getObject('::/Platform1/Prism' + str(i+1) + '/SuckPoint'))
    # get the pose of Cuboid/SuckPoint
    for i in range(4):
        print(sim.getObjectPose(pointHandles[i], -1))
    #-------------------------------------------------------------------------
        
        
    #-------------------------------------------------------------------------
    # following codes show how to call the build-in inverse kinematics solver
    # you may call the codes, or write your own IK solver
    # before you use the codes, you need to convert the above quaternions to X-Y'-Z' Euler angels 
    # you may write your own codes to do the conversion, or you can use other tools (e.g. matlab)
    iks = IKSolver()
    # return the joint angle vector q which belongs to [-PI, PI]
    # Position and orientation of the end-effector are defined by [x, y, z, rx, ry, rz]
    # x,y,z are in meters; rx,ry,rz are X-Y'-Z'Euler angles in radian
    angles = iks.solve(np.array([0.4, 0.12, 0.15, -np.pi, 0, -np.pi/2]))
    print(angles)
    #---------------------------------------------------------------------------
    
    """ this demo program shows a 3-postion picking task
    step1: the robot stats to run from the rest position (q0)
    step2: the robot moves to the picking position (q1) in 5s
    step3: turn on the vacumm gripper and picking in 0.5s
    step4: lift a block to position (q2) in 3s
    step5: the robot moves from q2 back to the rest positon q0
    q0 - initial joint angles of the robot
    q1 - joint angles when the robot contacts with a block
    q2 - final joint angels of the robot
    """
    global q0, q1, q2
    q0 = np.zeros(6) # initialize q0 with all zeros
    # angles of joint 1-6 obtained by solving the inverse kinematics
    q1 = np.array([1.35420811e+01,  7.29236025e+01,  3.34154795e+01, -1.63390821e+01,  6.82163217e-15,  1.35420811e+01]) / 180 * np.pi
    q2 = np.array([1.35420811e+01,  6.73350208e+01,  2.79938488e+01, -5.32886951e+00,  6.82163217e-15,  1.35420811e+01]) / 180 * np.pi
    #--------------------------------------------------------------------------
    
def sysCall_actuation_back():
    # put your actuation code in this function   
    
    # get absolute time, t
    t = sim.getSimulationTime()
    
    # if t>20s, pause the simulation
    if t > 20:
        sim.pauseSimulation()    
    # robot takes 5s to move from q0 to q1. 
    # the vaccum gripper takes effect after wating 0.2s. 
    if t < 5.2:
        # call the trajactory planning funcion
        # return the joint angles at time t
        q = trajPlaningDemo(q0, q1, t, 5)
        state = False # vacumm gripper is off
        
    # vacumm gripper takes effect from t=5.2s to 5.5s    
    elif t < 5.5:
        q = q1      # keeps the robot still at q1
        state = True  # vacumm gripper is on
    
    # lift a block and move to q2    
    elif t < 8.7:
        q = trajPlaningDemo(q1, q2, t-5.5, 3)
        state = True
    
    # release the vaccum gripper
    elif t < 9:
        q = q2 
        state = False
    else:
        # robot moves from q2 to q0 within 5s
        q = trajPlaningDemo(q2, q0, t-9, 5)
        state = False
    
    # check if the joint velocities beyond limitations.
    # if they do, the simulation will stops and report errors.
    runState = move(q, state)

    if not runState:
        sim.pauseSimulation()
        
    """
    The following codes shows a procedure of trajectory planning using the 5th-order polynomial
    You may write your own code to replace this function, e.g. trapezoidal velocity planning
    """
def trajPlaningDemo(start, end, t, time):
    """ Quintic Polynomial: x = k5*t^5 + k4*t^4 + k3*t^3 + k2*t^2 + k1*t + k0
    :param start: Start point
    :param end: End point
    :param t: Current time
    :param time: Expected time spent
    :return: The value of the current time in this trajectory planning
    """
    if t < time:
        tMatrix = np.matrix([
        [         0,           0,             0,          0,        0,   1],
        [   time**5,     time**4,       time**3,    time**2,     time,   1],
        [         0,           0,             0,          0,        1,   0],
        [ 5*time**4,   4*time**3,     3*time**2,     2*time,        1,   0],
        [         0,           0,             0,          2,        0,   0],
        [20*time**3,  12*time**2,        6*time,          2,        0,   0]])
        
        xArray = []
        for i in range(len(start)):
            xArray.append([start[i], end[i], 0, 0, 0, 0])
        xMatrix = np.matrix(xArray).T
        
        kMatrix = tMatrix.I * xMatrix
        
        timeVector = np.matrix([t**5, t**4, t**3, t**2, t, 1]).T
        x = (kMatrix.T * timeVector).T.A[0]
        
    else:
        x = end
    
    return x


####################################################
### You Don't Have to Change the following Codes ###
####################################################

def doSomeInit():
    global Joint_limits, Vel_limits, Acc_limits
    Joint_limits = np.array([[-200, -90, -120, -150, -150, -180],
                            [200, 90, 120, 150, 150, 180]]).transpose()/180*np.pi
    Vel_limits = np.array([100, 100, 100, 100, 100, 100])/180*np.pi
    Acc_limits = np.array([500, 500, 500, 500, 500, 500])/180*np.pi
    
    global lastPos, lastVel, sensorVel
    lastPos = np.zeros(6)
    lastVel = np.zeros(6)
    sensorVel = np.zeros(6)
    
    global robotHandle, suctionHandle, jointHandles
    robotHandle = sim.getObject('.')
    suctionHandle = sim.getObject('./SuctionCup')
    jointHandles = []
    for i in range(6):
        jointHandles.append(sim.getObject('./Joint' + str(i+1)))
    sim.writeCustomDataBlock(suctionHandle, 'activity', 'off')
    sim.writeCustomDataBlock(robotHandle, 'error', '0')
    
    global dataPos, dataVel, dataAcc, graphPos, graphVel, graphAcc
    dataPos = []
    dataVel = []
    dataAcc = []
    graphPos = sim.getObject('./DataPos')
    graphVel = sim.getObject('./DataVel')
    graphAcc = sim.getObject('./DataAcc')
    color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    for i in range(6):
        dataPos.append(sim.addGraphStream(graphPos, 'Joint'+str(i+1), 'deg', 0, color[i]))
        dataVel.append(sim.addGraphStream(graphVel, 'Joint'+str(i+1), 'deg/s', 0, color[i]))
        dataAcc.append(sim.addGraphStream(graphAcc, 'Joint'+str(i+1), 'deg/s2', 0, color[i]))

def sysCall_sensing():
    # put your sensing code here
    if sim.readCustomDataBlock(robotHandle,'error') == '1':
        return
    global sensorVel
    for i in range(6):
        pos = sim.getJointPosition(jointHandles[i])
        if i == 0:
            if pos < -160/180*np.pi:
                pos += 2*np.pi
        vel = sim.getJointVelocity(jointHandles[i])
        acc = (vel - sensorVel[i])/sim.getSimulationTimeStep()
        if pos < Joint_limits[i, 0] or pos > Joint_limits[i, 1]:
            print("Error: Joint" + str(i+1) + " Position Out of Range!")
            sim.writeCustomDataBlock(robotHandle, 'error', '1')
            return
        
        if abs(vel) > Vel_limits[i]:
            print("Error: Joint" + str(i+1) + " Velocity Out of Range!")
            sim.writeCustomDataBlock(robotHandle, 'error', '1')
            return
        
        if abs(acc) > Acc_limits[i]:
            print("Error: Joint" + str(i+1) + " Acceleration Out of Range!")
            sim.writeCustomDataBlock(robotHandle, 'error', '1')
            return
        
        sim.setGraphStreamValue(graphPos,dataPos[i], pos*180/np.pi)
        sim.setGraphStreamValue(graphVel,dataVel[i], vel*180/np.pi)
        sim.setGraphStreamValue(graphAcc,dataAcc[i], acc*180/np.pi)
        sensorVel[i] = vel

def sysCall_cleanup():
    # do some clean-up here
    sim.writeCustomDataBlock(suctionHandle, 'activity', 'off')
    sim.writeCustomDataBlock(robotHandle, 'error', '0')


def move(q, state):
    if sim.readCustomDataBlock(robotHandle,'error') == '1':
        return
    global lastPos, lastVel
    for i in range(6):
        if q[i] < Joint_limits[i, 0] or q[i] > Joint_limits[i, 1]:
            print("move(): Joint" + str(i+1) + " Position Out of Range!")
            return False
        if abs(q[i] - lastPos[i])/sim.getSimulationTimeStep() > Vel_limits[i]:
            print("move(): Joint" + str(i+1) + " Velocity Out of Range!")
            return False
        if abs(lastVel[i] - (q[i] - lastPos[i]))/sim.getSimulationTimeStep() > Acc_limits[i]:
            print("move(): Joint" + str(i+1) + " Acceleration Out of Range!")
            return False
            
    lastPos = q
    lastVel = q - lastPos
    
    for i in range(6):
        sim.setJointTargetPosition(jointHandles[i], q[i])
        
    if state:
        sim.writeCustomDataBlock(suctionHandle, 'activity', 'on')
    else:
        sim.writeCustomDataBlock(suctionHandle, 'activity', 'off')
    
    return True
    