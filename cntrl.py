# URSoftware 5.1.1

# Echo client program
import socket
import time
import math
import base64


def init_socket():
    #HOST = "169.254.52.193"    # The remote host
# HOST = "192.168.0.1"
    HOST = "192.168.0.10"
    PORT = 30003            # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((HOST, PORT))
    print('socket',s)
    return s



def debug():
    s.send ("set_digital_out(2,True)".encode() + "\n".encode())
    time.sleep(0.05)

# pose = [x,y,z,rx,ry,rz]
# acceleration and velocity have default values
# if you set a time 't', that overrides 'a' and 'v'
def movel(s, p, a=0.1,v=0.5,t=0,r=0):
    try:
        script = "movel(p[{},{},{},{},{},{}], a={}, v={}, t={}, r={})"
        script = script.format(p[0],p[1],p[2],p[3],p[4],p[5],a,v,t,r)
        s.send(bytes(script, 'utf-8')+bytes("\n", 'utf-8'))
        msg =  s.recv(1024)
    except socket.error as socketerror:
        print("..... Some kind of error :(...")
    return msg

# angles = [base,shoulder,elbow,wrist1,wrist2,wrist3]
# angles can be given in degrees
def movej(s,angles,a=0.1,v=0.5,t=0,r=0):
    # base = angles[0]*math.pi/180 # converting to radians
    # shoulder = angles[1]*math.pi/180
    # elbow = angles[2]*math.pi/180
    # w1 = angles[3]*math.pi/180
    # w2 = angles[4]*math.pi/180
    # w3 = angles[5]*math.pi/180
    print('movingj')
    base = angles[0]
    shoulder = angles[1]
    elbow = angles[2]
    w1 = angles[3]
    w2 = angles[4]
    w3 = angles[5]

    try:
        script = "movej([{},{},{},{},{},{}], a={}, v={}, t={}, r={})"
        script = script.format(base,shoulder,elbow,w1,w2,w3,a,v,t,r)
        s.send(bytes(script, 'utf-8')+bytes("\n", 'utf-8'))
        msg = s.recv(1024)
    except socket.error as socketerror:
        print("..... Some kind of error :(...")
    return msg

# moving to stable joint angles
#movej(s,[80.43,-40.59,61.4,-110.8,-89.77,80.45])


def servoj(s,angles,a=0.1,v=0.5,t=0, gain = 100, lookahead_time = 0.2):
    # base = angles[0]*math.pi/180 # converting to radians
    # shoulder = angles[1]*math.pi/180
    # elbow = angles[2]*math.pi/180
    # w1 = angles[3]*math.pi/180
    # w2 = angles[4]*math.pi/180
    # w3 = angles[5]*math.pi/180
    
    base = angles[0]
    shoulder = angles[1]
    elbow = angles[2]
    w1 = angles[3]
    w2 = angles[4]
    w3 = angles[5]

    try:
        
        script = "servoj([{},{},{},{},{},{}], a={}, v={}, t={}, lookahead_time={}, gain={})"
        script = script.format(base,shoulder,elbow,w1,w2,w3,a,v,t, lookahead_time, gain)
        s.send(bytes(script, 'utf-8')+bytes("\n", 'utf-8'))
        msg = s.recv(1024)
    except socket.error as socketerror:
        print("..... Some kind of error :(...")
    return msg



def initialise(s, a=0.5,v=0.5,t=0,r=0):


    x = 0       # mm
    y = -0.6
    z = 0.2
    rx = 2.221  # radians
    ry = 2.221
    rz = 0
    p = [x,y,z,rx,ry,rz]
    try:
        script = "movel(p[{},{},{},{},{},{}], a={}, v={}, t={}, r={})"
        script = script.format(p[0],p[1],p[2],p[3],p[4],p[5],a,v,t,r)
        s.send(bytes(script, 'utf-8')+bytes("\n", 'utf-8'))
        msg = s.recv(1024)

    except socket.error as socketerror:
        print("..... Some kind of error :(...")

    return msg


# This function doesn't work. The output doesn't make sense
def decode_position(s, msg):
    # Decode Pose or Joints from UR
    time.sleep(2)
    current_position = [0,0,0,0,0,0]
    data_start = 0
    data_end = 0
    n = 0
    x = 0
    while x < len(msg):
        if msg[x]=="," or msg[x]=="]" or msg[x]=="e":
            data_end = x
            current_position[n] = float(msg[data_start:data_end])
            if msg[x]=="e":
                current_position[n] = current_position[n]*math.pow(10,float(msg[x+1:x+4]))
                #print "e", msg[x+1:x+4]
                #print "e", int(msg[x+1:x+4])
                if n < 5:
                    x = x+5
                    data_start = x
                else:
                    break
            n=n+1
        if msg[x]=="[" or msg[x]==",":
            data_start = x+1
        x = x+1

    print(current_position)

# msg = movel(s,[0,-1,0.2,2.221,2.221,0])
# #decode_position(s, msg)
# print(msg.decode('ISO-8859-1'))
# s = init_socket()
# initialise(s, a=0.5,v=0.5,t=0,r=0)
# s.close()
# # print(base64.b64decode(msg))

# s.close()
