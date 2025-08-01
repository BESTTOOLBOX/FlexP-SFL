import time
import socket
import threading
import json
import sys

def recvmsg(tcp_tunnel_s):
    while 1:
        try:
            print("listening...")
            sock, addr = tcp_tunnel_s.accept()
            print("TCP Connected From" + str(addr))
            tcp_start_time_recv=time.time()
            total_data = b''
            data = sock.recv(1024)
            total_data += data
            num = len(data)
            while len(data) > 0:
                data = sock.recv(1024)
                num += len(data)
                total_data += data
                print(str(sys.getsizeof(total_data))+" Bytes Received "+str(round(sys.getsizeof(total_data)/(time.time()-tcp_start_time_recv)/1000,2))+" KB/s")
            clientdata=total_data
            print("Done")
        except:
            pass

local_HOST="127.0.0.1"
local_PORT=50000

tcp_tunnel_s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcp_tunnel_s.bind((local_HOST, local_PORT))
tcp_tunnel_s.listen(5)

thread_recvmsg=threading.Thread(target=recvmsg,args=(tcp_tunnel_s,))
thread_recvmsg.start()
