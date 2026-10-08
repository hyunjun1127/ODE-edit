"""Bounded official SSH -> target Unix WebSocket coordination, no inbox fallback."""
import argparse
import json
import subprocess
from .common import LOCAL,write

REMOTE = r'''
import hashlib,json,os,socket,struct,time
thread='01a0493a-074c-7f91-9a13-769116326fef'
message=__MESSAGE__
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(40)
sock.connect('/mnt/raid5/janghj/.codex/app-server-control/app-server-control.sock')
key=__import__('base64').b64encode(os.urandom(16)).decode()
sock.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: '+key+'\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
buffer=b''
while b'\r\n\r\n' not in buffer:buffer+=sock.recv(65536)
header,buffer=buffer.split(b'\r\n\r\n',1)
assert b'101' in header.split(b'\r\n')[0], 'UPGRADE_FAILED'
def take(n):
 global buffer
 while len(buffer)<n:buffer+=sock.recv(max(65536,n-len(buffer)))
 value,buffer=buffer[:n],buffer[n:];return value
def send(value,opcode=1):
 payload=json.dumps(value).encode() if opcode==1 else value
 n=len(payload);mask=os.urandom(4)
 prefix=bytes([128|opcode,128|n]) if n<126 else bytes([128|opcode,128|126])+struct.pack('!H',n) if n<65536 else bytes([128|opcode,128|127])+struct.pack('!Q',n)
 sock.sendall(prefix+mask+bytes(c^mask[i%4] for i,c in enumerate(payload)))
def receive():
 while True:
  first,second=take(2);n=second&127
  if n==126:n=struct.unpack('!H',take(2))[0]
  elif n==127:n=struct.unpack('!Q',take(8))[0]
  mask=take(4) if second&128 else None;data=take(n)
  if mask:data=bytes(c^mask[i%4] for i,c in enumerate(data))
  opcode=first&15
  if opcode==9:send(data,10);continue
  if opcode==8:raise RuntimeError('PEER_CLOSED')
  if opcode==1:return json.loads(data)
def rpc(number,method,params):
 send(dict(id=number,method=method,params=params))
 while True:
  value=receive()
  if value.get('id')==number:
   if 'error' in value:raise RuntimeError(method+':'+json.dumps(value['error']))
   return value.get('result')
rpc(1,'initialize',dict(clientInfo=dict(name='odeedit-sh1-peer',version='1.0'),capabilities=dict(experimentalApi=True)))
send(dict(method='initialized'))
resumed=rpc(2,'thread/resume',dict(threadId=thread))
t=resumed.get('thread',resumed)
if message.startswith('READ_TURN:'):
 wanted=message.split(':',1)[1]
 turns=[v for v in t.get('turns',[]) if v.get('id')==wanted]
 value=dict(transport='REGISTERED_SSH_UNIX_WEBSOCKET',method='thread/resume',thread=thread,turn=wanted,matched=bool(turns))
 if turns:
  value['status']=turns[0].get('status')
  value['owner_text']=[v.get('text','')[:6000] for v in turns[0].get('items',[]) if v.get('type')=='agentMessage']
 print(json.dumps(value,ensure_ascii=False));sock.close();raise SystemExit()
active=[v for v in t.get('turns',[]) if v.get('status')=='inProgress']
if active:
 result=rpc(3,'turn/steer',dict(threadId=thread,expectedTurnId=active[-1]['id'],input=[dict(type='text',text=message)]))
 method='turn/steer';turn=active[-1]['id']
else:
 result=rpc(3,'turn/start',dict(threadId=thread,input=[dict(type='text',text=message)]));method='turn/start'
 turn=result.get('turn',{}).get('id',result.get('turnId'))
receipt=dict(transport='REGISTERED_SSH_UNIX_WEBSOCKET',method=method,thread=thread,accepted=True,turn=turn,response_received=True,owner_ACK_observed=False)
deadline=time.monotonic()+35;parts=[]
while time.monotonic()<deadline:
 sock.settimeout(max(.1,deadline-time.monotonic()))
 try:value=receive()
 except socket.timeout:break
 params=value.get('params',{});item=params.get('item',{})
 text=item.get('text',params.get('delta',''))
 if isinstance(text,str):parts.append(text)
 joined=''.join(parts)
 if 'ACK' in joined and 'nonce=' in joined and '20261008-R1' in joined:
  receipt['owner_ACK_observed']=True;receipt['owner_text']=joined[:3000];break
 if value.get('method')=='turn/completed':receipt['completed']=True;break
print(json.dumps(receipt,ensure_ascii=False));sock.close()
'''

def relay(message,receipt_name):
    path=LOCAL/'native-repo-repair-r1'/receipt_name
    if path.exists():raise RuntimeError('CREATE_ONCE_RELAY_RECEIPT')
    script=REMOTE.replace('__MESSAGE__',repr(message))
    result=subprocess.run(['ssh','-F','/mnt/raid5/janghj/ODE-edit/servers/local/ssh_config',
        '-o','BatchMode=yes','-o','ConnectTimeout=8','rke-server2',
        '/mnt/raid5/janghj/EasyEdit/.venv/bin/python -'],input=script,text=True,capture_output=True,timeout=55)
    value=dict(returncode=result.returncode,automatic_retry=False,message_sha256=__import__('hashlib').sha256(message.encode()).hexdigest())
    if result.returncode==0:value.update(json.loads(result.stdout.strip().splitlines()[-1]))
    else:value.update(status='COMMUNICATION_HOLD',error=result.stderr[-2000:])
    write(path,value);print(json.dumps(value,ensure_ascii=False))
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--message',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();relay(a.message,a.receipt)
