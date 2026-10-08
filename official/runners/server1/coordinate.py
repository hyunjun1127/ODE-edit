"""Bounded existing-thread coordination; no experiment, secret or model transfer."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from official.experiments.prepare import write_new

TARGETS = {
    'GH': '01a04939-8873-7673-8dca-4c7fc5e31af0',
    'SH2': '01a0493a-074c-7f91-9a13-769116326fef',
    'SH3': '01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',
    'SH4': '01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
}

SCRIPT = r'''
import json,os,socket,struct,time
thread=THREAD_VALUE
message=MESSAGE_VALUE
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(35)
sock.connect(SOCKET_VALUE)
key=__import__('base64').b64encode(os.urandom(16)).decode()
sock.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: '+key+'\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
buffer=b''
while b'\r\n\r\n' not in buffer:buffer+=sock.recv(65536)
header,buffer=buffer.split(b'\r\n\r\n',1)
assert b'101' in header.split(b'\r\n')[0], 'UPGRADE_FAILED'
def take(n):
 global buffer
 while len(buffer)<n:
  chunk=sock.recv(max(65536,n-len(buffer)))
  if not chunk:raise RuntimeError('PEER_CLOSED')
  buffer+=chunk
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
rpc(1,'initialize',dict(clientInfo=dict(name='official-baselines-sh1',version='1.0'),capabilities=dict(experimentalApi=True)))
send(dict(method='initialized'))
resumed=rpc(2,'thread/resume',dict(threadId=thread))
t=resumed.get('thread',resumed)
active=[v for v in t.get('turns',[]) if v.get('status')=='inProgress']
if active:
 result=rpc(3,'turn/steer',dict(threadId=thread,expectedTurnId=active[-1]['id'],input=[dict(type='text',text=message)]))
 method='turn/steer';turn=active[-1]['id']
else:
 result=rpc(3,'turn/start',dict(threadId=thread,input=[dict(type='text',text=message)]));method='turn/start'
 turn=result.get('turn',{}).get('id',result.get('turnId'))
print(json.dumps(dict(transport='REGISTERED_UNIX_WEBSOCKET',method=method,thread=thread,accepted=True,turn=turn,owner_ACK_observed=False)))
sock.close()
'''


def relay(target, message, receipt):
    root = '/data/janghj' if target in ('SH3', 'SH4') else '/mnt/raid5/janghj'
    script = (SCRIPT.replace('THREAD_VALUE', repr(TARGETS[target])).replace('MESSAGE_VALUE', repr(message))
              .replace('SOCKET_VALUE', repr(root + '/.codex/app-server-control/app-server-control.sock')))
    argv = ['/mnt/raid5/janghj/EasyEdit/.venv/bin/python', '-']
    if target in ('SH2', 'SH3', 'SH4'):
        argv = ['ssh', '-F', '/mnt/raid5/janghj/ODE-edit/servers/local/ssh_config',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'rke-server' + target[-1],
                'python3 -']
    value = dict(target=target, thread=TARGETS[target], returncode=None,
                 message_sha256=hashlib.sha256(message.encode()).hexdigest(), automatic_retry=False)
    try:
        result = subprocess.run(argv, input=script, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        value.update(returncode=None, status='COMMUNICATION_HOLD', accepted='UNKNOWN',
                     error='BOUNDED_TRANSPORT_TIMEOUT_NO_ACK', owner_ACK_observed=False)
        write_new(receipt, value)
        return value
    value['returncode'] = result.returncode
    if result.returncode == 0:
        value.update(json.loads(result.stdout.strip().splitlines()[-1]))
    else:
        value.update(status='COMMUNICATION_HOLD', error=result.stderr[-1500:])
    write_new(receipt, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--message-file', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(relay(args.target, args.message_file.read_text(), args.receipt)))
