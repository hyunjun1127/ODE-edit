"""Bounded registered app-server coordination; never changes model/effort."""
import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import time


class Connection:
    def __init__(self):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(5)
        self.socket.connect('/mnt/raid5/janghj/.codex/app-server-control/app-server-control.sock')
        self.buffer = b''
        key = base64.b64encode(os.urandom(16)).decode()
        self.socket.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
            'Connection: Upgrade\r\nSec-WebSocket-Key: ' + key + '\r\n'
            'Sec-WebSocket-Version: 13\r\n\r\n').encode())
        while b'\r\n\r\n' not in self.buffer:
            self.buffer += self.socket.recv(4096)
            if len(self.buffer) > 65536:
                raise ValueError('HEADER_LIMIT')
        header, self.buffer = self.buffer.split(b'\r\n\r\n', 1)
        expected = base64.b64encode(hashlib.sha1((key +
            '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        if not header.startswith(b'HTTP/1.1 101') or expected not in header:
            raise ValueError('WEBSOCKET_HANDSHAKE')

    def exact(self, count):
        while len(self.buffer) < count:
            part = self.socket.recv(count - len(self.buffer))
            if not part:
                raise EOFError('WEBSOCKET_CLOSED')
            self.buffer += part
        value, self.buffer = self.buffer[:count], self.buffer[count:]
        return value

    def send(self, message, opcode=1):
        data = message if isinstance(message, bytes) else json.dumps(message).encode()
        count = len(data)
        mask = os.urandom(4)
        header = bytes([128 | opcode])
        header += (bytes([128 | count]) if count < 126 else
                   bytes([254]) + struct.pack('!H', count) if count < 65536 else
                   bytes([255]) + struct.pack('!Q', count))
        self.socket.sendall(header + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(data)))

    def receive(self):
        parts = []
        while True:
            first, second = self.exact(2)
            opcode, count = first & 15, second & 127
            if count == 126:
                count = struct.unpack('!H', self.exact(2))[0]
            elif count == 127:
                count = struct.unpack('!Q', self.exact(8))[0]
            if count > 128 * 1024 ** 2:
                raise ValueError('FRAME_LIMIT')
            mask = self.exact(4) if second & 128 else None
            data = self.exact(count)
            if mask:
                data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
            if opcode == 9:
                self.send(data, 10)
                continue
            if opcode == 10:
                continue
            if opcode == 8:
                raise EOFError('WEBSOCKET_CLOSED')
            parts.append(data)
            if first & 128:
                return json.loads(b''.join(parts))

    def rpc(self, number, method, params):
        self.send(dict(id=number, method=method, params=params))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            message = self.receive()
            if message.get('id') == number:
                if 'error' in message:
                    raise ValueError('RPC_' + method)
                return message['result']
        raise TimeoutError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--message', required=True)
    args = parser.parse_args()
    result = dict(nonce=args.nonce, target=args.thread, model_effort_override=False)
    connection = Connection()
    try:
        connection.rpc(1, 'initialize', dict(clientInfo=dict(name='official_sh2', version='1'),
                       capabilities=dict(experimentalApi=True)))
        connection.send(dict(method='initialized', params={}))
        thread = connection.rpc(2, 'thread/resume', dict(threadId=args.thread, excludeTurns=True))['thread']
        if thread['id'] != args.thread or thread['cwd'] not in (
                '/mnt/raid5/janghj/ODE-edit', '/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit'):
            raise ValueError('TARGET_BOUNDARY')
        params = dict(threadId=args.thread, input=[dict(type='text', text=args.message)])
        status = thread.get('status', {}).get('type')
        if status == 'idle':
            method = 'turn/start'
        elif status == 'active':
            last = connection.rpc(3, 'thread/turns/list', dict(threadId=args.thread,
                limit=1, sortDirection='desc', itemsView='full'))['data'][0]
            texts = [c.get('text', '') for item in last.get('items', [])
                     if item.get('type') == 'userMessage' for c in item.get('content', [])
                     if c.get('type') == 'text']
            if last.get('status') != 'inProgress' or not any(
                    marker in '\n'.join(texts) for marker in ('official', 'USER-OFFICIAL-BASELINES')):
                raise ValueError('UNRELATED_ACTIVE_NO_STEER')
            method = 'turn/steer'
            params['expectedTurnId'] = last['id']
        else:
            raise ValueError('THREAD_NOT_READY')
        accepted = connection.rpc(4, method, params)
        turn = accepted.get('turn', {}).get('id') or accepted.get('turnId')
        result.update(method=method, accepted_turn=turn, stage='DIRECT_ACCEPTED_NOT_COMPLETED')
        print(json.dumps(result), flush=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                message = connection.receive()
            except socket.timeout:
                continue
            params = message.get('params', {})
            if params.get('turnId') == turn and message.get('method') == 'item/completed':
                item = params.get('item', {})
                if item.get('type') == 'agentMessage':
                    print(json.dumps(dict(nonce=args.nonce, response=item.get('text', '')[:8000])), flush=True)
            if message.get('method') == 'turn/completed' and params.get('turn', {}).get('id') == turn:
                result.update(stage='TERMINAL', terminal_status=params['turn'].get('status'))
                break
    except Exception as error:
        result.update(stage='COMMUNICATION_HOLD', error_type=type(error).__name__, error=str(error)[:120])
    finally:
        connection.socket.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
