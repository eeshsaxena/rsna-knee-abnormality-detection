# verify_relabel: block every outbound socket connection and DNS lookup; log attempts.
import socket, sys, os
_LOG = os.environ.get('NONET_LOG')
def _note(what):
    if _LOG:
        with open(_LOG, 'a') as f:
            f.write(what + '\n')
_orig_connect = socket.socket.connect
def _connect(self, addr):
    _note(f'CONNECT {addr!r}')
    raise OSError('noNet: outbound connect blocked')
socket.socket.connect = _connect
socket.socket.connect_ex = lambda self, addr: (_note(f'CONNECT_EX {addr!r}'), 111)[1]
def _gai(*a, **k):
    _note(f'GETADDRINFO {a[:2]!r}')
    raise socket.gaierror('noNet: DNS blocked')
socket.getaddrinfo = _gai
socket.create_connection = lambda *a, **k: (_note(f'CREATE_CONNECTION {a[:1]!r}'), (_ for _ in ()).throw(OSError('noNet')))[1]
