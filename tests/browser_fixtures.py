"""Small generated media fixtures for local browser tests; no cloud media required."""
from http.server import SimpleHTTPRequestHandler
import io
import math
import struct
import wave
import zlib


def audio_fixture():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b''.join(struct.pack('<h', int(6000 * math.sin(2 * math.pi * 500 * i / 16000)))
                                   for i in range(80000)))
    return output.getvalue()


def image_fixture():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = b''.join(b'\x00' + bytes([row * 4, 80, 120]) * 128 for row in range(64))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 64, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


AUDIO = audio_fixture()
IMAGE = image_fixture()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.split('?', 1)[0].rsplit('/', 1)[-1]
        if name in ('download0.wav', 'output0.png', 'output1.png'):
            audio = name.endswith('.wav')
            data = AUDIO if audio else IMAGE
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav' if audio else 'image/png')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            return
        super().do_GET()
