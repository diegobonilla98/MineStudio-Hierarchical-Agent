import json
import socket
import struct
import time
from io import BytesIO
from pathlib import Path

from PIL import Image


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18765
CONNECT_TIMEOUT_SECONDS = 300
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "output" / "dgx_coach_transport_smoke.json"


def receive_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = connection.recv(byte_count - len(chunks))
        if not chunk:
            raise ConnectionError("MineStudio server disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def connect() -> socket.socket:
    deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(1.0)
        if connection.connect_ex((SERVER_HOST, SERVER_PORT)) == 0:
            connection.settimeout(CONNECT_TIMEOUT_SECONDS)
            return connection
        connection.close()
        time.sleep(0.5)
    raise ConnectionError("Timed out waiting for the tunneled coach server")


def main() -> None:
    started_at = time.monotonic()
    connection = connect()
    try:
        header_size = struct.unpack("!I", receive_exact(connection, 4))[0]
        header = json.loads(receive_exact(connection, header_size))
        image_size = struct.unpack("!I", receive_exact(connection, 4))[0]
        image_data = receive_exact(connection, image_size)
        image = Image.open(BytesIO(image_data))
        image.verify()
        shutdown = json.dumps({"shutdown": True}, separators=(",", ":")).encode("utf-8")
        connection.sendall(struct.pack("!I", len(shutdown)) + shutdown)
        result = {
            "elapsed_seconds": time.monotonic() - started_at,
            "image_bytes": image_size,
            "image_format": image.format,
            "image_size": list(image.size),
            "status": header.get("status"),
            "global_goal": header.get("global_goal"),
            "instruction": header.get("instruction"),
            "gemini_feedback_present": bool(header.get("feedback")),
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
