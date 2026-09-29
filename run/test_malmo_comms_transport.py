import struct
import unittest

from minestudio.simulator.minerl.env import comms


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def recv(self, count):
        return self.chunks.pop(0)


class TestMalmoCommsTransport(unittest.TestCase):
    def test_closed_header_raises_connection_error(self):
        with self.assertRaisesRegex(ConnectionError, "message length"):
            comms.recv_message(FakeSocket([b""]))

    def test_closed_payload_raises_connection_error(self):
        header = struct.pack("!I", 4)
        with self.assertRaisesRegex(ConnectionError, "message payload"):
            comms.recv_message(FakeSocket([header, b""]))

    def test_complete_message_is_returned(self):
        payload = b"test"
        header = struct.pack("!I", len(payload))
        self.assertEqual(comms.recv_message(FakeSocket([header, payload])), payload)


if __name__ == "__main__":
    unittest.main()
