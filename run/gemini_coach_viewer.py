import ctypes
import json
import queue
import socket
import struct
import threading
import time
import tkinter as tk
from io import BytesIO

from PIL import Image, ImageTk


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18765
CONNECT_TIMEOUT_SECONDS = 300
CAMERA_SENSITIVITY = 0.1
CAMERA_MAX_DEGREES = 10.0
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 790


class GeminiCoachViewer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MineStudio Gemini Coach")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+80+40")
        self.root.configure(bg="#101318")
        self.root.attributes("-topmost", True)
        self.root.after(1200, lambda: self.root.attributes("-topmost", False))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="#080a0d", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image_item = self.canvas.create_image(WINDOW_WIDTH // 2, 300, anchor="center")
        self.status_item = self.canvas.create_text(24, 610, anchor="nw", fill="#8bd5ff", font=("Segoe UI", 12, "bold"), text="Starting the MineStudio server...", width=1050)
        self.task_item = self.canvas.create_text(24, 650, anchor="nw", fill="white", font=("Segoe UI", 14, "bold"), text="", width=1050)
        self.hint_item = self.canvas.create_text(24, 690, anchor="nw", fill="#c2cad6", font=("Segoe UI", 11), text="", width=1050)
        self.feedback_item = self.canvas.create_text(24, 725, anchor="nw", fill="#a8e6a3", font=("Segoe UI", 10), text="", width=1050)
        self.controls_item = self.canvas.create_text(WINDOW_WIDTH // 2, 774, anchor="s", fill="#9da7b5", font=("Segoe UI", 9), text="WASD move | Mouse look | Left click attack | Right click use | Space jump | C capture mouse | Esc release mouse")

        self.frames: queue.Queue = queue.Queue(maxsize=2)
        self.status_messages: queue.Queue = queue.Queue()
        self.keys = set()
        self.mouse_buttons = set()
        self.camera = [0.0, 0.0]
        self.lock = threading.Lock()
        self.connection = None
        self.running = True
        self.session_complete = False
        self.mouse_captured = False
        self.ignore_warp = False
        self.photo = None

        self.root.bind("<KeyPress>", self.key_press)
        self.root.bind("<KeyRelease>", self.key_release)
        self.root.bind("<ButtonPress-1>", lambda event: self.mouse_press(1))
        self.root.bind("<ButtonRelease-1>", lambda event: self.mouse_release(1))
        self.root.bind("<ButtonPress-3>", lambda event: self.mouse_press(3))
        self.root.bind("<ButtonRelease-3>", lambda event: self.mouse_release(3))
        self.root.bind("<Motion>", self.mouse_motion)
        self.root.bind("<FocusOut>", self.focus_out)
        self.canvas.bind("<Button-1>", self.capture_from_click)

        self.network_thread = threading.Thread(target=self.network_loop, daemon=True)
        self.network_thread.start()
        self.root.after(15, self.render_latest)

    def receive_exact(self, byte_count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < byte_count:
            chunk = self.connection.recv(byte_count - len(chunks))
            if not chunk:
                raise ConnectionError("MineStudio server disconnected")
            chunks.extend(chunk)
        return bytes(chunks)

    def connect(self) -> None:
        deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
        while self.running and time.monotonic() < deadline:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(1.0)
            result = connection.connect_ex((SERVER_HOST, SERVER_PORT))
            if result == 0:
                connection.settimeout(None)
                self.connection = connection
                return
            connection.close()
            time.sleep(0.5)
        raise ConnectionError("Timed out waiting for the MineStudio server")

    def network_loop(self) -> None:
        received_frame = False
        while self.running and not self.session_complete:
            try:
                self.connect()
                self.status_messages.put(("Connected. Minecraft is loading...", "#8bd5ff"))
                while self.running:
                    header_size = struct.unpack("!I", self.receive_exact(4))[0]
                    header = json.loads(self.receive_exact(header_size))
                    image_size = struct.unpack("!I", self.receive_exact(4))[0]
                    image_data = self.receive_exact(image_size)
                    received_frame = True
                    while self.frames.full():
                        self.frames.get_nowait()
                    self.frames.put((header, image_data))
                    self.session_complete = bool(header.get("session_complete"))
                    action = self.current_action()
                    payload = json.dumps({"action": action}, separators=(",", ":")).encode("utf-8")
                    self.connection.sendall(struct.pack("!I", len(payload)) + payload)
            except (ConnectionError, OSError) as error:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
                if self.running and not self.session_complete and not received_frame:
                    self.status_messages.put(("Waiting for the DGX coach server...", "#8bd5ff"))
                    time.sleep(0.5)
                    continue
                if self.running and not self.session_complete:
                    self.status_messages.put((str(error), "#ff7b72"))
                break

    def current_action(self) -> dict:
        key_actions = {
            "w": "forward",
            "s": "back",
            "a": "left",
            "d": "right",
            "space": "jump",
            "shift_l": "sneak",
            "shift_r": "sneak",
            "control_l": "sprint",
            "control_r": "sprint",
            "e": "inventory",
        }
        action = {name: 0 for name in ["attack", "back", "forward", "jump", "left", "right", "sneak", "sprint", "use", "inventory"]}
        for number in range(1, 10):
            action[f"hotbar.{number}"] = int(str(number) in self.keys)
        with self.lock:
            for key_name, action_name in key_actions.items():
                if key_name in self.keys:
                    action[action_name] = 1
            action["attack"] = int(1 in self.mouse_buttons)
            action["use"] = int(3 in self.mouse_buttons)
            action["camera"] = [
                max(-CAMERA_MAX_DEGREES, min(CAMERA_MAX_DEGREES, value))
                for value in self.camera
            ]
            self.camera = [0.0, 0.0]
        return action

    def key_press(self, event) -> None:
        key_name = event.keysym.lower()
        if key_name == "c":
            self.set_mouse_capture(not self.mouse_captured)
        elif key_name == "escape":
            self.set_mouse_capture(False)
        else:
            with self.lock:
                self.keys.add(key_name)

    def key_release(self, event) -> None:
        with self.lock:
            self.keys.discard(event.keysym.lower())

    def mouse_press(self, button: int) -> None:
        with self.lock:
            self.mouse_buttons.add(button)

    def mouse_release(self, button: int) -> None:
        with self.lock:
            self.mouse_buttons.discard(button)

    def capture_from_click(self, event) -> None:
        if not self.mouse_captured:
            self.set_mouse_capture(True)

    def set_mouse_capture(self, captured: bool) -> None:
        self.mouse_captured = captured
        self.root.configure(cursor="none" if captured else "")
        if captured:
            self.recenter_mouse()

    def recenter_mouse(self) -> None:
        self.root.update_idletasks()
        center_x = self.root.winfo_rootx() + self.root.winfo_width() // 2
        center_y = self.root.winfo_rooty() + self.root.winfo_height() // 2
        self.ignore_warp = True
        ctypes.windll.user32.SetCursorPos(center_x, center_y)

    def mouse_motion(self, event) -> None:
        if not self.mouse_captured:
            return
        center_x = self.root.winfo_width() // 2
        center_y = self.root.winfo_height() // 2
        delta_x = event.x - center_x
        delta_y = event.y - center_y
        if self.ignore_warp and abs(delta_x) <= 2 and abs(delta_y) <= 2:
            self.ignore_warp = False
            return
        with self.lock:
            self.camera[0] += delta_y * CAMERA_SENSITIVITY
            self.camera[1] += delta_x * CAMERA_SENSITIVITY
        self.recenter_mouse()

    def focus_out(self, event) -> None:
        with self.lock:
            self.keys.clear()
            self.mouse_buttons.clear()
            self.camera = [0.0, 0.0]
        self.set_mouse_capture(False)

    def render_latest(self) -> None:
        while not self.status_messages.empty():
            status_text, status_color = self.status_messages.get_nowait()
            self.canvas.itemconfigure(self.status_item, text=status_text, fill=status_color)
        latest = None
        while not self.frames.empty():
            latest = self.frames.get_nowait()
        if latest is not None:
            header, image_data = latest
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = image.resize((1060, 596), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            progress = header["progress"]
            threshold = header["threshold"]
            position = header["player_pos"]
            status = header["status"]
            global_progress = header.get("global_progress", 0.0)
            global_threshold = header.get("global_threshold", 0.0)
            recording = f"REC {header['recorded_frames']} | " if header.get("recording") else ""
            self.canvas.itemconfigure(self.status_item, text=f"{recording}{status} | Global {global_progress:.1f}/{global_threshold:g} | Subtask {progress:.1f}/{threshold:g} | Health {header['health']:.0f} | Food {header['food_level']:.0f} | XYZ {position['x']:.1f}, {position['y']:.1f}, {position['z']:.1f}", fill="#8bd5ff")
            self.canvas.itemconfigure(self.task_item, text="TASK: " + header["instruction"])
            self.canvas.itemconfigure(self.hint_item, text="HINT: " + header["hint"])
            self.canvas.itemconfigure(self.feedback_item, text="GLOBAL: " + header.get("global_goal", "") + " | GEMINI: " + header["feedback"])
            if header.get("session_complete"):
                self.root.after(2000, self.close)
        if self.running:
            self.root.after(15, self.render_latest)

    def close(self) -> None:
        self.running = False
        if self.connection is not None:
            payload = json.dumps({"shutdown": True}, separators=(",", ":")).encode("utf-8")
            try:
                self.connection.sendall(struct.pack("!I", len(payload)) + payload)
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    viewer = GeminiCoachViewer()
    try:
        viewer.run()
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()
