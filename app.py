import os
import sys
import random
import threading
import queue 
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

from pynput import keyboard

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QCheckBox, QPushButton
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

APP_TITLE = "Keyvibes"
def resource_path(*parts: str) -> str:
    # Wenn als EXE gebaut: PyInstaller entpackt nach sys._MEIPASS
    base = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    return os.path.join(base, *parts)

SOUNDS_DIR = resource_path("sounds")
SAMPLE_RATE = 44100
CHANNELS = 2

def load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if sr != target_sr:
        raise ValueError(f"{path} hat {sr} Hz, erwartet {target_sr} Hz. Bitte WAV auf 44100Hz konvertieren.")
    return data

def list_packs() -> List[str]:
    if not os.path.isdir(SOUNDS_DIR):
        return []
    return sorted([d for d in os.listdir(SOUNDS_DIR) if os.path.isdir(os.path.join(SOUNDS_DIR, d))])

def list_category_files(pack: str) -> Dict[str, List[str]]:
    base = os.path.join(SOUNDS_DIR, pack)
    cats = ["normal", "space", "backspace", "enter", "modifier", "arrow", "function"]
    out: Dict[str, List[str]] = {}
    for c in cats:
        files = []
        if os.path.isdir(base):
            for fn in os.listdir(base):
                f = fn.lower()
                if f.startswith(c + "_") and f.endswith(".wav"):
                    files.append(os.path.join(base, fn))
        out[c] = sorted(files)
    return out

def key_to_category(key) -> str:
    if key == keyboard.Key.space:
        return "space"
    if key == keyboard.Key.enter:
        return "enter"
    if key == keyboard.Key.backspace:
        return "backspace"
    
    if key in (
        keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
        keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
        keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
        keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r,
        keyboard.Key.tab, keyboard.Key.caps_lock, keyboard.Key.esc
    ):
        return "modifier"
    
    if key in (
        keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right,
        keyboard.Key.home, keyboard.Key.end, keyboard.Key.page_down, keyboard.Key.page_up,
        keyboard.Key.insert, keyboard.Key.delete
    ):
        return "arrow"
    
    if isinstance(key, keyboard.Key) and str(key).startswith("Key.f"):
        return "function"
    
    return "normal"


@dataclass
class ActiveSound:
    data: np.ndarray
    pos: int
    gain: float


class AudioMixer:
    def __init__(self, samplerate=SAMPLE_RATE, channels=CHANNELS):
        self.samplerate = samplerate
        self.channels = channels
        self.volume = 0.5
        self._lock = threading.Lock()
        self._active: List[ActiveSound] = []
        self._event_q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue()

        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            blocksize=0
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def set_volume(self, v: float):
        self.volume = float(np.clip(v, 0.0, 1.0))

    def play(self, data: np.ndarray, gain: float = 1.0):
        self._event_q.put((data, gain))

    
    def _drain_events(self):
        try:
            while True:
                data, gain = self._event_q.get_nowait()
                self._active.append(ActiveSound(data=data, pos=0, gain=gain))
        except queue.Empty:
            return
        
    def _callback(self, outdata, frames, time, status):
        with self._lock:
            self._drain_events()
            mix = np.zeros((frames, self.channels), dtype=np.float32)

            still_active: List[ActiveSound] = []
            for s in self._active:
                remaining = s.data.shape[0] - s.pos
                n = min(frames, remaining)
                if n > 0:
                    mix[:n] += s.data[s.pos:s.pos + n] * (s.gain * self.volume)
                    s.pos += n
                if s.pos < s.data.shape[0]:
                    still_active.append(s)

            self._active = still_active

        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:] = mix


class SoundEngine:
    def __init__(self, mixer: AudioMixer):
        self.mixer = mixer
        self.pack = None
        self.sounds: Dict[str, List[np.ndarray]] = {}
        self.enabled = True

    def load_pack(self, pack: str):
        files = list_category_files(pack)
        loaded: Dict[str, List[np.ndarray]] = {}
        for cat, paths in files.items():
            arrs = []
            for p in paths:
                arrs.append(load_wav(p))
            loaded[cat] = arrs
        self.pack = pack
        self.sounds = loaded

    def trigger(self, category: str):
        if not self.enabled:
            return
        options = self.sounds.get(category, [])
        if not options:
            options = self.sounds.get("normal", [])
        if not options:
            return
        

        sample = random.choice(options)
        gain = random.uniform(0.85, 1.05)
        self.mixer.play(sample, gain=gain)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setFixedSize(360, 520)

        self.mixer = AudioMixer()
        self.engine = SoundEngine(self.mixer)
        self.listener = None

        #Tray
        self.tray = None
        self.tray_enabled = True

        self._build_ui()
        self._start_audio()
        self._start_listener()

        self._init_tray()

    def _on_tray_toggled(self, checked: bool):
        self.tray_enabled = bool(checked)
        if self.tray:
            self.tray.setVisible(self.tray_enabled)

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(18)

        title = QLabel("Keyvibes")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 34px; font-weight: 700;")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sound"))
        self.pack_box = QComboBox()
        packs = list_packs()
        self.pack_box.addItems(packs if packs else ["(keine Packs gefunden)"])
        self.pack_box.currentTextChanged.connect(self._on_pack_changed)
        row.addWidget(self.pack_box)
        layout.addLayout(row)

        self.vol_label = QLabel("Volume 50")
        layout.addWidget(self.vol_label)

        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_slider.valueChanged.connect(self._on_volume)
        layout.addWidget(self.vol_slider)

        self.enable_box = QCheckBox("Enable key sounds")
        self.enable_box.setChecked(True)
        self.enable_box.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_box)

        test_row = QHBoxLayout()
        btn_test = QPushButton("Test key")
        btn_test.clicked.connect(lambda: self.engine.trigger("normal"))
        test_row.addWidget(btn_test)

        btn_space = QPushButton("Test space")
        btn_space.clicked.connect(lambda: self.engine.trigger("space"))
        test_row.addWidget(btn_space)
        layout.addLayout(test_row)

        hint = QLabel(
            "WAV-Dateien in ./sounds/<pack>/\n"
            "z.B. normal_1.wav, space_1.wav, backspace_1.wav ..."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        

        first = self.pack_box.currentText()
        if first and not first.startswith("("):
            self.engine.load_pack(first)

        self.tray_box = QCheckBox("Show Tray Icon")
        self.tray_box.setChecked(True)
        self.tray_box.toggled.connect(self._on_tray_toggled)
        layout.addWidget(self.tray_box)

        self.setLayout(layout)

    def _start_audio(self):
        self.mixer.start()
    
    def _start_listener(self):
        def on_press(key):
            cat = key_to_category(key)
            self.engine.trigger(cat)

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def _on_pack_changed(self, pack: str):
        if not pack or pack.startswith("("):
            return
        try:
            self.engine.load_pack(pack)
        except Exception as e:
            print("Pack load error:", e)

    def _on_volume(self, v: int):
        self.vol_label.setText(f"Volume {v}")
        self.mixer.set_volume(v / 100.0)

    def _on_enabled_toggled(self, checked: bool):
        self.engine.enabled = bool(checked)
    
    def _init_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_enabled = False
            return

        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = (
            QIcon(icon_path)
            if os.path.exists(icon_path)
            else self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)
        )

        self.tray = QSystemTrayIcon(icon, self)
        menu = QMenu()

        act_show = QAction("Show / Hide", self)
        act_show.triggered.connect(self._toggle_window)

        act_enable = QAction("Enable Sounds", self)
        act_enable.setCheckable(True)
        act_enable.setChecked(True)
        act_enable.toggled.connect(lambda v: self.enable_box.setChecked(v))

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self._quit_app)

        menu.addAction(act_show)
        menu.addSeparator()
        menu.addAction(act_enable)
        menu.addSeparator()
        menu.addAction(act_quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _toggle_window(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._toggle_window()

    def _quit_app(self):
        self.tray_enabled = False
        self.close()
        QApplication.quit()


    def closeEvent(self, event):
        if self.tray_enabled and self.tray and self.tray.isVisible():
            self.hide()
            event.ignore()
            return

        try:
            if self.listener:
                self.listener.stop()
            self.mixer.stop()
        finally:
            event.accept()



def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()