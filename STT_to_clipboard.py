import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext, filedialog
import pyperclip
import winsound
import queue
import json
import time
import threading
import tempfile
from datetime import datetime
from transcription_history import TranscriptionHistory
from speech_engine import SpeechBackend, SpeechEngine
from whisper_models import MODEL_CHOICES, MODEL_NAMES, normalize_model, checkpoint_path
from audio_input import list_microphones
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("keyboard library not available - hotkey feature disabled")

# ===============================
# Windows Taskbar Icon Fix
# ===============================
# Tell Windows this is a separate app (not grouped with Python)
try:
    from ctypes import windll
    myappid = 'mycompany.stt_clipboard.app.1'  # arbitrary string
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except:
    pass  # Not on Windows or ctypes not available


# ===============================
# Configuration Management
# ===============================
APP_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
CONFIG_FILE = os.path.join(APP_DIR, "stt_settings.json")
MIN_WINDOW_WIDTH = 260
MIN_WINDOW_HEIGHT = 38

def resource_path(path):
    return path if os.path.isabs(path) else os.path.join(RESOURCE_DIR, path)

DEFAULT_CONFIG = {
    "always_on_top": False,
    "hotkey": "ctrl+shift+r",
    "notification_sound": "notification sound.wav",
    "window_size": "600x240",
    "real_time_mode": False,
    "whisper_model": "base",
    "history_limit": 10
}

def load_config():
    """Load configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
    except Exception as e:
        print(f"Error loading config: {e}")
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=os.path.dirname(CONFIG_FILE),
                                         prefix='.settings-', suffix='.tmp', delete=False) as f:
            temporary = f.name
            json.dump(config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, CONFIG_FILE)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        messagebox.showerror("Settings", f"Settings could not be saved.\n\n{e}")
        return False
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

# ===============================
# Sound Notification
# ===============================
def play_notification_sound():
    """Play notification sound when transcription completes"""
    try:
        sound_file = resource_path(app_config.get("notification_sound", "notification sound.wav"))
        if os.path.exists(sound_file):
            winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            print(f"Sound file not found: {sound_file}")
    except Exception as e:
        print(f"Could not play notification sound: {e}")


# ===============================
# Recording controls (all Tk and clipboard operations stay on the UI thread)
# ===============================
def replace_transcript(text):
    scroll_position = transcription_text_area.yview()
    follow_tail = not transcription_frame.winfo_ismapped() or scroll_position[1] >= .98
    transcription_text_area.config(state="normal")
    transcription_text_area.delete("1.0", tk.END)
    transcription_text_area.insert(tk.END, text)
    if follow_tail:
        transcription_text_area.see(tk.END)
    else:
        transcription_text_area.yview_moveto(scroll_position[0])
    if state != "idle":
        transcription_text_area.config(state="disabled")


def recording_button_text():
    small = root.winfo_width() < 320
    if state == "loading":
        return ("Cancel" if small else "Cancel loading") if real_time_mode else ("Stop" if small else "Stop Recording")
    if state == "recording":
        return "Stop" if small else "Stop Recording"
    if state == "finalizing":
        return "Wait..." if small else "Transcribing..."
    if time.monotonic() < restart_allowed_at:
        return "Wait..." if small else "Please wait..."
    return "Start" if small else "Start Recording"


def refresh_controls():
    busy = state != "idle"
    guarded = time.monotonic() < restart_allowed_at
    record_button.config(text=recording_button_text(),
                         state="disabled" if state == "finalizing" or guarded else "normal")
    mode_button.config(state="disabled" if busy or guarded else "normal")
    real_time_var.set(real_time_mode)
    transcription_text_area.config(state="disabled" if busy else "normal")
    hotkey_button.config(state="disabled" if busy else "normal")
    model_combo.config(state="disabled" if busy else "readonly")
    retry_button.config(state="normal" if not busy and engine.pending_audio is not None else "disabled")
    microphone_combo.config(state="disabled" if busy or microphones_loading else "readonly")
    microphone_refresh.config(state="disabled" if busy or microphones_loading else "normal")
    mode_label.config(text="Real-time transcription" if real_time_mode else "Standard recording")
    if state == "recording":
        status_indicator.config(text=("RT QUIET" if real_time_mode else "QUIET") if input_quiet
                                else ("\u25cf RT REC" if real_time_mode else "\u25cf REC"),
                                bg="#ffaa00" if input_quiet else "#00ff00")
    else:
        draw_input_level(0)
    if history_restore_button is not None and history_restore_button.winfo_exists():
        history_restore_button.config(state="disabled" if busy else "normal")
    update_window_layout()


def release_restart_guard():
    if closing:
        return
    remaining = restart_allowed_at - time.monotonic()
    if remaining > 0:
        root.after(max(1, int(remaining * 1000) + 1), release_restart_guard)
    else:
        refresh_controls()


def guard_restart():
    global restart_allowed_at
    restart_allowed_at = time.monotonic() + restart_guard_seconds
    root.after(int(restart_guard_seconds * 1000) + 1, release_restart_guard)


def toggle_mode():
    global real_time_mode
    if state != "idle" or time.monotonic() < restart_allowed_at:
        real_time_var.set(real_time_mode)
        return
    real_time_mode = not real_time_mode
    refresh_controls()


def toggle_recording(requested_at=None):
    global state, active_id, previous_transcript, session_preview, input_quiet
    now = time.monotonic()
    if state == "finalizing" or now < restart_allowed_at:
        return
    # Reject hotkey presses queued during finalization, even if delivered later.
    if requested_at is not None and requested_at < restart_allowed_at:
        return
    if state in ("loading", "recording"):
        state = "finalizing"
        status_indicator.config(bg="#ffaa00", fg="white", text="\u25cf WAIT")
        refresh_controls()  # Disable synchronously, before stopping capture.
        engine.stop()
    else:
        input_quiet = False
        session_preview = ""
        previous_transcript = transcription_text_area.get("1.0", "end-1c")
        state = "loading"
        status_indicator.config(bg="#ffaa00", fg="white", text="\u25cf LOAD")
        model_status.set(f"Preparing {app_config['whisper_model']} — wait for green REC before speaking."
                         if real_time_mode else "Starting microphone…")
        active_id = engine.start(real_time_mode, app_config.get("microphone"), app_config['whisper_model'])
    refresh_controls()


def cancel_recording():
    global state, active_id
    if state == "idle":
        return
    engine.cancel()
    active_id = None
    state = "idle"
    guard_restart()
    replace_transcript(previous_transcript)
    status_indicator.config(bg="#ff0000", fg="white", text="\u25cf READY")
    refresh_controls()


def process_events():
    global state, active_id, session_preview, input_quiet
    if closing:
        return
    # Keyboard callbacks only enqueue commands; they never call Tkinter.
    for _ in range(100):
        try:
            action = ui_events.get_nowait()
        except queue.Empty:
            break
        action()
    for _ in range(100):
        try:
            session_id, kind, value = engine.events.get_nowait()
        except queue.Empty:
            break
        if session_id != active_id:
            continue
        if kind == "level":
            if state == "recording":
                input_quiet = value['quiet']
                draw_input_level(value['level'])
                audio_status.set("Very low input: check mute, microphone volume, or the selected device."
                                 if input_quiet else "Receiving audio. The bar shows input volume, not speech recognition accuracy.")
        elif kind == "loading":
            if state == "loading" or (state == "finalizing" and not real_time_mode):
                status_indicator.config(text="\u25cf LOAD")
                model_status.set((value or "Preparing model…") + (" Wait for green REC before speaking."
                                 if real_time_mode else " Your recording is held in memory."))
        elif kind == "download":
            if state == "loading" or (state == "finalizing" and not real_time_mode):
                status_indicator.config(text=f"DL {value}%")
                model_status.set(f"Downloading {app_config['whisper_model']}: {value}% — " +
                                 ("microphone not started yet." if real_time_mode else "recording held in memory."))
        elif kind == "recording":
            if state != "finalizing":
                state = "recording"
                status_indicator.config(bg="#00ff00", fg="white", text="\u25cf REC")
                model_status.set(f"Recording with {app_config['whisper_model']}.")
        elif kind == "finalizing":
            state = "finalizing"
            status_indicator.config(bg="#ffaa00", fg="white", text="\u25cf WAIT")
            model_status.set("Processing your recording…")
        elif kind == "partial":
            session_preview = value
            replace_transcript(value)
        elif kind == "recovered":
            session_preview = session_preview or value
        elif kind == "final":
            state = "idle"
            active_id = None
            guard_restart()
            replace_transcript(value if value else previous_transcript)
            status_indicator.config(bg="#ff0000", fg="white", text="\u25cf READY")
            update_model_status()
            if value:
                remember_transcription(value)
                try:
                    pyperclip.copy(value)
                except Exception as exc:
                    messagebox.showerror("Clipboard", f"Transcript is ready, but copying failed: {exc}")
                play_notification_sound()
        elif kind == "error":
            state = "idle"
            active_id = None
            guard_restart()
            replace_transcript(session_preview if real_time_mode and session_preview else previous_transcript)
            status_indicator.config(bg="#ff0000", fg="white", text="\u25cf ERROR")
            retry_hint = ("\n\nYour recording is still in memory. Open Settings and choose Retry last failed recording. "
                          "You can select a different model first. Closing the app discards this audio."
                          if engine.pending_audio is not None else "")
            model_status.set("Recording failed. Retry is available in Settings." if retry_hint else "Recording failed. See the error message.")
            refresh_controls()
            messagebox.showerror("Speech to Text", value + retry_hint)
        refresh_controls()
    root.after(50, process_events)


def close_app():
    global closing
    closing = True
    if root.state() == "normal":
        app_config["window_size"] = f"{root.winfo_width()}x{root.winfo_height()}"
        save_config(app_config)
    engine.cancel()
    engine.pending_audio = None
    try:
        if KEYBOARD_AVAILABLE:
            keyboard.unhook_all()
    finally:
        root.destroy()


# ===============================
# Transcript and history actions
# ===============================
def remember_transcription(text):
    try:
        history.add(text)
    except OSError as exc:
        messagebox.showwarning("Transcription history", f"Your transcript is available, but history could not be saved.\n\n{exc}")
    history_button.config(text=f"History ({len(history.entries)})")
    if history_refresh is not None:
        history_refresh()


def restore_history_entry(entry):
    if state != "idle":
        return False
    replace_transcript(entry["text"])
    status_indicator.config(bg="#ff0000", fg="white", text="\u25cf READY")
    return True


def open_history():
    global history_window, history_refresh, history_restore_button
    open_settings(history_tab)
    if history_refresh is not None:
        history_refresh()
        return
    history_window = settings_window
    dialog = history_tab
    heading = ttk.Frame(dialog, padding=(22, 18, 22, 10))
    heading.pack(fill=tk.X)
    ttk.Label(heading, text="Your recent transcriptions", style="Heading.TLabel").pack(anchor="w")
    ttk.Label(heading, textvariable=history_summary,
              style="TLabel").pack(anchor="w", pady=(5, 0))
    body = ttk.Frame(dialog, padding=(22, 0, 22, 0))
    body.pack(fill=tk.BOTH, expand=True)
    table = ttk.Treeview(body, columns=("date", "preview"), show="headings", selectmode="browse", height=5)
    table.heading("date", text="Recorded")
    table.heading("preview", text="Transcription")
    table.column("date", width=150, stretch=False)
    table.column("preview", width=540)
    scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    table.pack(fill=tk.BOTH, expand=True)
    ttk.Label(dialog, text="FULL TRANSCRIPTION", style="TLabel").pack(anchor="w", padx=22, pady=(14, 6))
    preview = scrolledtext.ScrolledText(dialog, height=5, wrap=tk.WORD, font=("Consolas", 10),
                bg="white", fg="black", relief="sunken", bd=1, padx=5, pady=5)
    preview.pack(fill=tk.BOTH, expand=True, padx=22)
    feedback = ttk.Label(dialog, text="Select a recording to read, restore, or copy.", style="TLabel")
    feedback.pack(anchor="w", padx=22, pady=(8, 0))
    actions = ttk.Frame(dialog, padding=(22, 10, 22, 18))
    actions.pack(fill=tk.X)

    def selected():
        ids = table.selection()
        return next((e for e in history.entries if ids and e["id"] == ids[0]), None)

    def show_selection(event=None):
        entry = selected()
        preview.config(state="normal")
        preview.delete("1.0", tk.END)
        preview.insert("1.0", entry["text"] if entry else "Completed transcriptions will appear here.")
        preview.config(state="disabled")

    def refresh():
        selection = table.selection()
        table.delete(*table.get_children())
        for entry in history.entries:
            try:
                date = datetime.fromisoformat(entry["created_at"]).strftime("%b %d, %I:%M %p")
            except ValueError:
                date = entry["created_at"]
            table.insert("", tk.END, iid=entry["id"], values=(date, " ".join(entry["text"].split())[:160]))
        if history.entries:
            item = selection[0] if selection and table.exists(selection[0]) else history.entries[0]["id"]
            table.selection_set(item)
        show_selection()

    def restore():
        entry = selected()
        if entry and restore_history_entry(entry):
            feedback.config(text="Restored to the main transcription. Your clipboard is unchanged.")
        elif entry:
            feedback.config(text="Wait for the current recording to finish before restoring.")

    def copy():
        entry = selected()
        if entry:
            try:
                pyperclip.copy(entry["text"])
                feedback.config(text="Copied to clipboard.")
            except Exception as exc:
                feedback.config(text=f"Copy failed: {exc}")

    history_restore_button = ttk.Button(actions, text="Restore to editor", command=restore)
    history_restore_button.pack(side=tk.LEFT)
    ttk.Button(actions, text="Copy to clipboard", command=copy).pack(side=tk.LEFT, padx=8)
    ttk.Button(actions, text="Close", command=close_settings).pack(side=tk.RIGHT)
    table.bind("<<TreeviewSelect>>", show_selection)
    history_refresh = refresh
    refresh()
    refresh_controls()
    if history.load_error:
        feedback.config(text="Saved history could not be read. New entries are available for this session only.")


# ===============================
# Settings Functions
# ===============================
def update_model_status():
    name = app_config['whisper_model']
    if checkpoint_path(RESOURCE_DIR, name).is_file():
        model_status.set(f"{name}: downloaded. Loads after Stop (before Start in real-time mode).")
    else:
        model_status.set(f"{name}: downloads after Stop (before Start in real-time mode).")


def draw_input_level(level):
    meter_canvas.coords(meter_fill, 0, 0, max(0, meter_canvas.winfo_width() * level), 5)
    meter_canvas.itemconfig(meter_fill, fill="#ffc247" if input_quiet else "#00d85a")


def refresh_microphones():
    global microphones_loading
    if state != "idle" or microphones_loading:
        return
    microphones_loading = True
    audio_status.set("Looking for input devices…")
    refresh_controls()
    def finished(devices, error):
        global microphones_loading, microphone_choices
        microphones_loading = False
        microphone_choices = [None, *devices]
        selection = app_config.get('microphone')
        index = 0
        if selection is not None:
            matches = [i for i, d in enumerate(microphone_choices) if d is not None and
                       ((isinstance(selection, dict) and d['name'] == selection.get('name')
                         and d['host_api'] == selection.get('host_api')) or d['index'] == selection)]
            if matches:
                index = matches[0]
            else:
                microphone_choices.append(selection)
                index = len(microphone_choices) - 1
        labels = ['Windows default microphone'] + [f"{d['name']} [{d['host_api']}] (#{d['index']})" for d in devices]
        if len(microphone_choices) > len(labels):
            labels.append('Selected microphone unavailable — choose a connected device')
        microphone_combo.config(values=labels)
        microphone_combo.current(index)
        audio_status.set(error or "Choose an input, then record. Watch the bar beneath REC in the main window.")
        refresh_controls()
    def discover():
        try:
            devices, error = list_microphones(), ''
        except Exception as exc:
            devices, error = [], f"Could not list microphones: {exc}"
        ui_events.put(lambda: finished(devices, error))
    threading.Thread(target=discover, daemon=True).start()


def select_microphone(event=None):
    if state != "idle" or microphones_loading:
        return
    index = microphone_combo.current()
    if index < 0 or index >= len(microphone_choices):
        return
    previous = app_config.get('microphone')
    app_config['microphone'] = microphone_choices[index]
    if not save_config(app_config):
        app_config['microphone'] = previous
        refresh_microphones()
        return
    audio_status.set("Microphone saved. It will be used for your next recording.")


def retry_recording():
    global state, active_id, real_time_mode, previous_transcript, session_preview
    if state != "idle" or engine.pending_audio is None:
        return
    previous_transcript = transcription_text_area.get("1.0", "end-1c")
    session_preview = ""
    real_time_mode = False
    state = "finalizing"
    status_indicator.config(bg="#ffaa00", fg="white", text="\u25cf WAIT")
    model_status.set("Retrying the saved recording in memory…")
    active_id = engine.retry(app_config['whisper_model'])
    refresh_controls()


def select_model(event=None):
    if state != "idle":
        model_combo.current(MODEL_NAMES.index(app_config['whisper_model']))
        return
    previous = app_config['whisper_model']
    app_config['whisper_model'] = MODEL_NAMES[model_combo.current()]
    if not save_config(app_config):
        app_config['whisper_model'] = previous
        model_combo.current(MODEL_NAMES.index(previous))
    update_model_status()


def apply_history_limit():
    try:
        history.set_limit(history_limit_var.get().strip())
    except (ValueError, OSError) as exc:
        history_feedback.set(str(exc))
        return
    app_config['history_limit'] = history.limit
    saved = save_config(app_config)
    history_summary.set(f"Last {history.limit} completed recordings  |  Saved on this computer  |  Text only")
    history_button.config(text=f"History ({len(history.entries)})")
    history_feedback.set(f"Saved: keep the last {history.limit} completed transcriptions." if saved
                         else "History updated, but the limit could not be saved for the next launch. Try Apply again.")
    if history_refresh is not None:
        history_refresh()


def toggle_always_on_top():
    """Toggle the always on top setting"""
    is_on_top = always_on_top_var.get()
    root.attributes('-topmost', is_on_top)
    app_config["always_on_top"] = is_on_top
    save_config(app_config)

def open_hotkey_dialog():
    dialog = tk.Toplevel(root)
    dialog.title("Configure Hotkey")
    dialog.geometry("420x160")
    dialog.transient(root)
    dialog.grab_set()
    ttk.Label(dialog, text="Hotkey (examples: ctrl+shift+r, alt+r, f9)").pack(pady=12)
    entry = ttk.Entry(dialog, width=40)
    entry.insert(0, app_config.get("hotkey", "ctrl+shift+r"))
    entry.pack(padx=15)
    feedback = ttk.Label(dialog, text="")
    feedback.pack()

    def save():
        global hotkey_handle
        new_hotkey = entry.get().strip().lower()
        if not new_hotkey:
            feedback.config(text="Enter a key combination.")
            return
        if KEYBOARD_AVAILABLE:
            try:
                new_handle = keyboard.add_hotkey(new_hotkey, hotkey_toggle_recording, trigger_on_release=True)
            except Exception as exc:
                feedback.config(text=f"Invalid hotkey: {exc}")
                return
            if hotkey_handle is not None:
                keyboard.remove_hotkey(hotkey_handle)
            hotkey_handle = new_handle
        app_config["hotkey"] = new_hotkey
        save_config(app_config)
        hotkey_label.config(text=f"Hotkey: {new_hotkey}")
        dialog.destroy()

    ttk.Button(dialog, text="Save", command=save).pack(pady=8)


def select_notification_sound():
    """Open file dialog to select notification sound"""
    filename = filedialog.askopenfilename(
        title="Select Notification Sound",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        initialdir=APP_DIR
    )

    if filename:
        app_config["notification_sound"] = filename
        save_config(app_config)
        # Show just the filename in the button
        sound_name = os.path.basename(filename)
        if len(sound_name) > 20:
            sound_name = sound_name[:17] + "..."
        sound_button.config(text=f"Sound: {sound_name}")
        messagebox.showinfo("Success", f"Notification sound changed to:\n{os.path.basename(filename)}")

def hotkey_toggle_recording():
    """Called when hotkey is pressed"""
    if not closing:
        requested_at = time.monotonic()
        ui_events.put(lambda: toggle_recording(requested_at))

def open_settings(tab=None):
    settings_notebook.select(settings_frame if tab is None else tab)
    settings_window.deiconify()
    settings_window.lift()


def close_settings():
    settings_window.withdraw()


def settings_tab_changed(event=None):
    if settings_notebook.select() == str(history_tab) and history_refresh is None:
        open_history()
    elif settings_notebook.select() == str(audio_tab) and not microphone_combo['values']:
        refresh_microphones()


def update_window_layout(event=None):
    """Keep recording controls usable when the user makes the window smaller."""
    global layout_state
    if event is not None and event.widget is not root:
        return
    narrow = root.winfo_width() < 460
    tiny = root.winfo_width() < 320
    hide_transcript = root.winfo_height() < 85
    current_layout = (narrow, tiny, hide_transcript, real_time_mode)
    if current_layout == layout_state:
        return
    layout_state = current_layout
    settings_button.config(text="\u2699" if narrow else "\u2699 Settings",
                           width=3 if narrow else -1)
    record_button.config(text=recording_button_text(), width=6 if tiny else -1)
    status_indicator.config(width=7 if narrow else 8, padx=3 if narrow else 5)
    transcription_frame.config(text="Transcription | REAL-TIME ON" if real_time_mode else "Transcription")
    if hide_transcript:
        transcription_frame.pack_forget()
    else:
        transcription_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 3))


# ===============================
# GUI Setup -- classic Windows controls, matching the compact original
# ===============================
root = tk.Tk()
root.title("Speech to Text & Clipboard")
root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
root.resizable(True, True)
if os.path.isfile(resource_path("stt_Clipboard.ico")):
    root.iconbitmap(resource_path("stt_Clipboard.ico"))
app_config = load_config()
app_config['whisper_model'] = normalize_model(app_config.get('whisper_model', 'base'))
try:
    app_config['history_limit'] = TranscriptionHistory.validate_limit(app_config.get('history_limit', 10))
except ValueError:
    app_config['history_limit'] = 10
try:
    width, height = map(int, app_config.get("window_size", "600x240").split("x"))
    root.geometry(f"{max(MIN_WINDOW_WIDTH, width)}x{max(MIN_WINDOW_HEIGHT, height)}")
except (ValueError, AttributeError, tk.TclError):
    root.geometry("600x240")
history = TranscriptionHistory(os.path.join(APP_DIR, "transcription_history.json"), app_config['history_limit'])
history_summary = tk.StringVar(value=f"Last {history.limit} completed recordings  |  Saved on this computer  |  Text only")

style = ttk.Style(root)
if "vista" in style.theme_names():
    style.theme_use("vista")
style.configure("TButton", padding=(5, 8), font=("Segoe UI", 9))
style.configure("Recorder.TButton", padding=(3, 3), font=("Segoe UI", 9))
style.configure("Recorder.TCheckbutton", font=("Segoe UI", 9))
style.configure("TLabel", font=("Segoe UI", 10))
style.configure("TCheckbutton", font=("Segoe UI", 10))
style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))
style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

# Settings and history share a separate window.
settings_window = tk.Toplevel(root)
settings_window.withdraw()
settings_window.title("Speech to Text - Settings")
settings_window.geometry("820x600")
settings_window.minsize(680, 580)
settings_window.transient(root)
settings_window.protocol("WM_DELETE_WINDOW", close_settings)
settings_window.bind("<Escape>", lambda event: close_settings())
settings_notebook = ttk.Notebook(settings_window)
settings_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
settings_frame = ttk.Frame(settings_notebook, padding=15)
history_tab = ttk.Frame(settings_notebook)
audio_tab = ttk.Frame(settings_notebook, padding=15)
settings_notebook.add(settings_frame, text="Settings")
settings_notebook.add(audio_tab, text="Audio")
settings_notebook.add(history_tab, text="Transcription history")
settings_notebook.bind("<<NotebookTabChanged>>", settings_tab_changed)

always_on_top_var = tk.BooleanVar(value=app_config.get("always_on_top", False))
ttk.Checkbutton(settings_frame, text="Always on top", variable=always_on_top_var,
                command=toggle_always_on_top).grid(row=0, column=0, sticky="w", padx=(0, 15), pady=5)
hotkey_button = ttk.Button(settings_frame, text="Configure hotkey", command=open_hotkey_dialog)
hotkey_button.grid(row=1, column=0, sticky="w", pady=5)
hotkey_label = ttk.Label(settings_frame, text=f"Hotkey: {app_config.get('hotkey', 'ctrl+shift+r')}")
hotkey_label.grid(row=1, column=1, padx=12, sticky="w")
sound_name = os.path.basename(app_config.get("notification_sound", "notification sound.wav"))
sound_button = ttk.Button(settings_frame, text=f"Sound: {sound_name[:23]}", command=select_notification_sound)
sound_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
mode_label = ttk.Label(settings_frame, text="Standard recording")
mode_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=12)
history_button = ttk.Button(settings_frame, text=f"History ({len(history.entries)})", command=open_history)
history_button.grid(row=5, column=0, sticky="w", pady=(12, 5))

model_frame = ttk.LabelFrame(settings_frame, text="Whisper model", padding=10)
model_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
model_combo = ttk.Combobox(model_frame, values=[label for _, label in MODEL_CHOICES],
                          state="readonly", width=48, height=len(MODEL_CHOICES))
model_combo.current(MODEL_NAMES.index(app_config['whisper_model']))
model_combo.pack(anchor="w")
model_combo.bind("<<ComboboxSelected>>", select_model)
ttk.Label(model_frame, text="Smallest to largest · Approximate download / disk size (RAM use is higher).\n"
          "Applies to recording and real-time. This app transcribes in English.\n"
          "Recording: downloads after Stop. Real-time: downloads before recording starts.",
          wraplength=580).pack(anchor="w", pady=(8, 4))
model_status = tk.StringVar()
ttk.Label(model_frame, textvariable=model_status, wraplength=580).pack(anchor="w")
update_model_status()

retention_frame = ttk.Frame(settings_frame)
retention_frame.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
ttk.Label(retention_frame, text="Keep the last").pack(side=tk.LEFT)
history_limit_var = tk.StringVar(value=str(history.limit))
ttk.Spinbox(retention_frame, from_=1, to=1000, textvariable=history_limit_var,
            width=6).pack(side=tk.LEFT, padx=8)
ttk.Label(retention_frame, text="completed transcriptions").pack(side=tk.LEFT)
ttk.Button(retention_frame, text="Apply", command=apply_history_limit).pack(side=tk.LEFT, padx=12)
ttk.Label(settings_frame, text="1–1000 entries. Reducing this limit removes the oldest entries. Audio is never saved.",
          wraplength=600).grid(row=7, column=0, columnspan=2, sticky="w", pady=4)
history_feedback = tk.StringVar()
ttk.Label(settings_frame, textvariable=history_feedback, wraplength=600).grid(row=8, column=0, columnspan=2, sticky="w")
retry_button = ttk.Button(settings_frame, text="Retry last failed recording", command=retry_recording, state="disabled")
retry_button.grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
ttk.Label(audio_tab, text="Microphone", style="Heading.TLabel").pack(anchor="w", pady=(0, 10))
microphone_combo = ttk.Combobox(audio_tab, state="readonly", width=60)
microphone_combo.pack(fill=tk.X)
microphone_combo.bind("<<ComboboxSelected>>", select_microphone)
microphone_refresh = ttk.Button(audio_tab, text="Refresh microphones", command=refresh_microphones)
microphone_refresh.pack(anchor="w", pady=10)
ttk.Label(audio_tab, text="Windows default follows the microphone selected in Windows.\n"
          "A specific device is remembered by its name and audio driver.\n\n"
          "While recording, watch the small level bar below REC in the main window.\n"
          "After two seconds of very low input, REC turns amber and says QUIET.\n"
          "This can mean silence, mute, low volume, or the wrong input. Recording continues.\n\n"
          "If the same microphone appears more than once, the entries use different\n"
          "Windows audio drivers. Try the MME entry if another driver cannot open it.",
          wraplength=600).pack(anchor="w", pady=10)
audio_status = tk.StringVar(value="Open this tab to load microphones.")
ttk.Label(audio_tab, textvariable=audio_status, wraplength=600).pack(anchor="w", pady=10)
root.attributes('-topmost', bool(app_config.get("always_on_top", False)))

recording_frame = ttk.Frame(root)
recording_frame.pack(fill=tk.X, padx=6, pady=(5, 4))
recording_frame.columnconfigure(0, weight=1)
recording_frame.columnconfigure(1, weight=1)
recording_frame.columnconfigure(2, weight=1)
settings_button = ttk.Button(recording_frame, text="\u2699 Settings", command=open_settings, style="Recorder.TButton")
settings_button.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
record_button = ttk.Button(recording_frame, text="Start Recording", command=toggle_recording, style="Recorder.TButton")
record_button.grid(row=0, column=1, sticky="nsew", padx=(0, 4))
real_time_var = tk.BooleanVar(value=False)
mode_button = ttk.Checkbutton(recording_frame, text="Real-time", variable=real_time_var, command=toggle_mode,
                             style="Recorder.TCheckbutton")
mode_button.grid(row=0, column=2, sticky="nsew", padx=(0, 4))
status_frame = tk.Frame(recording_frame, bg="black", relief="solid", borderwidth=1)
status_frame.grid(row=0, column=3, sticky="nsew")
meter_canvas = tk.Canvas(status_frame, height=5, width=1, bg="#24513b", highlightthickness=0, bd=0)
meter_canvas.pack(side=tk.BOTTOM, fill=tk.X)
status_indicator = tk.Label(status_frame, text="\u25cf READY", bg="#ff0000", fg="white",
    font=("Segoe UI", 9, "bold"), borderwidth=0, padx=5, pady=0, width=8)
status_indicator.pack(fill=tk.BOTH, expand=True)
meter_fill = meter_canvas.create_rectangle(0, 0, 0, 5, fill="#00d85a", outline="")

transcription_frame = ttk.LabelFrame(root, text="Transcription")
transcription_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 3))
transcription_text_area = scrolledtext.ScrolledText(transcription_frame, wrap=tk.WORD, width=50, height=2,
    font=("Consolas", 10), bg="white", fg="black", relief="sunken", bd=1)
transcription_text_area.pack(fill=tk.BOTH, expand=True, padx=4, pady=3)

state = "idle"
active_id = None
previous_transcript = ""
session_preview = ""
closing = False
hotkey_handle = None
ui_events = queue.Queue()
# Every launch starts in standard mode; real-time is an explicit session choice.
real_time_mode = False
restart_allowed_at = 0.0
restart_guard_seconds = 0.8
try:
    from ctypes import windll
    restart_guard_seconds = max(restart_guard_seconds, windll.user32.GetDoubleClickTime() / 1000 + .1)
except (ImportError, AttributeError):
    pass
history_window = history_refresh = history_restore_button = None
layout_state = None
input_quiet = False
microphones_loading = False
microphone_choices = [None]
engine = SpeechEngine(SpeechBackend(RESOURCE_DIR))

# Register hotkey if keyboard library is available
if KEYBOARD_AVAILABLE:
    try:
        hotkey = app_config.get("hotkey", "ctrl+shift+r")
        hotkey_handle = keyboard.add_hotkey(hotkey, hotkey_toggle_recording, trigger_on_release=True)
        print(f"Hotkey registered: {hotkey}")
    except Exception as e:
        print(f"Failed to register hotkey: {e}")
else:
    print("Hotkey feature disabled - 'keyboard' module not installed")
    print("To enable hotkeys, run: pip install keyboard")

refresh_controls()
root.protocol("WM_DELETE_WINDOW", close_app)
root.bind("<Configure>", update_window_layout, add="+")
root.after(50, process_events)
root.mainloop()
