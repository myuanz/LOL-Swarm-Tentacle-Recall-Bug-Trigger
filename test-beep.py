# # %%
# 
# frequency = 500  # 设置频率(Hz)
# duration = 1000  # 设置持续时间(毫秒)
# winsound.Beep(frequency, duration)
# %%
import win32gui
import win32con
import win32api
import sys
import ctypes
import ctypes
from ctypes import wintypes
import winsound
from ctypes import wintypes, windll
from datetime import datetime, timedelta
from threading import Thread, Timer

def is_caps_lock_on():
    return win32api.GetKeyState(win32con.VK_CAPITAL) & 0x0001 != 0

def toggle_caps_lock():
    win32api.keybd_event(win32con.VK_CAPITAL, 0, 0, 0)
    win32api.keybd_event(win32con.VK_CAPITAL, 0, win32con.KEYEVENTF_KEYUP, 0)

user32 = ctypes.windll.user32
HRESULT = ctypes.c_long

uxtheme = ctypes.windll.uxtheme
SetWindowTheme = uxtheme.SetWindowTheme
SetWindowTheme.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR]
SetWindowTheme.restype = HRESULT

ctypes.windll.comctl32.InitCommonControlsEx()
# 定义窗口类名和标题
WINDOW_CLASS = "HotkeyDemoClass"
WINDOW_TITLE = "热键监视器"

# 定义按钮ID
ID_START = 1001
ID_STOP = 1002

# 全局变量
is_monitoring = False
hotkey_id = 1

records: list[datetime] = []
KEYEVENTF_EXTENDEDKEY = 0x1
KEYEVENTF_KEYUP = 0x2
VK_CAPITAL = 0x14


def handle_timer():
    print(f"[{datetime.now()}] 定时器被触发!")
    winsound.Beep(1000, 300)

def handle_hotkey():
    now = datetime.now()
    print(f"[{now}] 热键被触发!")
    
    records.append(now)
    winsound.Beep( min(len(records), 10) * 100 + 500, 500)

def thread_proc():
    提前时间 = timedelta(seconds=0.5)
    next_beeptime = datetime.now()
    while not thread_need_exit:
        if len(records) >= 2 and datetime.now() >= next_beeptime:
            delta = records[-1] - records[-2]
            next_beeptime = datetime.now() + delta - 提前时间
            print(f"[{datetime.now()}] 下一次响铃时间: {next_beeptime}")
            winsound.Beep(1000, 300)
            win32api.Sleep(int(提前时间.total_seconds() * 1000))
        win32api.Sleep(50)
        

last_caps_lock_state = is_caps_lock_on()
thread_need_exit = False

thread = Thread(target=thread_proc)
thread.start()

try:
    while True:
        caps_lock_state = is_caps_lock_on()
        if caps_lock_state != last_caps_lock_state:
            last_caps_lock_state = caps_lock_state
            handle_hotkey()
        win32api.Sleep(50)
except KeyboardInterrupt:
    thread_need_exit = True
    thread.join()

exit()
def register_hotkey(hwnd):
    global is_monitoring
    if not is_monitoring:
        try:
            win32gui.RegisterHotKey(hwnd, hotkey_id, win32con.MOD_CONTROL, ord('B'))
        except:
            print("热键注册失败")
        else:
            print("热键注册成功")
            is_monitoring = True
            win32gui.EnableWindow(win32gui.GetDlgItem(hwnd, ID_START), False)
            win32gui.EnableWindow(win32gui.GetDlgItem(hwnd, ID_STOP), True)

def unregister_hotkey(hwnd):
    global is_monitoring
    if is_monitoring:
        user32.UnregisterHotKey(hwnd, hotkey_id)
        print("热键注销成功")
        is_monitoring = False
        win32gui.EnableWindow(win32gui.GetDlgItem(hwnd, ID_START), True)
        win32gui.EnableWindow(win32gui.GetDlgItem(hwnd, ID_STOP), False)

def create_button(hwnd, text, x, y, width, height, id):
    return win32gui.CreateWindow(
        "BUTTON",
        text,
        win32con.WS_TABSTOP | win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.BS_DEFPUSHBUTTON,
        x, y, width, height,
        hwnd,
        id,
        win32gui.GetWindowLong(hwnd, win32con.GWL_HINSTANCE),
        None
    )

def win_proc(hwnd, msg, wparam, lparam):
    if msg == win32con.WM_DESTROY:
        unregister_hotkey(hwnd)
        win32gui.PostQuitMessage(0)
    elif msg == win32con.WM_HOTKEY:
        handle_hotkey()
    elif msg == win32con.WM_COMMAND:
        if wparam == ID_START:
            register_hotkey(hwnd)
        elif wparam == ID_STOP:
            unregister_hotkey(hwnd)
    else:
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
    return 0

def create_window():
    wc = win32gui.WNDCLASS()
    wc.lpfnWndProc = win_proc
    wc.lpszClassName = WINDOW_CLASS
    wc.hbrBackground = win32con.COLOR_BTNFACE + 1
    win32gui.RegisterClass(wc)

    hwnd = win32gui.CreateWindow(
        WINDOW_CLASS,
        WINDOW_TITLE,
        win32con.WS_OVERLAPPEDWINDOW,
        100, 100, 300, 200,
        0, 0, 0, None
    )
    if SetWindowTheme:
        SetWindowTheme(hwnd, "Explorer", None)

    start_button = create_button(hwnd, "开始监视", 50, 50, 100, 30, ID_START)
    stop_button = create_button(hwnd, "停止监视", 150, 50, 100, 30, ID_STOP)
    if SetWindowTheme:
        SetWindowTheme(start_button, "Explorer", None)

    win32gui.EnableWindow(stop_button, False)

    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    win32gui.UpdateWindow(hwnd)

    return hwnd

if __name__ == '__main__':
    hwnd = create_window()
    win32gui.PumpMessages()
    # msg = ctypes.wintypes.MSG()
    # while win32gui.GetMessage(ctypes.byref(msg), 0, 0, 0) != 0:
    #     win32gui.TranslateMessage(ctypes.byref(msg))
    #     win32gui.DispatchMessage(ctypes.byref(msg))
# %%
