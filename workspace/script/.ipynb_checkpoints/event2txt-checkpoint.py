import time
import win32process
import win32con
from pathlib import Path
from pynput.mouse import Button, Controller

def open_program(EXE_NAME, wait=0.5):
    si = win32process.STARTUPINFO()
    si.dwFlags = win32con.STARTF_USESHOWWINDOW
    si.wShowWindow = win32con.SW_MAXIMIZE
    h_proc, h_thr, pid, tid = win32process.CreateProcess(None, EXE_NAME, None, None, False, 0, None, None, si)
    time.sleep(wait)
    return h_proc

def close_program(h_proc, wait=0.5):
    time.sleep(wait)
    win32process.TerminateProcess(h_proc, 0)
    
def click_over(x, y, wait=0.5):
    mouse = Controller()
    mouse.position = (x, y)
    mouse.press(Button.left)
    mouse.release(Button.left)
    time.sleep(wait)
    
def export_file(filename, wait):
    p = open_program("siwim_eventview.exe " + filename, wait)
    click_over(60, 30, wait)
    click_over(60, 200, wait)
    close_program(p, wait)
    
                             
if __name__ == '__main__':
    
    #root = "../data/normandie/"
    root = "../data/senlis/"
    
    for filename in Path(root).glob('**/*.event'):
        export_file(str(filename), wait=0.5)