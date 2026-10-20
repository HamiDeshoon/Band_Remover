# gui/__init__.py
import sys
import os

if sys.platform == "win32":
    import ctypes
    # Ensure Windows System32 icuuc.dll is preferred over older Anaconda DLLs
    sys_icu = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icuuc.dll")
    if os.path.exists(sys_icu):
        try:
            ctypes.windll.kernel32.LoadLibraryW(sys_icu)
        except Exception:
            pass
