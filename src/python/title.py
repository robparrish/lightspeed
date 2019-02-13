import time
import os
from . import lightspeed as pls

_title_time = None

def title_header(
    ):

    global _title_time 
    _title_time = timings_header()

    fun = """=======================================================
   _     _       _     _                           _ 
  | |   (_)     | |   | |                         | |
  | |    _  __ _| |__ | |_ ___ _ __   ___  ___  __| |
  | |   | |/ _` | '_ \| __/ __| '_ \ / _ \/ _ \/ _` |
  | |___| | (_| | | | | |_\__ \ |_) |  __/  __/ (_| |
  |_____/_|\__, |_| |_|\__|___/ .__/ \___|\___|\__,_|
            __/ |             | |                    
           |___/              |_|                    
  
  A Domain-Specific Language for Electronic Structure
  
=======================================================
"""

    fun += '\n'
    fun += 'GIT SHA:    %s\n' % (pls.Config.git_sha())
    fun += 'GIT Status: %s\n' % ('Dirty' if pls.Config.git_dirty() else 'Clean')
    fun += '\n'

    fun += 'Optional Modules:\n'
    fun += '  %-8s: %r\n' % ('OpenMP', pls.Config.has_openmp())
    fun += '  %-8s: %r\n' % ('CUDA', pls.Config.has_cuda())
    fun += '  %-8s: %r\n' % ('TeraChem', pls.Config.has_terachem())
    fun += '  %-8s: %r\n' % ('LibXC', pls.Config.has_libxc())
    
    print(fun)    

def title_footer(
    ):

    if _title_time is None: raise RuntimeError("Must call title_header first")

    timings_footer(_title_time)

    print("  \"Traveling through hyperspace ain't like dusting crops, boy!\"")
    print("                             --Han Solo")
    print("")

def timings_header(
    title="Lightspeed",
    ):

    start_time = time.time()

    print('*** %s started at: %s on %s ***' % (
        title,
        time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime()),
        os.uname()[1],
        ))
    print('')

    return start_time

def timings_footer(
    start_time,
    title="Lightspeed",
    ):

    stop_time = time.time()

    s = ''
    print('*** %s stopped at: %s on %s ***' % (
        title,
        time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime()),
        os.uname()[1],
        ))
    print('*** %s runtime: %.3f [s] ***' % (
        title,
        stop_time - start_time))
    print('')
