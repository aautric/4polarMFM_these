# hb_any.py <script.py> <args...>: runs the script (SGD_batch.py, SGD_precision_claude.py) with a heartbeat: the time
# of the last LM step written to $HEARTBEAT at most every 10 s (for a watchdog), native stacks possible
# (py-spy dump --native), Python stacks on SIGUSR1
import sys, os, time, ctypes, runpy, signal, faulthandler
ctypes.CDLL(None).prctl(0x59616d61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER, PR_SET_PTRACER_ANY
faulthandler.register(signal.SIGUSR1, all_threads=True)
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root of the repo (tools/hb_any.py); was a fixed path to main
sys.path.insert(0, R)
import stream_fit
HB, last = os.environ['HEARTBEAT'], [0.]
def beat(s):
    open(HB, 'w').write(f'{time.time():.0f} {s}\n'); last[0] = time.time()
orig = stream_fit.stream
def hb_stream(n, slots, init, refill, step, *a, **k):
    cnt = [0]
    def step2(state, ids, pool):
        if cnt[0] == 0: beat('before first step')
        s = step(state, ids, pool); cnt[0] += 1
        if time.time() - last[0] > 10: beat(f'step {cnt[0]}')
        return s
    return orig(n, slots, init, refill, step2, *a, **k)
stream_fit.stream = hb_stream
beat('start')
script = os.path.join(R, sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name='__main__')
