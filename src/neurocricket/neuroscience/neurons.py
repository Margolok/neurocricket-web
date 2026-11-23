"""Physical-voltage LIF. syn and drive represent R*I, measured in volts."""
LIF_EQUATIONS = """
dv/dt = (rest - v + drive + syn)/tau_m : volt (unless refractory)
dsyn/dt = -syn/tau_syn : volt
dadapt/dt = -adapt/tau_adapt : volt
drive : volt
"""

INPUT_EQUATIONS = """
base : second
interval : second
within : second
size : integer
cursor : integer
total : integer
"""
INPUT_THRESHOLD = "(cursor < total) and (t >= base + floor(cursor/size)*interval + (cursor % size)*within)"
