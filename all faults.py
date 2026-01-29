import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import pandapower.shortcircuit as sc


# --- INIT NETWORK ---   
net = pp.create_empty_network()

# 2) register custom transformer type
# Register LV distribution transformer type
pp.create_std_type(net, {
    "sn_mva": 0.4,
    "vn_hv_kv": 33,
    "vn_lv_kv": 0.415,     # use 0.415 kV, standard value
    "vk_percent": 6,
    "vkr_percent": 0.5,
    "pfe_kw": 1.2,
    "i0_percent": 0.3,
    "shift_degree": 30
}, name="0.4 MVA 33/0.415 kV", element="trafo")

pp.create_std_type(
    net,
    {
        "sn_mva": 100,
        "vn_hv_kv": 132,
        "vn_lv_kv": 33,
        "vk_percent": 12,
        "vkr_percent": 0.3,
        "pfe_kw": 50,
        "i0_percent": 0.1,
        "shift_degree": 30
    },
    name="100 MVA 132/33 kV",
    element="trafo"
)

# 3) create two buses
bus_hv = pp.create_bus(net, vn_kv=132)
bus_mv = pp.create_bus(net, vn_kv=33)

# 4) USE the transformer type
pp.create_transformer(net, hv_bus=bus_hv, lv_bus=bus_mv, std_type="100 MVA 132/33 kV")

# 5) test power flow
pp.create_ext_grid(net, bus_hv)

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

def run_powerflow_safe(net):
    """
    Tries multiple solvers and returns convergence status.
    """
    try:
        pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-6)
        return True
    except LoadflowNotConverged:
        pass

    try:
        pp.runpp(net, algorithm="iwamoto_nr", max_iteration=50)
        return True
    except LoadflowNotConverged:
        pass

    try:
        pp.runpp(net, algorithm="bfsw", max_iteration=100)
        return True
    except LoadflowNotConverged:
        return False


print(net.std_types["trafo"].keys())       # shows available types
print(net.res_trafo)                       # results


# ============================================================
# 1) HIGH VOLTAGE (HV) SUBSTATION SECTION
# ============================================================

# HV buses (132 kV level)
b_hv1 = pp.create_bus(net, vn_kv=132, name="HV Bus 1")
b_hv2 = pp.create_bus(net, vn_kv=132, name="HV Bus 2")   # ring/tie backup

# External grid connection at HV bus
pp.create_ext_grid(net, b_hv1, s_sc_max_mva=1000, s_sc_min_mva=500, name="Grid Connection")

# HV ring connection line (transmission-style)
pp.create_line_from_parameters(net, b_hv1, b_hv2, length_km=10, r_ohm_per_km=0.05,
                               x_ohm_per_km=0.25, c_nf_per_km=10, max_i_ka=0.8, name="HV Line 1-2")

# ============================================================
# 2) HV → MV TRANSFORMATION
# ============================================================

# MV buses (33 kV)
b_mv1 = pp.create_bus(net, vn_kv=33, name="Primary MV Bus")
b_mv2 = pp.create_bus(net, vn_kv=33, name="Secondary MV Bus")

# Transformer HV→MV
pp.create_transformer(net, hv_bus=b_hv1, lv_bus=b_mv1, std_type="100 MVA 132/33 kV",
                      name="T1 HV-MV")

# A second MV bus via short section or switching
pp.create_line_from_parameters(net, b_mv1, b_mv2, length_km=1, r_ohm_per_km=0.1,
                               x_ohm_per_km=0.25, c_nf_per_km=10, max_i_ka=0.6, name="MV Bus Tie")

# ============================================================
# 3) MV FEEDERS WITH TIE SWITCHES (33 kV → feeders)
# ============================================================

# Create 10 MV buses (feeding several LV transformers later)
mv_buses = [pp.create_bus(net, vn_kv=33, name=f"MV Feeder Bus {i}") for i in range(3,13)]

# Connect MV feeders radially from MV bus 1
for i, bus in enumerate(mv_buses):
    prev = b_mv1 if i == 0 else mv_buses[i-1]
    line = pp.create_line_from_parameters(net, prev, bus, length_km=2,
                                          r_ohm_per_km=0.1, x_ohm_per_km=0.25,
                                          c_nf_per_km=10, max_i_ka=0.6,
                                          name=f"MV Line {prev}-{bus}")

# Tie switch between MV Feeder Bus 5 and MV Feeder Bus 9 for reliability loop
pp.create_switch(net, mv_buses[4], mv_buses[8], et="b", closed=False, name="MV Tie Switch")

# ============================================================
# 4) MV → LV TRANSFORMATION + LV FEEDERS
# ============================================================

# For simplicity: 12 LV buses connected 1-per-MV-bus
lv_buses = []
for idx, mvb in enumerate(mv_buses):
    blv = pp.create_bus(net, vn_kv=0.4, name=f"LV Bus {idx+1}")
    lv_buses.append(blv)
    pp.create_transformer(net, mvb, blv, name=f"TRF_{idx+1}",
                          std_type="0.4 MVA 33/0.415 kV")

# LV feeder connections (radial)
for i in range(len(lv_buses)-1):
    pp.create_line_from_parameters(net, lv_buses[i], lv_buses[i+1],
                                   length_km=0.3, r_ohm_per_km=0.5, x_ohm_per_km=0.2,
                                   c_nf_per_km=0, max_i_ka=0.4,
                                   name=f"LV Line {i+1}-{i+2}")

# ============================================================
# 5) LOADS (spread across LV buses)
# ============================================================

for i, blv in enumerate(lv_buses):
    pp.create_load(net, blv, p_mw=0.2 + 0.05*i, q_mvar=0.05, name=f"Load_{i+1}")
    
# ============================================================
# 6) FAULT VARIANTS 
# ============================================================
#feeder outage

def FO_H1(net):  # head section down
    net.line.in_service.iloc[0] = False

def FO_M1(net):  # middle section down
    mid = len(net.line)//2
    net.line.in_service.iloc[mid] = False

def FO_T1(net):  # tail section down
    net.line.in_service.iloc[-2] = False

def FO_D1(net):  # distributed outage (two nonadjacent)
    net.line.in_service.iloc[1] = False
    net.line.in_service.iloc[-3] = False

#islanding
    
def IS_S1(net):
    net.switch.closed.iloc[0] = False

def IS_S2(net):
    net.switch.closed.iloc[1] = False

def IS_S3(net): # deep island — still connected to HV slack elsewhere
    net.transformer.in_service.iloc[0] = False
    net.switch.closed.iloc[1] = True   # alternate source maintained

#loss of redundancy
    
def LR_R1(net): net.switch.closed.iloc[0] = False
def LR_R2(net): net.switch.closed.iloc[1] = False
def LR_R3(net): 
    net.switch.closed.iloc[0] = False
    net.switch.closed.iloc[1] = False

#looping
    
def LP_T1(net): net.switch.closed.iloc[0] = True
def LP_T2(net): net.switch.closed.iloc[1] = True
def LP_T3(net): net.switch.closed[:] = True   # fully looped MV

#misoperation

def SM_OPEN_F1(net): net.line.in_service.iloc[2] = False
def SM_OPEN_F2(net): net.line.in_service.iloc[3] = False
def SM_OPEN_L1(net): net.load.p_mw.iloc[0] = 0   # dropped load ≈ misreport

#wrong switching sequence

def WS_S1(net):
    net.switch.closed.iloc[0] = False
    net.switch.closed.iloc[1] = True
    net.switch.closed.iloc[0] = True

def WS_S2(net):
    net.switch.closed.iloc[1] = False
    net.switch.closed.iloc[0] = True
    net.switch.closed.iloc[1] = True

#open circuit
    
def OC_U1(net): net.line.in_service.iloc[1] = False
def OC_U2(net): net.line.in_service.iloc[-1] = False
def OC_U3(net): 
    net.line.in_service.iloc[1] = False
    net.switch.closed.iloc[0] = True   # ensures convergence

#high impedance fault
    
def HIF_H200(net): net.line.r_ohm_per_km *= 1.2
def HIF_H400(net): net.line.r_ohm_per_km *= 1.4
def HIF_H800(net): net.line.r_ohm_per_km *= 1.8   # strong

#voltage sag/swell

def SAG_5(net): net.gen.vm_pu.iloc[0] = 0.95
def SAG_10(net): net.gen.vm_pu.iloc[0] = 0.90
def SAG_20(net): net.gen.vm_pu.iloc[0] = 0.80     # mild sag or DSTATCOM event

def SWE_5(net): net.gen.vm_pu.iloc[0] = 1.05
def SWE_10(net): net.gen.vm_pu.iloc[0] = 1.10

scenario_variants = {
    "FO_H1": FO_H1, "FO_M1": FO_M1, "FO_T1": FO_T1, "FO_D1": FO_D1,
    "IS_S1": IS_S1, "IS_S2": IS_S2, "IS_S3": IS_S3,
    "LR_R1": LR_R1, "LR_R2": LR_R2, "LR_R3": LR_R3,
    "LP_T1": LP_T1, "LP_T2": LP_T2, "LP_T3": LP_T3,
    "SM_OPEN_F1": SM_OPEN_F1, "SM_OPEN_F2": SM_OPEN_F2, "SM_OPEN_L1": SM_OPEN_L1,
    "WS_S1": WS_S1, "WS_S2": WS_S2,
    "OC_U1": OC_U1, "OC_U2": OC_U2, "OC_U3": OC_U3,
    "HIF_H200": HIF_H200, "HIF_H400": HIF_H400, "HIF_H800": HIF_H800,
    "SAG_5": SAG_5, "SAG_10": SAG_10, "SAG_20": SAG_20,
    "SWE_5": SWE_5, "SWE_10": SWE_10
}


# ============================================================
# DONE: ~24 buses total = 2 HV + 2 MV + 10 MV feeders + 12 LV
# ============================================================

print(f"Total buses created: {len(net.bus)}")

# choose the line index
line_index = 5
net.line.at[line_index, "in_service"] = False

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

def run_powerflow_safe(net):
    """
    Tries multiple solvers and returns convergence status.
    """
    try:
        pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-6)
        return True
    except LoadflowNotConverged:
        pass

    try:
        pp.runpp(net, algorithm="iwamoto_nr", max_iteration=50)
        return True
    except LoadflowNotConverged:
        pass

    try:
        pp.runpp(net, algorithm="bfsw", max_iteration=100)
        return True
    except LoadflowNotConverged:
        return False

print("Feeder outage: Voltages\n", net.res_bus.vm_pu)

