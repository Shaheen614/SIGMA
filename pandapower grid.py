import pandapower as pp
import pandapower.plotting as plot
import pandapower.topology as top
import pandas as pd

# Create empty network
net = pp.create_empty_network()

# --------------------
# Create buses
# --------------------
buses = {}
for i in range(1, 13):
    buses[i] = pp.create_bus(net, vn_kv=11, name=f"Bus {i}")

# High-voltage bus for transformer
hv_bus = pp.create_bus(net, vn_kv=33, name="HV Bus")

# --------------------
# Create external grid
# --------------------
pp.create_ext_grid(net, bus=hv_bus, vm_pu=1.02, name="Grid Connection")

# --------------------
# Create transformer (33kV → 11kV)
# --------------------
pp.create_transformer_from_parameters(
    net,
    hv_bus=hv_bus,
    lv_bus=buses[1],
    sn_mva=10,
    vn_hv_kv=33,
    vn_lv_kv=11,
    vk_percent=10,
    vkr_percent=0.5,
    pfe_kw=20,
    i0_percent=0.1,
    name="Main Transformer"
)

# --------------------
# Create lines
# --------------------
line_length = 1  # km

for i in range(1, 12):
    pp.create_line_from_parameters(
        net,
        from_bus=buses[i],
        to_bus=buses[i + 1],
        length_km=line_length,
        r_ohm_per_km=0.4,
        x_ohm_per_km=0.3,
        c_nf_per_km=210,
        max_i_ka=0.4,
        name=f"Line {i}-{i+1}"
    )

# --------------------
# Create loads
# --------------------
pp.create_load(net, bus=buses[3], p_mw=0.6, q_mvar=0.2, name="Load 1")
pp.create_load(net, bus=buses[5], p_mw=0.8, q_mvar=0.3, name="Load 2")
pp.create_load(net, bus=buses[7], p_mw=0.5, q_mvar=0.15, name="Load 3")
pp.create_load(net, bus=buses[10], p_mw=1.0, q_mvar=0.4, name="Load 4")

# --------------------
# Run power flow
# --------------------
pp.runpp(net)

# --------------------
# Print results
# --------------------
print("\nBus Voltages (pu):")
print(net.res_bus.vm_pu)

print("\nLine Loadings (%):")
print(net.res_line.loading_percent)

print("\nTransformer Loading (%):")
print(net.res_trafo.loading_percent)

# Run power flow
pp.runpp(net)

print("Power flow converged:", net.converged)
