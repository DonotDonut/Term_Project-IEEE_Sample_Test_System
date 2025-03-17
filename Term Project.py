import numpy as np
 

def read_File(file_path, bus_start_marker, branch_start_marker, stop_marker):
    bus_section = []
    branch_section = []
    reading_bus_section = False
    reading_branch_section = False

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if reading_bus_section:
                    if stop_marker in line:
                        reading_bus_section = False
                        continue
                    bus_section.append(line.strip())
                elif reading_branch_section:
                    if stop_marker in line:
                        reading_branch_section = False
                        continue
                    branch_section.append(line.strip())
                elif bus_start_marker in line:
                    reading_bus_section = True
                elif branch_start_marker in line:
                    reading_branch_section = True
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    return bus_section, branch_section

def read_branch_section(branch_section):
    branch_data = []
    for line in branch_section:
        values = line.split()
        from_bus = int(values[0])
        to_bus = int(values[1])
        resistance = float(values[6])
        reactance = float(values[7])
        line_charging = float(values[8])
        branch_data.append((from_bus, to_bus, resistance, reactance, line_charging))
    return np.array(branch_data)

def read_bus_section(bus_section):
    busdata = []
    for line in bus_section:
        values = line.split()
        bus_number = int(values[0])
        
        # Bus Type: 3 = slack, 2 = PV, 0 = PQ
        bus_type = int(values[5])
        
        # Cpnverting to Per unit values using 100 as the Sbase
        # Real & reactive power generated  
        Pg = float(values[10]) / 100 # Real Power generated
        Qg = float(values[11]) / 100 # Reactive Power generated
        
        # Real & reactive power demand 
        Pd = float(values[8]) / 100 # Real Power demand
        Qd = float(values[9]) / 100 # Reactive Power demand
        
        # voltage per bus 
        Vmag = float(values[6]) # Voltage magnitude (in Per unit) 
        Vangle = float(values[7]) # Voltage angle (in degrees)
        
        # min & max reactive power of generator 
        Qmin = float(values[15]) / 100  
        Qmax = float(values[14]) / 100  
        
        # Initial guesses for different bus types:
        if bus_type == 0: # PQ bus
            Vmag = 1.0
            Vangle = 0.0
        elif bus_type == 2: # PV bus
            Vangle = 0.0
            
        busdata.append((bus_number, bus_type, Pg, Qg, Pd, Qd, Vmag, Vangle, Qmin, Qmax))
    return np.array(busdata)

def create_Ybus(linedata):
  
    R = linedata[:, 2]  # resistance  
    X = linedata[:, 3]  # reactance  
    B_line = 1j * linedata[:, 4]  # line charging susceptance

    Z = R + 1j * X # Line impedance
    Y_line = 1.0 / Z # Line admittance

    nbus = int(np.max(linedata[:, :2])) 
    
    Ybus = np.zeros((nbus, nbus), dtype=complex)  # Initialize Ybus matrix

    for k in range(len(linedata)):
        i = int(linedata[k, 0]) - 1  
        j = int(linedata[k, 1]) - 1
        
        # Off-diagonal elements (mutual admittance)
        Ybus[i, j] -= Y_line[k]
        Ybus[j, i] = Ybus[i, j]
        
        # Diagonal elements (sum of connected admittances)
        Ybus[i, i] += Y_line[k] + B_line[k]
        Ybus[j, j] += Y_line[k] + B_line[k]
    
    return Ybus

def cal_powerflow(nbus, Vangle, Vmag, Ymag, Yangle): 
  
    P_load_flow = np.zeros(nbus)
    Q_load_flow = np.zeros(nbus)
    
    for i in range(nbus):
        for j in range(nbus):
            P_load_flow[i] += Vmag[i] * Vmag[j] * Ymag[i, j] * np.cos(Vangle[i] - Vangle[j] - Yangle[i, j]) 
            Q_load_flow[i] += Vmag[i] * Vmag[j] * Ymag[i, j] * np.sin(Vangle[i] - Vangle[j] - Yangle[i, j]) 
           
    return P_load_flow, Q_load_flow

def mismatch_power(P_net, Q_net, P_flow_load, Q_flow_load, PQ_index, nPQ, slack_index):
    
    # Pnet = Pg - Pd 
    # P_flow_load = Pg - Pd 
    # 0 = Pg - Pd - P_flow_load // goes w/ Q 
    # if use 0 = Pg - Pd - P_flow_load, can't calculate mismatch 
    
    dP =  P_net - P_flow_load  
    dQ =  Q_net - Q_flow_load
    
    # Q is used to solve for Vmag  
    mismatch_Q = np.zeros(nPQ)
    for i in range(nPQ):
        n = PQ_index[i]
        mismatch_Q[i] = dQ[n]
        
    # P mismatch is used for Vangle 
    mismatch_P = np.delete(dP, slack_index)  # Exclude slack bus
    mismatch_PQ = np.concatenate((mismatch_P, mismatch_Q))
    
    return mismatch_PQ 


def Jacobian(Vmag, Vangle, nbus, PQ_buses, nPQ, Ymag, Yangle, non_slack):
    # J1: dP/d(Vangle)
    J1 = np.zeros((nbus, nbus))
    for k in range(nbus):
        for n in range(nbus):
            if n != k:
                # J1_kn (off-diagonal)
                sin_term = np.sin(Yangle[k, n] + Vangle[n] - Vangle[k])
                J1[k, n] = -Vmag[k] * Vmag[n] * Ymag[k, n] * sin_term
            else:
                # J1_kk (diagonal)
                for m in range(nbus):
                    if m != k:
                        sin_term = np.sin(Yangle[k, m] + Vangle[m] - Vangle[k])
                        J1[k, k] += Vmag[k] * Vmag[m] * Ymag[k, m] * sin_term
    J11 = J1[np.ix_(non_slack, non_slack)]
    
    # J2: dP/d(Vmag)
    J2 = np.zeros((nbus, nbus))
    for k in range(nbus):
        for n in range(nbus):
            angle_term = Yangle[k, n] + Vangle[n] - Vangle[k]
            if n != k:
                # J2_kn (off-diagonal)
                J2[k, n] = Vmag[k] * Ymag[k, n] * np.cos(angle_term)
            else:
                # J2_kk (diagonal)
                J2[k, k] = Vmag[k] * Ymag[k, k] * np.cos(Yangle[k, k])
                for m in range(nbus):
                    if m != k:
                        J2[k, k] += Vmag[m] * Ymag[k, m] * np.cos(Yangle[k, m] + Vangle[m] - Vangle[k])
    J22 = J2[np.ix_(non_slack, PQ_buses)]
    

    # J3: dQ/d(Vangle)
    J3 = np.zeros((nbus, nbus))
    for k in range(nbus):
        for n in range(nbus):
            angle_term = Yangle[k, n] + Vangle[n] - Vangle[k]
            if n != k:
                # J3_kn (off-diagonal)
                J3[k, n] = -Vmag[k] * Vmag[n] * Ymag[k, n] * np.cos(angle_term)
            else:
                # J3_kk (diagonal)
                for m in range(nbus):
                    if m != k:
                        J3[k, k] += Vmag[k] * Vmag[m] * Ymag[k, m] * np.cos(Yangle[k, m] + Vangle[m] - Vangle[k])
    J33 = J3[np.ix_(PQ_buses, non_slack)]
    

    # J4: dQ/d(Vmag)
    J4 = np.zeros((nbus, nbus))
    for k in range(nbus):
        for n in range(nbus):
            if n == k:
                # J4_kk (diagonal)
                J4[k, k] = -2 * Vmag[k] * Ymag[k, k] * np.sin(Yangle[k, k])
                for m in range(nbus):
                    if m != k:
                    # J4_kn (non-diagonal)
                        J4[k, k] -= Vmag[m] * Ymag[k, m] * np.sin(Yangle[k, m] + Vangle[m] - Vangle[k])
    J44 = J4[np.ix_(PQ_buses, PQ_buses)]
    

    # Assemble the full Jacobian
    J = np.block([[J11, J22], [J33, J44]])
    
    return J


   

# Test Case File Path
TestCase14 = 'Data/ieee14cdf.txt'
TestCase30 = 'Data/ieee30cdf.txt'
TestCase57 = 'Data/ieee57cdf.txt'
TestCase118 = 'Data/ieee118cdf.txt'  
TestCase300 = 'Data/ieee300cdf.txt'

# Markers for file reading 
bus_start_marker = "BUS DATA FOLLOWS"
branch_start_marker = "BRANCH DATA FOLLOWS"
stop_marker = "-999"
    
# Read the file
bus_section, branch_section = read_File(TestCase14, bus_start_marker, branch_start_marker, stop_marker)    

# Process branch and bus data
linedata = read_branch_section(branch_section)
busdata = read_bus_section(bus_section)

'''  
print(f"\nBus Data")
for bus in busdata:
    print(f"Bus#: {bus[0]}, Type: {bus[1]}, PG: {bus[2]:.3f}, QG: {bus[3]:.3f}, Pd: {bus[4]:.3f}, Qd: {bus[5]:.3f}, |V|: {bus[6]:.3f}, Delta: {bus[7]:.3f}, Pmin: {bus[8]:.3f}, Pmax: {bus[9]:.3f}")
#'''

'''  
print(f"\nBranch Data")
for branch in linedata:
    print(f"From Bus: {branch[0]}, To Bus: {branch[1]}, Resistance: {branch[2]}, Reactance: {branch[3]}, Line Charging: {branch[4]}")
#'''

# Extract bus quantities from busdata
bus_type = busdata[:, 1]
Pg = busdata[:, 2].copy()
Qg = busdata[:, 3].copy()
Pd = busdata[:, 4].copy()
Qd = busdata[:, 5].copy()
Qmin = busdata[:, 8].copy()
Qmax = busdata[:, 9].copy()
Vmag = busdata[:, 6].copy() 
Vangle = busdata[:, 7].copy()

# Extract bus quantities from linedata
From = linedata[:, 0]
To = linedata[:, 1]
R = linedata[:, 2]
X = linedata[:, 3]

# Create the Y-bus admittance matrix
Ybus = create_Ybus(linedata)
#print(Ybus)
Ymag = np.abs(Ybus)
Yangle = np.angle(Ybus)

nbus = len(busdata)
nline = len(linedata)

# Net Power 
P_net = Pg - Pd
Q_net = Qg - Qd

# Determine indices for bus types:
# Slack bus: type 3, PQ bus: type 0, PV bus: type 2
slack_index = np.where(busdata[:, 1] == 3)[0]
PQ_buses = np.where(busdata[:, 1] == 0)[0]
PV_buses = np.where(busdata[:, 1] == 2)[0]
non_slack = np.where(busdata[:, 1] != 3)[0]

# gets the total number of each bus type 
nslack = len(slack_index)
nPQ = len(PQ_buses)
nPV = len(PV_buses)

# Newton-Raphson iteration parameters
Error_measure = 1.0  
tolerance = 1e-5 
iter_count = 0

V = Vmag * (np.cos(Vangle) + 1j * np.sin(Vangle))

# Newton-Raphson Iteration Loop
while Error_measure > tolerance:
    
    P_flow_load, Q_flow_load = cal_powerflow(nbus, Vangle, Vmag, Ymag, Yangle)
    
    
    for i in non_slack: 
        
        # check if Q_flow_load is w/in range  
        for i in non_slack: 
        
            if i in PV_buses: 
                if Qmax[i] != 0: 
                    if Q_flow_load[i] > Qmax[i]: 
                        Q_flow_load[i] = Qmax[i]
                    elif Q_flow_load[i] < Qmin[i]:
                        Q_flow_load[i] = Qmin[i]
                    else: 
                        busdata[i, 1] = 2  # Restore PV bus status
                        Vmag[i] = busdata[i, 6]  # Reset voltage magnitude

    
    mismatch_PQ = mismatch_power(P_net, Q_net, P_flow_load, Q_flow_load, PQ_buses, nPQ, slack_index)
    
    J = Jacobian(Vmag, Vangle, nbus, PQ_buses, nPQ, Ymag, Yangle, non_slack)
    
    # Solve for correction vector using inverse of J (for small systems)
    
    mismatch_X = np.linalg.solve(J, mismatch_PQ) 
    
    dTeth = mismatch_X[:nbus - 1]
    dV = mismatch_X[nbus - 1:]
    
    # Update angles for non-slack buses 
    Vangle[np.arange(nbus) != slack_index] += dTeth

    
    # Update voltage magnitudes for PQ buses
    for i in range(nPQ):
        n = PQ_buses[i]
        Vmag[n] += dV[i]
    
    # updates values 
    V = Vmag * (np.cos(Vangle) + 1j * np.sin(Vangle))
    
    Error_measure = np.max(np.abs(mismatch_PQ))
    iter_count += 1
    

print("Number of iterations:", iter_count)

# Final Bus Voltages calculation 
print("\nBus Voltages:")
header = f"{'Bus#':<5} {'|V| (p.u.)':<12} {'Angle (deg)':<14}"
print(header)
print("-" * len(header))
for i in range(nbus):
    angle_deg = np.degrees(Vangle[i])
    print(f"{i+1:<5} {Vmag[i]:<12.3f} {angle_deg:<14.3f}")
    

# Final flow load calculation  
print("\nFlow Load:")
header1 = f"{' ':<5} {'P flow':<12} {'Q flow':<12} {'P flow':<12} {'Q flow':<12} {'P flow':<12} {'Q flow':<12}"
header2 = f"{'Bus#':<5} {'Load (P.U)':<12} {'Load (P.U)':<12} {'Load (MW)':<12} {'Load (MVAR)':<12} {'Load (P.U)':<12} {'Load (P.U)':<12}"
print(header1)
print(header2)
print("-" * len(header1))

for i in range(nbus):
    print(f"{i+1:<5} {P_flow_load[i]:<12.3f} {Q_flow_load[i]:<12.3f} {P_flow_load[i]*100:<12.1f} {Q_flow_load[i]*100:<14.1f} {P_net[i]:<12.3f} {Q_net[i]:<14.3f}")
#'''
    
    
# Final Branch Current Flow Calculation
print("\nBranch Current Flows:")
branch_header = f"{'Branch':<8} {'From Bus':<10} {'To Bus':<8} {'I (mag, p.u.)':<16} {'I (angle, deg)':<16}"
print(branch_header)
print("-" * len(branch_header))
for k in range(nline):
    i = int(From[k]) - 1
    j = int(To[k]) - 1
    I_branch = (V[i] - V[j]) / (R[k] + 1j*X[k])
    I_mag = np.abs(I_branch)
    I_angle = np.degrees(np.angle(I_branch))
    print(f"{k+1:<8} {i+1:<10} {j+1:<8} {I_mag:<16.3f} {I_angle:<16.3f}")
#''' 
# Generator Output Calculation
print("\nGenerator Output:")
header_gen = f"{'Bus#':<5} {'P gen (MW)':<12} {'Q gen (MVAR)':<12}"
print(header_gen)
print("-" * len(header_gen))

for i in range(nbus):
    P_gen = Pd[i] + P_flow_load[i]  
    Q_gen = Qd[i] + Q_flow_load[i]  
    print(f"{i+1:<5} {P_gen*100:<12.1f} {Q_gen*100:<12.1f}")
