import pybamm

model = pybamm.equivalent_circuit.Thevenin()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()