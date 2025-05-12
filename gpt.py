#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 / Philipp Scior 2021
#
# Calculate proton QPDF with A2A method
#
import gpt as g
import os
from gpt_qpdf_utils import proton_qpdf_measurement


def create_fw_prop_QPDF(self, prop_f, W):
    g.message("Creating list of W*prop_f for all z")
    prop_list = [prop_f,]

    for z in range(1,self.zmax):
        prop_list.append(g.eval(W[z]*g.cshift(prop_f,2,z)))
    
    return prop_list 


def create_WL(self, U):
    W = []
    W.append(g.qcd.gauge.unit(U[2].grid)[0])
    for dz in range(0, self.zmax):
        W.append(g.eval(W[dz-1] * g.cshift(U[2], 2, dz)))
            
    return W


##### small dummy used for testing
grid = g.grid([16,16,16,16], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)


# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime


L = U[0].grid.fdimensions

Measurement = proton_qpdf_measurement(parameters)


prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

phases = Measurement.make_mom_phases(U[0].grid)


source_positions_sloppy = [
    [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    for j in range(jobs[job]["sloppy"])
]
source_positions_exact = [
    [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    for j in range(jobs[job]["exact"])
]



g.message("Starting Wilson loops")
W = Measurement.create_WL(U)

# exact positions
for pos in source_positions_exact:

    g.message("STARTING EXACT MEASUREMENTS")

    g.message("Generatring boosted src's")
    srcDp = Measurement.create_src(pos, trafo, U[0].grid)

    g.message("Starting prop exact")
    prop_exact_f = g.eval(prop_exact * srcDp)
    g.message("forward prop done")
    g.message(f"prop grid: {prop_exact_f.grid}")
    g.message(f"trafo grid: {trafo.grid}")

    tag = "%s/%s" % ("exact", str(pos)) 

    g.message("Starting 2pt contraction (includes sink smearing)")
    Measurement.contract_2pt(prop_exact_f, phases, trafo, tag)
    g.message("2pt contraction done")

    g.message("Create seq. backwards prop")
    prop_bw = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo)

    g.message("Create list of W * forward prop")
    prop_f = Measurement.create_fw_prop_QPDF(prop_exact_f, W)

    g.mem_report(details=False)
    g.message("Start QPDF contractions")
    Measurement.contract_QPDF(prop_f, prop_bw, phases, tag)
    g.message("PQDF done")

    if(parameters["save_propagators"]):
        Measurement.propagator_output(tag, prop_exact_f)

    del prop_exact_f
    del prop_bw

    g.message("STARTING SLOPPY MEASUREMENTS")

    g.message("Starting prop sloppy")
    prop_sloppy_f = g.eval(prop_sloppy * srcDp)
    g.message("forward prop done")

    del srcDp

    tag = "%s/%s" % ("sloppy", str(pos))



    g.message("Starting 2pt contraction (includes sink smearing)")
    Measurement.contract_2pt(prop_sloppy_f, phases, trafo, tag)
    g.message("2pt contraction done")

    g.message("Create seq. backwards prop")
    prop_bw = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_f, trafo)

    g.message("Create list of W * forward prop")
    prop_f = Measurement.create_fw_prop_QPDF(prop_sloppy_f, W)

    g.mem_report(details=False)
    g.message("Start QPDF contractions")
    Measurement.contract_QPDF(prop_f, prop_bw, phases, tag)
    g.message("PQDF done")

    if(parameters["save_propagators"]):
        Measurement.propagator_output(tag, prop_sloppy_f)

    del prop_bw
    del prop_sloppy_f
    
g.message("exact positions done")

# sloppy positions
for pos in source_positions_sloppy:

    g.message("STARTING SLOPPY MEASUREMENTS")
    tag = "%s/%s" % ("sloppy", str(pos))

    g.message("Starting DA 2pt function")

    g.message("Generatring boosted src's")
    srcDp = Measurement.create_src(pos, trafo, U[0].grid)  

    g.message("Starting prop sloppy")
    prop_sloppy_f = g.eval(prop_sloppy * srcDp)
    g.message("forward prop done")

    del srcDp

    tag = "%s/%s" % ("sloppy", str(pos))



    g.message("Starting 2pt contraction (includes sink smearing)")
    Measurement.contract_2pt(prop_sloppy_f, phases, trafo, tag)
    g.message("2pt contraction done")

    g.message("Create seq. backwards prop")
    prop_bw = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_f, trafo)

    g.message("Create list of W * forward prop")
    prop_f = Measurement.create_fw_prop_QPDF(prop_sloppy_f, W)

    g.message("Start QPDF contractions")
    Measurement.contract_QPDF(prop_f, prop_bw, phases, tag)
    g.message("PQDF done")

    if(parameters["save_propagators"]):
        Measurement.propagator_output(tag, prop_sloppy_f) 


g.message("sloppy positions done")
        
#del pin
