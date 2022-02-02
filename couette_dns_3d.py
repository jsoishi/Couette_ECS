import time
from configparser import ConfigParser
from pathlib import Path
import sys
import logging
import numpy as np

import dedalus.public as d3

logger = logging.getLogger(__name__)
debug = False

# Parsing .cfg passed
config_file = Path(sys.argv[-1])
logger.info("Running with config file {}".format(str(config_file)))

# Setting global params
runconfig = ConfigParser()
runconfig.read(str(config_file))
datadir = Path("couette_runs") / config_file.stem

# Params
timestepper = d3.RK222
params = runconfig['params']
nx = params.getint('nx')
ny = params.getint('ny')
nz = params.getint('nz')
Lx = params.getfloat('Lx')
Ly = params.getfloat('Ly')
Lz = 2 # fixed by problem definition

ampl = params.getfloat('ampl')
Re = params.getfloat('Re')

run_params = runconfig['run']
restart = run_params.get('restart_file')
stop_wall_time = run_params.getfloat('stop_wall_time')
stop_sim_time = run_params.getfloat('stop_sim_time')
stop_iteration = run_params.getint('stop_iteration')
sim_dt = run_params.getfloat('dt')

# Create bases and domain
start_init_time = time.time()
dealias = 3/2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias = dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias = dealias)
zbasis = d3.ChebyshevT(coords['z'], size=nz, bounds = (-1,1), dealias = dealias)
x = xbasis.local_grid(1)
y = xbasis.local_grid(1)
z = zbasis.local_grid(1)

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'x'),'y'), 'z')

# Fields
ba = (xbasis,ybasis,zbasis)
ba_p = (xbasis,ybasis)

p = dist.Field(name='p', bases=ba)
u = dist.VectorField(coords, name='u', bases=ba)
tau1u = dist.VectorField(coords, name='tau1u', bases=ba_p)
tau2u = dist.VectorField(coords, name='tau2u', bases=ba_p)

# NCC
U = dist.VectorField(coords, name='U', bases=(xbasis,ybasis,zbasis))
U['g'][0] = z

ex = dist.VectorField(coords, name='ex')
ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ey['g'][1] = 1
ez['g'][2] = 1

lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
#grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction

if dist.comm.rank == 0:
    if not datadir.exists():
        datadir.mkdir()

problem = d3.IVP([p, u, tau1u, tau2u], namespace=locals())

problem.add_equation("div(u) + dot(lift(tau2u,-1),ez) = 0")
problem.add_equation("dt(u) - lap(u)/Re + grad(p) + lift(tau2u,-2) + lift(tau1u,-1) = -dot(u,grad(u))")
problem.add_equation("dot(ex,u)(z=-1) = -1")
problem.add_equation("dot(ex,u)(z=1) = 1")
problem.add_equation("dot(ey,u)(z=-1) = 0")
problem.add_equation("dot(ey,u)(z=1) = 0")
problem.add_equation("dot(ez,u)(z=-1) = 0")
problem.add_equation("dot(ez,u)(z=1) = 0", condition="nx != 0 or ny != 0")
problem.add_equation("p(z=1) = 0", condition="nx == 0 and ny == 0") # Pressure gauge

solver = problem.build_solver(timestepper)
logger.info("Solver built")

solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Initial conditions
A = dist.VectorField(coords, name='A', bases=ba)
A.fill_random('g', seed=42, distribution='normal')
A.low_pass_filter(scales=(0.5, 0.5, 0.5))
A['g'] *= Lz**2*(z+1)/Lz * (1 - (z+1)/Lz) # Damp noise at walls

# no curl...don't worry about div(u) for now...
up = A #d3.curl(A).evaluate()
up.set_scales(1, keep_data=True)
u['g'] = 1e-3*up['g'] + U['g']

KE = 0.5 * d3.DotProduct(u,u)
KE.name = 'KE'
u_pert = u - U
KE_pert = 0.5 * d3.DotProduct(u_pert,u_pert)

check = solver.evaluator.add_file_handler(datadir / Path('checkpoints'), iter=10, max_writes=1, virtual_file=True)
check.add_tasks(solver.state)
# check_c = solver.evaluator.add_file_handler(datadir / Path('checkpoints_c'),iter=500,max_writes=100)
# check_c.add_tasks(solver.state, layout='c')

timeseries = solver.evaluator.add_file_handler(datadir / Path('timeseries'), iter=100)
timeseries.add_task(integ(KE), name='KE')
timeseries.add_task(integ(KE_pert), name = 'KE_pert')



flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(KE, name='KE')
flow.add_property(d3.div(u), name='div_u')
flow.add_property(KE_pert, name = 'KE_pert')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))

try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.proceed:
        #dt = CFL.compute_dt()
        solver.step(sim_dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, sim_dt))
            logger.info('Max KE = %e; Max div(u) = %e' %(flow.max('KE'), flow.max('div_u')))
            logger.info('Max perturbation KE = %e' %flow.max('KE_pert'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise

finally:
    end_run_time = time.time()

    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*dist.comm_cart.size))
    
