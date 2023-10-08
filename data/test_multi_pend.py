# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify
from scipy.integrate import odeint


def integrate_pendulum(n,
                       times,
                       initial_positions=135,
                       initial_velocities=0,
                       lengths=None,
                       masses=1):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model

    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass)
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')

    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u, kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)

    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.deg2rad(
        np.concatenate([
            np.broadcast_to(initial_positions, n),
            np.broadcast_to(initial_velocities, n)
        ]))

    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y, t, args):
        vals = np.concatenate((y, args))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
        return np.array(sol).T[0]

    # ODE integration
    return odeint(gradient, y0, times, args=(parameter_vals, ))


def energy(loc, vel):
    stick_mass = 1.
    stick_length = 1. / 3
    g = 9.81
    # disables division by zero warning, since I fix it with fill_diagonal
    with np.errstate(divide='ignore'):
        U = -stick_mass * stick_length * g / 2 * (5 * np.cos(loc[:, 0]) +
                                                  3 * np.cos(loc[:, 1]) +
                                                  1 * np.cos(loc[:, 2]))
        K = stick_mass * stick_length * stick_length / 6 * (
            9 * vel[:, 1] * vel[:, 0] * np.cos(loc[:, 0] - loc[:, 1]) +
            3 * vel[:, 2] * vel[:, 0] * np.cos(loc[:, 0] - loc[:, 2]) +
            3 * vel[:, 2] * vel[:, 1] * np.cos(loc[:, 1] - loc[:, 2]) +
            7 * vel[:, 0] * vel[:, 0] + 4 * vel[:, 1] * vel[:, 1] +
            1 * vel[:, 2] * vel[:, 2])

        print('U: ', U)
        print('K: ', K)
        print('energy:', U + K)

        return U, K, U + K


if __name__ == '__main__':
    t = np.linspace(0, 1, 100)
    res = integrate_pendulum(n=2, times=t)
    # get the dimension
    dim = res.shape[1] // 2
    # get the p: momentum and q: position
    vel, loc = res[:, dim:], res[:, :dim]
    vel /= 1.0  # hardcode mass to be 1
    # get the energy
    U, K, E = energy(loc, vel)
    # plot the U K E respectively
    plt.plot(t, U, label='U')
    plt.plot(t, K, label='K')
    plt.plot(t, E, label='U+K')
    plt.legend()
    # save plot
    plt.savefig('energy.png')