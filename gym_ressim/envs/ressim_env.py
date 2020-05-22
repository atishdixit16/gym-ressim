import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import functools

from gym_ressim.envs.spatial_expcov import batch_generate
from gym_ressim.envs.ressim import Grid, SaturationEquation, PressureEquation
from gym_ressim.envs.utils import quadratic_mobility, lamb_fn, f_fn, df_fn


class ResSimEnv(gym.Env):
    def __init__(self,
                action_steps=11,
                nx = 10,
                ny = 10,
                lx = 1.0,
                ly = 1.0,
                mu_w = 1.0,
                mu_o = 2.0,
                s_wir = 0.2,
                s_oir = 0.2,
                k = 1.0,
                phi = 0.1,
                dt = 1e-3,
                n_steps = 50,
                k_type='uniform',
                max_steps=5,
                state_seq_n=3):

        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.grid = Grid(nx=self.nx, ny=self.ny, lx=self.lx, ly=self.ly)  # unit square, 64x64 grid
        self.k = k*np.ones(self.grid.shape) #uniform permeability
        self.mu_w, self.mu_o = mu_w, mu_o  # viscosities
        self.s_wir, self.s_oir = s_wir, s_oir  # irreducible saturations
        self.phi = np.ones(self.grid.shape)*phi  # uniform porosity
        self.s_init = np.ones(self.grid.shape) * self.s_wir  # initial water saturation equals s_wir
        self.s_load = self.s_init
        self.p_init = np.ones(self.grid.shape)
        self.p_load = self.p_init
        self.dt = dt  # timestep
        self.n_steps = n_steps
        self.k_type = k_type
        self.state_seq_n = state_seq_n # has to be smaller than n_steps
        self.max_steps = max_steps

        self.step_no=0

        self.q = np.zeros(self.grid.shape)
        self.q[0,0]=-0.5 # producer 1 
        self.q[-1,0]=-0.5 # producer 2
        self.q[0,-1]=1.0 # injector 1

        if self.k_type=='random':
            k=batch_generate(nx=self.nx, ny=self.ny, length=1.0, sigma=1.0, lx=self.lx, ly=self.ly, sample_size=1)
            self.k = np.exp(k[0])

        # RL parameters
        high = np.array([1e5,1e5,1e5]*self.state_seq_n)
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        self.action_space = spaces.Discrete(int(action_steps)) # should be a perfect square number

        # Model definition
        self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)
    
    def step(self, action):
        
        # source term for producer 1: q[0,0]
        self.q[0,0] = ( -1 / ( self.action_space.n - 1 ) ) * action
        self.q[-1,0] = -1 - self.q[0,0] # since q[0,0] + q[-1,0] = -1
        
        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
        self.solverS = SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        state_n_gap = int(self.n_steps/self.state_seq_n)
        state_indices = []
        for i in range(self.state_seq_n):
            state_indices.append(self.n_steps-i*state_n_gap)
        
        state=[]
        reward = 0.0
        for i in range(self.n_steps):
            self.solverP.s = self.s_load
            self.solverP.step()
            # solve saturation
            self.solverS.v = self.solverP.v
            self.solverS.step_mrst(self.dt)

            self.s_load = self.solverS.s
            self.p_load = self.solverP.p

            if i+1 in state_indices:
                state.extend( [self.s_load[0,-1],self.s_load[-1,0],self.s_load[0,0]] )
            
            reward +=  -self.q[0,0] * (1 - self.s_load[0,0]) + -self.q[-1,0] * ( 1 - self.s_load[-1,0] ) 

        self.step_no += 1

        if self.step_no >= self.max_steps:
            done=True
            self.step_no=0
        else:
            done=False

        # states are represented by values of sturation and pressure at producers and injectors
        state = np.array( state )
        return state, reward, done, {}

    def get_sw(self, x_ind, y_ind):
        return self.s_load[x_ind, y_ind]

    def reset(self):

        self.q[0,0]=-0.5 # producer 1 
        self.q[-1,0]=-0.5 # producer 2
        self.q[0,-1]=1.0 # injector 1

        self.s_load = self.s_init
        if self.k_type=='random':
            k=batch_generate(nx=self.nx, ny=self.ny, length=1.0, sigma=1.0, lx=self.lx, ly=self.ly, sample_size=1)
            self.k = np.exp(k[0])

        state = np.array( [ self.s_init[0,-1],self.s_init[-1,0],self.s_init[0,0]]*self.state_seq_n )
        return state

    def render(self):
        pass

    def close(self):
        pass
