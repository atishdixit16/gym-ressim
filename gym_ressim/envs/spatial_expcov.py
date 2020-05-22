"""
Functions to sample zero-centered nx x ny random vector with exponential spatial covariance
See https://math.stackexchange.com/questions/163470/generating-correlated-random-numbers-why-does-cholesky-decomposition-work
"""
import argparse

import numpy
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist

def exponential_covariance(distance, length, sigma):
    return numpy.exp(-distance/numpy.float(length)) * sigma**2

def build_covariance_matrix(nx, ny, length, sigma, lx=1.0, ly=1.0):
    dx, dy = lx/nx, ly/ny
    x, y = numpy.meshgrid((numpy.arange(nx)+.5)*dx, (numpy.arange(ny)+.5)*dy)
    coords = list(zip(x.ravel(), y.ravel()))
    distance_matrix = cdist(coords, coords)
    covariance_matrix = exponential_covariance(distance_matrix, length, sigma)
    return covariance_matrix

def generate(nx, ny, length, sigma, lx=1.0, ly=1.0):
    """ generate single realization """
    chol = _chol(nx, ny, length, sigma, lx, ly)
    u = numpy.random.randn(ny, nx).ravel()
    v = chol.dot(u).reshape(ny, nx)
    return v

def batch_generate(nx, ny, length, sigma, lx=1.0, ly=1.0, sample_size=1):
    """ generate multiple realizations """
    chol = _chol(nx, ny, length, sigma, lx, ly)
    ubatch = numpy.random.randn(ny, nx, sample_size).reshape(ny*nx, sample_size)
    vbatch = chol.dot(ubatch)
    vbatch = vbatch.T.reshape(sample_size, ny, nx)
    return vbatch

def _chol(nx, ny, length, sigma, lx, ly):
    covariance_matrix = build_covariance_matrix(nx, ny, length, sigma, lx, ly)
    chol = cholesky(covariance_matrix, lower=True)
    return chol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outfile', default='tmp.npy', help='output filename')
    parser.add_argument('--grid_size', nargs=2, type=int, default=[64, 64], help='resolution of the grid')
    parser.add_argument('--domain_size', nargs=2, type=float, default=[1.0, 1.0], help='default is a unit square')
    parser.add_argument('--length', type=float, default=1.0, help='length scale of exponential covariance')
    parser.add_argument('--sigma', type=float, default=1.0, help='amplitude of exponential covariance')
    parser.add_argument('--sample_size', type=int, default=10)
    args = parser.parse_args()

    numpy.random.seed(args.seed)

    nx, ny = args.grid_size
    lx, ly = args.domain_size

    data = batch_generate(nx, ny, args.length, args.sigma, lx, ly, args.sample_size)
    numpy.save(args.outfile, data)

    print('Generated data saved at {}'.format(args.outfile))
