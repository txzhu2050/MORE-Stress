from dolfinx import fem, default_scalar_type
import ufl
import numpy as np
from scipy.interpolate import CubicSpline, lagrange
import os

class suppress_stdout_stderr(object):
        def __init__(self):
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
            self.save_fds = [os.dup(1), os.dup(2)]

        def __enter__(self):
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

        def __exit__(self, *_):
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            for fd in self.null_fds + self.save_fds:
                os.close(fd)

def assign_materials(domain, material_tags, material_values, cell_tags):
    Q = fem.functionspace(domain, ('DG', 0))
    functions = []
    for i in range(len(material_values)):
        material_value = material_values[i]
        function = fem.Function(Q)
        functions.append(function)
        for j, tag in enumerate(material_tags):
            cells = cell_tags.find(tag)
            function.x.array[cells] = np.full_like(cells, material_value[j], dtype=default_scalar_type)
    
    return functions

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, lambda_, mu):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def a(u, v, lambda_, mu):
    return ufl.inner(sigma(u, lambda_, mu), epsilon(v)) * ufl.dx

def L(v, alpha, temperature):
    return ufl.inner(alpha * ufl.Identity(3) * temperature, epsilon(v)) * ufl.dx

def sigma_wT(u, lambda_, mu, alpha, temperature):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u) - alpha * ufl.Identity(3) * temperature

def von_mises(u, lambda_, mu, alpha, temperature):
    s = sigma_wT(u, lambda_, mu, alpha, temperature) - 1. / 3 * ufl.tr(sigma_wT(u, lambda_, mu, alpha, temperature)) * ufl.Identity(len(u))
    return ufl.sqrt(3. / 2 * ufl.inner(s, s))
    
def lagrange_interpolation(x, interp_point, interp_num, scale):
    x = x.copy()
    x[0] = x[0] / scale[0] + 1/2; x[1] = x[1] / scale[1] + 1/2; x[2] = x[2] / scale[2] + 1/2
    rst = 1
    for i in range(3):
        xx = np.linspace(0, 1, interp_num[i])
        yy = np.full_like(xx, 0.)
        yy[interp_point[i]] = 1.0
        poly = lagrange(xx, yy)
        rst = rst * poly(x[i]) * 0.1
    tmp = [np.full_like(rst, 0.)]*3
    tmp[interp_point[-1]] = rst

    return np.vstack(tmp)

def spline_interpolation(x, interp_point, interp_num, scale):
    x = x.copy()
    x[0] = x[0] / scale[0] + 1/2; x[1] = x[1] / scale[1] + 1/2; x[2] = x[2] / scale[2] + 1/2
    rst = 1
    for i in range(3):
        xx = np.linspace(0, 1, interp_num[i])
        yy = np.full_like(xx, 0.)
        yy[interp_point[i]] = 1.0
        cs = CubicSpline(xx, yy, bc_type='natural')
        rst = rst * cs(x[i]) * 0.1
    tmp = [np.full_like(rst, 0.)]*3
    tmp[interp_point[-1]] = rst

    return np.vstack(tmp)

def get_rank_parts(parts, rank, size):
    parts_per_rank = len(parts)//size
    res = len(parts) - parts_per_rank*size
    if rank < res:
        return parts[(parts_per_rank+1)*rank:(parts_per_rank+1)*(rank+1)]
    else:
        return parts[(parts_per_rank+1)*res+parts_per_rank*(rank-res):(parts_per_rank+1)*res+parts_per_rank*(rank+1-res)]

def reshape_to_contour(data, lx, ly, dx, dy, dz):
    layer = []
    for k in range(dz):
        col = []
        for i in range(ly):
            row = []
            for j in range(lx):
                row.append(data[i*lx+j].reshape((dx, dy, dz))[:,:,k])
            col.append(np.hstack(row))
        layer.append(np.vstack(col))
    
    return layer

def flip_to_contour(data):
    tmp = np.hstack((data, np.fliplr(data)))

    return np.vstack((tmp, np.flipud(tmp)))


if __name__ == "__main__": 
    pass



