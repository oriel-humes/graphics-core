import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.spatial.transform import Rotation as R

# you don't need to use astropy to do this, I'm just an astronomer
obj = ascii.read('bunny.obj', names = ['type', 'n1', 'n2', 'n3'])

# retrieve vertices
f = obj['type'] == 'v'
xs = obj[f]['n1']
ys = obj[f]['n2']
zs = obj[f]['n3']

# retrieve face indices
f = obj['type'] == 'f'
i1s = obj[f]['n1']
i2s = obj[f]['n2']
i3s = obj[f]['n3']

# change from 1-indexing to 0-indexing and reformat the triangles
tri = np.transpose([i1s-1, i2s-1, i3s-1])

# initialize and plot the bunny
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xs, ys, zs, triangles = tri)
plt.show()

# rotating the bunny so it's standing up
# you can use scipi transforms to arbitrarily position the bunny!
r = R.from_euler('x', -90, degrees=True)
vert = np.transpose([xs, ys, zs])
newx, newy, newz = np.transpose(np.dot(np.array(vert), r.as_matrix()))

# initialize and plot the bunny
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(newx, newy, newz, triangles = tri)
plt.show()
