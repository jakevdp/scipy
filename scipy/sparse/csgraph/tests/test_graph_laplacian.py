# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD

import numpy as np
from scipy import sparse

from scipy.sparse import csgraph


def test_graph_laplacian():
    for mat in (np.arange(10) * np.arange(10)[:, np.newaxis],
                np.ones((7, 7)),
                np.eye(19),
                np.vander(np.arange(4)) + np.vander(np.arange(4)).T,
                sparse.diags([1, 1], [-1, 1], shape=(4, 4)).toarray(),
               ):
        sp_mat = sparse.csr_matrix(mat)
        for normed in (True, False):
            laplacian = csgraph.laplacian(mat, normed=normed)
            n_nodes = mat.shape[0]
            if not normed:
                np.testing.assert_array_almost_equal(laplacian.sum(axis=0),
                                                     np.zeros(n_nodes))
            np.testing.assert_array_almost_equal(laplacian.T,
                                                 laplacian)
            np.testing.assert_array_almost_equal(\
                laplacian,
                csgraph.laplacian(sp_mat, normed=normed).todense())
