#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:24:43 2025

@author: vrath
"""
import numpy as np
from scipy.sparse import csr_array, csc_array, coo_array, issparse

def prune_rebuild(A, threshold):
    # convert to COO format (triplet)
    coo = A.tocoo()
    absdata = np.abs(coo.data)
    keep = absdata >= threshold
    if not keep.all():
        # build new CSR from filtered coordinates
        A = csr_array((coo.data[keep], (coo.row[keep], coo.col[keep])),
                                 shape=A.shape)
        return A
    else:
        return A.tocsr()


def prune_inplace(A, threshold):
    # ensure CSR for data/indices/indptr access

    if issparse(A):
        if not A.format == 'csr':  # isspmatrix_csr(A):
            A = A.tocsr()
    else:
        A = csr_array(A)

    # mark tiny entries as explicit zeros in data array
    mask = np.abs(A.data) < threshold
    if mask.any():
        A.data[mask] = 0
        A.eliminate_zeros()
    return A

def prune_csr_arrays(A, threshold):
    if not issparse(A):
        A = A.tocsr()
    data, indices, indptr = A.data, A.indices, A.indptr
    n_rows = A.shape[0]

    # estimate new nnz and allocate lists (py loops unavoidable per-row)
    new_data = []
    new_idx = []
    new_indptr = np.empty(n_rows + 1, dtype=indptr.dtype)
    p = 0
    new_indptr[0] = 0
    for i in range(n_rows):
        start, stop = indptr[i], indptr[i+1]
        row_data = data[start:stop]
        row_idx = indices[start:stop]
        keep_mask = np.abs(row_data) >= threshold
        if keep_mask.any():
            new_data.append(row_data[keep_mask])
            new_idx.append(row_idx[keep_mask])
            p += keep_mask.sum()
        new_indptr[i+1] = p

    if p == data.size:
        return A  # nothing removed
    # concatenate and create CSR with existing index dtype
    new_data = np.concatenate(new_data) if new_data else np.array([], dtype=data.dtype)
    new_idx = np.concatenate(new_idx) if new_idx else np.array([], dtype=indices.dtype)
    A2 = csr_array((new_data, new_idx, new_indptr), shape=A.shape)
    return A2

# keep top-k largest magnitudes per row
def dense_to_csr_topk_per_row(X, k, dtype=None):
    import numpy as np
    rows, cols, data = [], [], []
    for i in range(X.shape[0]):
        row = X[i]
        if k >= row.size:
            mask = row != 0
            cols_i = np.nonzero(mask)[0]; data_i = row[cols_i]
        else:
            idx = np.argpartition(-np.abs(row), k-1)[:k]
            mask = row[idx] != 0
            cols_i = idx[mask]; data_i = row[cols_i]
        rows.append(np.full(cols_i.shape, i, dtype=np.int64))
        cols.append(cols_i.astype(np.int64))
        data.append(data_i.astype(dtype if dtype is not None else X.dtype))
    if not rows:
        return csr_array(X.shape, dtype=dtype if dtype is not None else X.dtype)
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    return csr_array((data, (rows, cols)), shape=X.shape)

def dense_to_csr_chunked(X, threshold=0.0, chunk_rows=1000, dtype=None):
    from collections import deque
    rows_list = []
    cols_list = []
    data_list = []
    nrows = X.shape[0]
    for r0 in range(0, nrows, chunk_rows):
        r1 = min(nrows, r0 + chunk_rows)
        block = X[r0:r1]
        mask = np.abs(block) > threshold
        rr, cc = np.nonzero(mask)
        rows_list.append((rr + r0).astype(np.int64))
        cols_list.append(cc.astype(np.int64))
        data_list.append(block[rr, cc].astype(dtype if dtype is not None else X.dtype))
    if not rows_list:
        return csr_array(X.shape, dtype=dtype if dtype is not None else X.dtype)
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    return csr_array((data, (rows, cols)), shape=X.shape)

