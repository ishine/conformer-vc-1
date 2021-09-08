import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void dtw_c(float[:,::1] path, double[:,::1] dist, int[:,::1] mask1, int[:,::1] mask2, int t_x, int t_y, float max_pos_val) nogil:
  cdef int x
  cdef int y
  cdef float c1
  cdef float c2
  cdef float c3

  mask1[0, 0] = 1
  for x in range(1, t_x):
    for y in range(1, t_y):
      if mask1[x - 1, y - 1] == 1:
        mask1[x, y] = 1
      elif x > 1 and mask1[x - 2, y - 1] == 1:
        mask1[x, y] = 1
      elif y > 1 and mask1[x - 1, y - 2] == 1:
        mask1[x, y] = 1
  mask2[t_x - 1, t_y - 1] = 1
  for x in range(t_x - 2, -1, -1):
    for y in range(t_y - 2, -1, -1):
      if mask2[x + 1, y + 1] == 1:
        mask2[x, y] = 1
      elif x < t_x - 2 and mask2[x + 2, y + 1] == 1:
        mask2[x, y] = 1
      elif y < t_y - 2 and mask2[x + 1, y + 2] == 1:
        mask2[x, y] = 1

  for x in range(t_x):
    for y in range(t_y):
      if x == 0 and y == 0:
        continue
      if mask1[x, y] and mask2[x, y]:
        c1 = dist[x - 1, y - 1]
        if x >= 2:
          c1 = min(c1, dist[x - 2, y - 1])
        if y >= 2:
          c1 = min(c1, dist[x - 1, y - 2])
        dist[x, y] += c1
      else:
        dist[x, y] = max_pos_val

  x = t_x - 1
  y = t_y - 1
  path[x, y] = 1
  while x > 0 or y > 0:
    if x == 1 and y == 2:
      path[x, y - 1] = 1
      path[x - 1, y - 2] = 1
      x -= 1
      y -= 2
    elif x == 2 and y == 1:
      path[x - 2, y - 1] = 1
      x -= 2
      y -= 1
    elif x == 1 and y == 1:
      path[x - 1, y - 1] = 1
      x -= 1
      y -= 1
    else:
      c1 = dist[x - 1, y - 2]
      c2 = dist[x - 1, y - 1]
      c3 = dist[x - 2, y - 1]
      if mask1[x - 1, y - 2] and mask2[x - 1, y - 2] and c1 <= c2 and c1 <= c3:
        path[x, y - 1] = 1
        path[x - 1, y - 2] = 1
        x -= 1
        y -= 2
      elif mask1[x - 2, y - 1] and mask2[x - 2, y - 1] and c3 <= c1 and c3 <= c2:
        path[x - 2, y - 1] = 1
        x -= 2
        y -= 1
      elif mask1[x - 1, y - 1] and mask2[x - 1, y - 1] and c2 <= c1 and c2 <= c3:
        path[x - 1, y - 1] = 1
        x -= 1
        y -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void dtw_c_interp(float[:,::1] path, double[:,::1] dist, int[:,::1] mask1, int[:,::1] mask2, int t_x, int t_y, float max_pos_val) nogil:
  cdef int x
  cdef int y
  cdef float c1
  cdef float c2
  cdef float c3

  mask1[0, 0] = 1
  for x in range(1, t_x):
    for y in range(1, t_y):
      if mask1[x - 1, y - 1] == 1:
        mask1[x, y] = 1
      elif x > 1 and mask1[x - 2, y - 1] == 1:
        mask1[x, y] = 1
      elif y > 1 and mask1[x - 1, y - 2] == 1:
        mask1[x, y] = 1
  mask2[t_x - 1, t_y - 1] = 1
  for x in range(t_x - 2, -1, -1):
    for y in range(t_y - 2, -1, -1):
      if mask2[x + 1, y + 1] == 1:
        mask2[x, y] = 1
      elif x < t_x - 2 and mask2[x + 2, y + 1] == 1:
        mask2[x, y] = 1
      elif y < t_y - 2 and mask2[x + 1, y + 2] == 1:
        mask2[x, y] = 1

  for x in range(t_x):
    for y in range(t_y):
      if x == 0 and y == 0:
        continue
      if mask1[x, y] and mask2[x, y]:
        c1 = dist[x - 1, y - 1]
        if x >= 2:
          c1 = min(c1, dist[x - 2, y - 1])
        if y >= 2:
          c1 = min(c1, dist[x - 1, y - 2])
        dist[x, y] += c1
      else:
        dist[x, y] = max_pos_val

  x = t_x - 1
  y = t_y - 1
  path[x, y] = 1
  while x > 0 or y > 0:
    if x == 1 and y == 2:
      path[x - 1, y - 1] = 1
      path[x - 1, y - 2] = 1
      x -= 1
      y -= 2
    elif x == 2 and y == 1:
      path[x, y] = 0.5
      path[x - 2, y] = 0.5
      path[x - 2, y - 1] = 1
      x -= 2
      y -= 1
    elif x == 1 and y == 1:
      path[x - 1, y - 1] = 1
      x -= 1
      y -= 1
    else:
      c1 = dist[x - 1, y - 2]
      c2 = dist[x - 1, y - 1]
      c3 = dist[x - 2, y - 1]
      if mask1[x - 1, y - 2] and mask2[x - 1, y - 2] and c1 <= c2 and c1 <= c3:
        path[x - 1, y - 1] = 1
        path[x - 1, y - 2] = 1
        x -= 1
        y -= 2
      elif mask1[x - 2, y - 1] and mask2[x - 2, y - 1] and c3 <= c1 and c3 <= c2:
        path[x, y] = 0.5
        path[x - 2, y] = 0.5
        path[x - 2, y - 1] = 1
        x -= 2
        y -= 1
      elif mask1[x - 1, y - 1] and mask2[x - 1, y - 1] and c2 <= c1 and c2 <= c3:
        path[x - 1, y - 1] = 1
        x -= 1
        y -= 1
