"""Online range/depth mapping and A* flight planning, without scene-object access."""
from __future__ import annotations
import heapq
import math
import numpy as np


def unproject(frame, u, v, depth=None):
    intr = frame['intrinsics']
    z = float(frame['depth'][int(np.clip(v, 0, frame['depth'].shape[0]-1)),
                             int(np.clip(u, 0, frame['depth'].shape[1]-1))] if depth is None else depth)
    ray = np.array([(u-intr['cx'])/intr['fx'], -(v-intr['cy'])/intr['fy'], -1.0])
    return np.asarray(frame['cameraPosition']) + np.asarray(frame['cameraRotation']) @ (ray*z)


class Navigator:
    def __init__(self, resolution=0.5):
        self.resolution = resolution
        self.origin = np.array([-8.5, -7.5])
        self.shape = (30, 34)
        self.height = np.full(self.shape, -np.inf)
        self.seen = np.zeros(self.shape, bool)
        self.covered = np.zeros(self.shape, bool)
        self.path = []
        self.replans = 0
        self.visited = []
        self.map_revision = 0
        self._blocked_cache = {}
        self._expanded_height = None
        self._range_samples = np.arange(0., 12., self.resolution)

    def cell(self, point):
        x, y = np.floor((np.asarray(point)[:2]-self.origin)/self.resolution).astype(int)
        return (int(np.clip(y, 0, self.shape[0]-1)), int(np.clip(x, 0, self.shape[1]-1)))

    def point(self, cell, altitude):
        y, x = cell
        return [float(self.origin[0]+(x+.5)*self.resolution),
                float(self.origin[1]+(y+.5)*self.resolution), float(altitude)]

    def inside(self, point):
        p = (np.asarray(point)[:2]-self.origin)/self.resolution
        return 0 <= p[0] < self.shape[1] and 0 <= p[1] < self.shape[0]

    def observe_ranges(self, rays):
        if not rays:
            return
        origins = np.asarray([ray['origin'] for ray in rays], dtype=float)
        directions = np.asarray([ray['direction'] for ray in rays], dtype=float)
        distance = np.asarray([12. if ray.get('distance') is None else ray['distance'] for ray in rays], dtype=float)
        samples = self._range_samples
        points = origins[:, None, :] + directions[:, None, :] * samples[None, :, None]
        cells = np.floor((points[..., :2] - self.origin) / self.resolution).astype(int)
        good = (np.isfinite(distance)[:, None] & (samples[None, :] < np.minimum(distance[:, None], 12.)) &
                (cells[..., 0] >= 0) & (cells[..., 0] < self.shape[1]) &
                (cells[..., 1] >= 0) & (cells[..., 1] < self.shape[0]))
        x, y = cells[..., 0][good], cells[..., 1][good]
        self.seen[y, x] = True
        hits = np.isfinite(distance) & (distance >= 0) & (distance < 12.)
        endpoints = origins[hits] + directions[hits] * distance[hits, None]
        self._observe_heights(endpoints)

    def _observe_heights(self, points):
        """Replace directly observed surface heights; retain unseen cells."""
        if len(points) == 0:
            return
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]
        cells = np.floor((points[:, :2] - self.origin) / self.resolution).astype(int)
        good = ((cells[:, 0] >= 0) & (cells[:, 0] < self.shape[1]) &
                (cells[:, 1] >= 0) & (cells[:, 1] < self.shape[0]))
        cells, points = cells[good], points[good]
        observed = np.full(self.shape, -np.inf)
        np.maximum.at(observed, (cells[:, 1], cells[:, 0]), points[:, 2])
        visible = np.isfinite(observed)
        self.seen[visible] = True
        old, current = self.height[visible], observed[visible]
        changed = (~np.isfinite(old)) | (np.abs(current - old) > .025)
        self.height[visible] = current
        if not np.array_equal(old, current):
            self._blocked_cache.clear()
            self._expanded_height = None
        if np.any(changed):
            self.map_revision += 1
            self._blocked_cache.clear()

    def observe_frame(self, frame, cover=True):
        depth = frame['depth'][::6,::6]
        v,u = np.mgrid[0:frame['depth'].shape[0]:6,0:frame['depth'].shape[1]:6]
        k=frame['intrinsics']
        rays=np.stack([(u-k['cx'])/k['fx'],-(v-k['cy'])/k['fy'],-np.ones_like(u)],axis=-1)
        points=(rays*depth[...,None])@np.asarray(frame['cameraRotation']).T+np.asarray(frame['cameraPosition'])
        cells=np.floor((points[...,:2]-self.origin)/self.resolution).astype(int)
        good=(np.isfinite(depth)&(depth>.05)&(depth<35)&(cells[...,0]>=0)&
              (cells[...,0]<self.shape[1])&(cells[...,1]>=0)&(cells[...,1]<self.shape[0]))
        x,y=cells[...,0][good],cells[...,1][good]
        self.seen[y,x]=True
        if cover: self.covered[y,x]=True
        self._observe_heights(points[good])

    def blocked(self, altitude):
        key = round(float(altitude), 3)
        if key in self._blocked_cache:
            return self._blocked_cache[key].copy()
        result = self._clearance_heights() > altitude - .65
        self._blocked_cache[key] = result.copy()
        return result

    def _clearance_heights(self):
        if self._expanded_height is None:
            padded = np.pad(self.height, 1, constant_values=-np.inf)
            self._expanded_height = np.maximum.reduce([
                padded[y:y + self.shape[0], x:x + self.shape[1]]
                for y in range(3) for x in range(3)])
        return self._expanded_height

    def path_safe(self, current, path):
        """Check a pending flight against current measured geometry and bounds."""
        points = [np.asarray(current, dtype=float), *[np.asarray(p, dtype=float) for p in path]]
        samples = []
        for start, end in zip(points, points[1:]):
            if not np.isfinite(start).all() or not np.isfinite(end).all():
                return False
            count = max(2, int(np.ceil(np.linalg.norm(end - start) / (self.resolution * .4))) + 1)
            samples.append(np.linspace(start, end, count))
        if not samples:
            return True
        samples = np.concatenate(samples)
        cells = np.floor((samples[:, :2] - self.origin) / self.resolution).astype(int)
        if np.any((cells[:, 0] < 0) | (cells[:, 0] >= self.shape[1]) |
                  (cells[:, 1] < 0) | (cells[:, 1] >= self.shape[0])):
            return False
        heights = self._clearance_heights()[cells[:, 1], cells[:, 0]]
        # Clear ground remains reachable during takeoff and landing.
        return not np.any(heights > np.maximum(.81, samples[:, 2]) - .65)

    def plan(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        altitude = max(float(start[2]), float(goal[2]))
        blocked = self.blocked(altitude)
        source, target = self.cell(start), self.cell(goal)
        blocked[source] = False
        if blocked[target]: raise ValueError('The measured obstacle map blocks that destination.')
        queue = [(0., source)]
        costs, parents = {source:0.}, {}
        while queue:
            _, c = heapq.heappop(queue)
            if c == target: break
            for dy,dx in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                n = c[0]+dy,c[1]+dx
                if not (0<=n[0]<self.shape[0] and 0<=n[1]<self.shape[1]) or blocked[n]: continue
                if dx and dy and (blocked[c[0],n[1]] or blocked[n[0],c[1]]): continue
                score = costs[c]+math.hypot(dx,dy)*(1 if self.seen[n] else 1.15)
                if score < costs.get(n,float('inf')):
                    costs[n], parents[n] = score,c
                    heapq.heappush(queue,(score+math.dist(n,target),n))
        if target not in costs: raise ValueError('No safe route in the measured range map.')
        cells, c = [target], target
        while c != source:
            c = parents[c]; cells.append(c)
        cells.reverse()
        # Keep changes in direction, rather than every grid sample.
        points = []
        previous = None
        for i in range(1,len(cells)):
            direction = (cells[i][0]-cells[i-1][0],cells[i][1]-cells[i-1][1])
            if previous is not None and direction != previous: points.append(self.point(cells[i-1],altitude))
            previous = direction
        points.append(goal.tolist())
        self.path = [start.tolist(),*points]
        self.replans += 1
        return [np.array(p) for p in points]

    def next_scan(self, position, altitude=4.5):
        # Score unobserved area around potential camera positions. No zone sequence.
        best = None
        blocked = self.blocked(altitude)
        visited = np.asarray(self.visited) if self.visited else np.empty((0, 3))
        for y in np.arange(-5.5,6.,2.5):
            for x in np.arange(-6.5,7.,2.5):
                p = np.array([x,y,altitude])
                if len(visited) and np.any(np.linalg.norm(visited[:, :2] - p[:2], axis=1) < 1.0): continue
                c = self.cell(p)
                if blocked[c]: continue
                ys=slice(max(0,c[0]-5),min(self.shape[0],c[0]+6))
                xs=slice(max(0,c[1]-6),min(self.shape[1],c[1]+7))
                unseen = np.count_nonzero(~self.covered[ys,xs])
                score = unseen/(2+np.linalg.norm(p[:2]-np.asarray(position)[:2]))
                if best is None or score>best[0]: best=(score,p)
        return None if best is None or best[0]<.5 else best[1]

    def snapshot(self):
        occupied = np.argwhere(np.isfinite(self.height) & (self.height>.3))
        obstacles = np.column_stack([
            self.origin[0] + (occupied[:, 1] + .5) * self.resolution,
            self.origin[1] + (occupied[:, 0] + .5) * self.resolution,
            self.height[occupied[:, 0], occupied[:, 1]]]).tolist()
        return {'path':self.path, 'replans':self.replans, 'mapRevision':self.map_revision, 'mapResolution':self.resolution,
                'mappedPercent':round(100*float(self.seen.mean()),1),
                'coveragePercent':round(100*float(self.covered.mean()),1),
                'obstacles':obstacles,
                'source':'onboard ray ranges and rendered depth'}
