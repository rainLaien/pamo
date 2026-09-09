"""Boundary-aware analytic model candidates, certified over the complete patch."""
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from cad_mesh.analytic_remesh import _Surface, _basis, _sample_points


def reassess_analytic_patch(vertices, faces, boundary, patch, deviation):
    samples = _sample_points(vertices, faces)
    center = samples.mean(axis=0)
    scale = max(float(np.linalg.norm(np.ptp(samples, axis=0))), 1e-12)
    normalized = (samples - center) / scale
    edge_points = vertices[boundary]
    boundary_sites = np.concatenate([edge_points[:, 0] * (1 - t) + edge_points[:, 1] * t
                                     for t in np.linspace(0, 1, 5)])
    distances = cKDTree(boundary_sites).query(samples)[0]
    # Distance is a seed-selection heuristic only. Certification below uses
    # every source vertex, edge midpoint and face centroid without trimming.
    interior = np.flatnonzero(distances >= np.quantile(distances, .6))
    interior = interior[::max(1, int(math.ceil(len(interior) / 1024)))]
    selected = np.arange(0, len(samples), max(1, int(math.ceil(len(samples) / 2048))))
    tolerance = max(deviation / scale, 1e-10)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-30)
    axes = [np.asarray(patch['parameters']['axis_direction'], dtype=float)]
    centered_normals = normals - np.average(normals, axis=0, weights=np.maximum(lengths, 1e-30))
    covariance = centered_normals.T @ (centered_normals * lengths[:, None])
    axes.append(np.linalg.eigh(covariance)[1][:, 0])
    records, candidates = [], []
    for kind in ('Cylinder', 'Cone'):
        best = None
        for initial_axis in axes:
            initial_axis = initial_axis / np.linalg.norm(initial_axis)
            first, second = _basis(initial_axis)

            def decode(x):
                axis = initial_axis + x[3] * first + x[4] * second
                return x[:3], axis / np.linalg.norm(axis), x[5], x[6] if kind == 'Cone' else 0.

            def residual(x, ids):
                origin, axis, radius, slope = decode(x)
                offsets = normalized[ids] - origin
                h = offsets @ axis
                r = np.linalg.norm(offsets - h[:, None] * axis, axis=1)
                return (r - radius - slope * h) / math.sqrt(1 + slope * slope)

            h = normalized[interior] @ initial_axis
            r = np.linalg.norm(normalized[interior] - h[:, None] * initial_axis, axis=1)
            slope, radius = np.linalg.lstsq(np.column_stack((h, np.ones(len(h)))), r, rcond=None)[0]
            x = np.r_[np.zeros(5), max(float(np.mean(r)), 1e-6)]
            if kind == 'Cone':
                x = np.r_[x, np.clip(slope, -5., 5.)]
            bounds = (np.r_[[-4.] * 3, [-1.] * 2, 1e-8, [-10.] if kind == 'Cone' else []],
                      np.r_[[4.] * 3, [1.] * 2, 10., [10.] if kind == 'Cone' else []])
            try:
                seed = least_squares(residual, x, args=(interior,), bounds=bounds,
                                     loss='soft_l1', f_scale=tolerance, max_nfev=60)
                fit = least_squares(residual, seed.x, args=(selected,), bounds=bounds,
                                    loss='soft_l1', f_scale=tolerance, max_nfev=60)
                origin, axis, radius, slope = decode(fit.x)
                parameters = {'axis_origin': (center + origin * scale).tolist(),
                              'axis_direction': axis.tolist()}
                if kind == 'Cylinder':
                    parameters['radius'] = float(radius * scale)
                else:
                    if abs(slope) < 1e-6:
                        continue
                    parameters['axis_origin'] = (center + (origin - axis * radius / slope) * scale).tolist()
                    parameters['axis_direction'] = (axis * np.sign(slope)).tolist()
                    parameters['semi_angle_radians'] = math.atan(abs(slope))
                candidate = dict(patch, type=kind, parameters=parameters)
                model = _Surface(candidate)
                maximum = float(model.distance(samples).max())
                alignment = np.sum(model.normals(triangles.mean(axis=1)) * normals, axis=1)
                alignment *= 1 if np.median(alignment) >= 0 else -1
                angle = float(np.degrees(np.arccos(np.clip(alignment.min(), -1, 1))))
                qualifies = maximum <= deviation + scale * np.finfo(float).eps * 128 and angle <= 5. + 1e-8
                score = (not qualifies, maximum)
                if best is None or score < best[3]:
                    best = (maximum, angle, candidate, score)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
        if best is None:
            records.append({'type': kind, 'accepted': False, 'reason': 'fit_failed'})
            continue
        maximum, angle, candidate, _ = best
        valid = maximum <= deviation + scale * np.finfo(float).eps * 128 and angle <= 5. + 1e-8
        records.append({'type': kind, 'accepted': valid, 'sampled_maximum_deviation': maximum,
                        'maximum_normal_deviation_degrees': angle})
        if valid:
            candidates.append(candidate)
    records.append({'type': 'Freeform', 'accepted': not candidates,
                    'reason': 'retain_complete_reference_mesh' if not candidates else 'analytic_candidate_available'})
    return candidates, {'original_type': patch['type'], 'seed_method': 'farthest_40_percent_from_sampled_boundary',
                        'interior_seed_samples': len(interior), 'full_validation_samples': len(samples),
                        'candidates': records}
