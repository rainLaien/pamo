import os
import copy
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import torch
from torch import nn
from torch.autograd import Function
import trimesh
import numpy
from pdmc import DMC
import time
import torch.nn.functional as F
from . import _C
import numpy as np
import trimesh
import pamo_safe_project
import torchcumesh2sdf


class PaMO(nn.Module):
    def __init__(self, input_mesh, use_stage1 = True, use_stage3 = True):
        super().__init__()
        pamo = _C.CUDSP_Free()

        self.use_stage1 = use_stage1
        self.use_stage3 = use_stage3

        print("Stage1 : ", self.use_stage1)
        print("Stage3 : ", self.use_stage3)
        
        self.bbox = input_mesh.bounding_box.bounds
        diameter = np.abs(self.bbox[1] - self.bbox[0]).max()
        scale = 1.0 / diameter
        self.gt_mesh = copy.deepcopy(input_mesh)
        
        if self.use_stage3:
            self.config = pamo_safe_project.config.Stage3Config()  # default config
            self.system = pamo_safe_project.system.Stage3System(self.config)  # create a system (with all the cuda arrays)

        class DSPFunction(Function):
            @staticmethod
            def forward(ctx, points, triangles, vertices_undo, num_vertices_undo, scale, threshold, is_stuck, init):
                verts, faces, verts_occ, verts_map, verts_undo = pamo.forward(points, triangles, vertices_undo, num_vertices_undo, scale, threshold, is_stuck, init)
                ctx.points = points
                ctx.triangles = triangles
                return verts, faces, verts_occ, verts_map, verts_undo

        self.func = DSPFunction
        # vol2mesh params
        self.vol2mesh = DMC(dtype=torch.float32).cuda()
        # mesh2vol params
        self.set_remesh_resolution(256)
        self.target_faces = None

    def set_remesh_resolution(self, resolution):
        """Set the SDF grid resolution and its dependent normalization values."""
        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("Remesh resolution must be a positive integer.")

        self.R = resolution
        self.band = 3 / self.R
        self.margin = self.band * 2 + 1

    def tri_area(self, v0, v1, v2):
        cross_prod = torch.cross(v1 - v0, v2 - v0)
        return 0.5 * torch.norm(cross_prod, dim=1)


    def preprocess_mesh(self, points, triangles, band, margin):
        tris = points[triangles]
        tris = tris.cpu().numpy()
        
        tris_mean = tris.mean(axis=1).mean(axis=0)
        tris = tris - tris_mean
        
        tris_min = tris.min(0).min(0)
        tris = tris - tris_min
        tris_max = tris.max()
        tris = (tris / tris_max + band) / margin
        
        return tris, tris_min, tris_max, tris_mean

    
    def _normalized_input_vertices(self, tris_min, tris_max, tris_mean):
        vertices = np.asarray(self.gt_mesh.vertices, dtype=np.float32)
        vertices = vertices - np.asarray(tris_mean, dtype=np.float32)
        vertices = vertices - np.asarray(tris_min, dtype=np.float32)
        vertices = (
            vertices / float(tris_max) + self.band
        ) / self.margin
        return vertices

    def remesh(
        self,
        tris,
        tris_min,
        tris_max,
        tris_mean,
        sdf_mode="auto",
    ):
        #print("preprocess start")
        # tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(points, triangles, self.band, self.margin)
        
        # tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')
        # torch.cuda.synchronize()
        
        from .sdf_field import (
            apply_inside_sign,
            fast_winding_inside_mask,
            resolve_sdf_mode,
        )

        resolved_mode, mode_reason = resolve_sdf_mode(
            self.gt_mesh,
            sdf_mode,
        )
        unsigned_distance = torchcumesh2sdf.get_udf(
            tris,
            self.R,
            self.band,
        )
        if resolved_mode == "exact":
            normalized_vertices = self._normalized_input_vertices(
                tris_min,
                tris_max,
                tris_mean,
            )
            inside = fast_winding_inside_mask(
                normalized_vertices,
                np.asarray(self.gt_mesh.faces),
                self.R,
            )
            d = apply_inside_sign(unsigned_distance, inside)
            print(
                "SDF semantics : exact signed zero "
                "({}; inside cells: {})".format(
                    mode_reason,
                    int(np.count_nonzero(inside)),
                )
            )
            print("SDF isovalue : 0.0 (original input surface)")
        else:
            repair_offset_voxels = 0.9
            d = unsigned_distance - repair_offset_voxels / self.R
            print(
                "SDF semantics : repair envelope ({})".format(
                    mode_reason
                )
            )
            print(
                "SDF isovalue : unsigned distance = {:.1f} voxel; "
                "this is an offset envelope, not the original SDF=0 surface."
                .format(repair_offset_voxels)
            )
        self.last_sdf_mode = resolved_mode
        self.last_sdf = d.detach()
        
        v, f = self.vol2mesh(d, return_quads=False) #Dual MC
        if resolved_mode == "exact":
            from .sdf_optimize import project_vertices_to_sdf_zero

            v = project_vertices_to_sdf_zero(v, d)

        v, f = v.cpu().numpy(), f.cpu().numpy()
        # pdmc normalizes unpadded grid indices by (R - 1), while the
        # cumesh2sdf samples represent cell centers at (i + 0.5) / R.
        normalized_vertices = (
            v * (self.R - 1) + 0.5
        ) / self.R
        v = (
            (
                normalized_vertices * self.margin - self.band
            )
            * tris_max
            + tris_min
        )
        
        v = torch.from_numpy(v).float().cuda()
        f = torch.from_numpy(f).int().cuda()
        
        return v, f

    @torch.no_grad()
    def remesh_only(
        self,
        points,
        triangles,
        resolution=256,
        sdf_mode="auto",
    ):
        """Run only the SDF and Dual Marching Cubes remeshing stage."""
        self.set_remesh_resolution(resolution)
        print("Remesh resolution : {}".format(self.R))

        tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(
            points,
            triangles,
            self.band,
            self.margin,
        )
        tris = torch.as_tensor(tris, dtype=torch.float32, device=points.device)

        start_stage1 = time.time()
        verts, faces = self.remesh(
            tris,
            tris_min,
            tris_max,
            tris_mean,
            sdf_mode=sdf_mode,
        )
        end_stage1 = time.time()
        print(f"Time for Remeshing: {end_stage1 - start_stage1} sec")

        verts = verts.cpu().numpy() + tris_mean
        faces = faces.cpu().numpy()
        return verts, faces

    def sdf_optimize(
        self,
        points,
        triangles,
        resolution=256,
        iterations=20,
        smoothing_step=0.2,
        projection_steps=3,
        feature_angle=45.0,
        sdf_mode="auto",
    ):
        """Remesh with an SDF, then improve triangle quality on its zero set."""
        from .sdf_optimize import optimize_mesh_on_sdf

        self.set_remesh_resolution(resolution)
        print("Remesh resolution : {}".format(self.R))
        tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(
            points,
            triangles,
            self.band,
            self.margin,
        )
        tris = torch.as_tensor(
            tris,
            dtype=torch.float32,
            device=points.device,
        )

        start_stage1 = time.time()
        verts, faces = self.remesh(
            tris,
            tris_min,
            tris_max,
            tris_mean,
            sdf_mode=sdf_mode,
        )
        end_stage1 = time.time()
        print(f"Time for Remeshing: {end_stage1 - start_stage1} sec")

        normalized_vertices = (
            (
                (verts - torch.as_tensor(tris_min, device=verts.device))
                / float(tris_max)
                + self.band
            )
            / self.margin
        )
        sdf_grid_vertices = (
            normalized_vertices * self.R - 0.5
        ) / (self.R - 1)
        start_optimization = time.time()
        sdf_grid_vertices, faces = optimize_mesh_on_sdf(
            sdf_grid_vertices,
            faces.cpu().numpy(),
            self.last_sdf,
            iterations=iterations,
            smoothing_step=smoothing_step,
            projection_steps=projection_steps,
            feature_angle=feature_angle,
        )
        end_optimization = time.time()
        print(
            "Time for SDF Optimization: {} sec".format(
                end_optimization - start_optimization
            )
        )

        normalized_vertices = (
            sdf_grid_vertices * (self.R - 1) + 0.5
        ) / self.R
        verts = (
            (
                normalized_vertices * self.margin - self.band
            )
            * tris_max
            + tris_min
            + tris_mean
        )
        return verts, faces

    @torch.no_grad()
    def original_constrained_remesh(
        self,
        points,
        triangles,
        resolution=256,
        projection_distance=None,
        feature_snap_distance=None,
        coplanar_angle_tolerance=0.1,
        coplanar_distance_ratio=1e-6,
        feature_edge_target_length=None,
        feature_edge_output_angle=None,
        feature_edge_max_splits=None,
        sdf_mode="auto",
        max_edge_length=None,
        feature_angle=5.0,
        max_splits=100000,
    ):
        """
        Refine the exact zero surface with hard original-edge constraints.

        DMC connectivity cannot retain original edge identities. For exact
        SDF=0 semantics this operation therefore uses the original triangle
        connectivity as the constraint skeleton. Edges sharper than
        ``feature_angle`` are never collapsed or flipped; longest-edge
        bisection may only replace them with collinear child edges.
        """
        from .original_constrained import (
            refine_original_mesh_by_longest_edge,
        )
        from .sdf_field import resolve_sdf_mode

        resolved_mode, mode_reason = resolve_sdf_mode(
            self.gt_mesh,
            sdf_mode,
        )
        if resolved_mode != "exact":
            raise ValueError(
                "Strict original-constrained remeshing cannot use the SDF "
                "repair envelope because its offset topology cannot retain "
                "original edge identities. Supply a watertight, consistently "
                "wound mesh and use sdf_mode='exact'."
            )

        if (
            max_edge_length is not None
            and feature_edge_target_length is not None
            and not np.isclose(
                float(max_edge_length),
                float(feature_edge_target_length),
            )
        ):
            raise ValueError(
                "Conflicting max_edge_length and legacy "
                "feature_edge_target_length values."
            )
        if max_edge_length is None:
            max_edge_length = feature_edge_target_length

        if feature_edge_output_angle is not None:
            if (
                feature_angle != 5.0
                and not np.isclose(
                    float(feature_angle),
                    float(feature_edge_output_angle),
                )
            ):
                raise ValueError(
                    "Conflicting feature_angle and legacy "
                    "feature_edge_output_angle values."
                )
            feature_angle = feature_edge_output_angle
        if feature_edge_max_splits is not None:
            if (
                max_splits != 100000
                and int(max_splits) != int(feature_edge_max_splits)
            ):
                raise ValueError(
                    "Conflicting max_splits and legacy "
                    "feature_edge_max_splits values."
                )
            max_splits = feature_edge_max_splits

        legacy_projection_requested = (
            projection_distance is not None
            or feature_snap_distance is not None
            or not np.isclose(float(coplanar_angle_tolerance), 0.1)
            or not np.isclose(float(coplanar_distance_ratio), 1e-6)
        )
        if legacy_projection_requested:
            print(
                "Warning: projection/coplanar options are ignored by strict "
                "original-constrained refinement; no DMC projection or edge "
                "matching is performed."
            )

        print(
            "Original-constrained SDF semantics : exact zero surface "
            "({}). Original connectivity is retained as the hard constraint "
            "skeleton instead of replacing it with DMC connectivity.".format(
                mode_reason
            )
        )
        start_constraints = time.time()
        verts, faces, _ = refine_original_mesh_by_longest_edge(
            self.gt_mesh,
            max_edge_length=max_edge_length,
            feature_angle_degrees=feature_angle,
            max_splits=max_splits,
        )
        print(
            "Time for Strict Original Constraint Refinement: {} sec".format(
                time.time() - start_constraints
            )
        )
        return verts, faces

    @torch.no_grad()
    def feature_remesh(
        self,
        points,
        triangles,
        resolution=256,
        projection_iterations=5,
        feature_edge_target_length=None,
        feature_edges=None,
        feature_edge_angle=45.0,
        feature_edge_match_tolerance=None,
        feature_edge_max_splits=100000,
        sdf_mode="auto",
    ):
        """
        Run SDF remeshing and safely project the result toward the input mesh.

        The SDF stage produces a closed surface. Safe projection then restores
        geometric details without changing the remeshed connectivity.
        """
        if not self.use_stage3:
            raise RuntimeError(
                "Feature remeshing requires PaMO(..., use_stage3=True)."
            )

        projection_iterations = int(projection_iterations)
        if projection_iterations <= 0:
            raise ValueError("Projection iterations must be a positive integer.")

        verts, faces = self.remesh_only(
            points,
            triangles,
            resolution=resolution,
            sdf_mode=sdf_mode,
        )

        start_stage3 = time.time()
        verts, faces = pamo_safe_project.process(
            self.gt_mesh.vertices,
            self.gt_mesh.faces,
            verts,
            faces,
            projection_iterations,
            system=self.system,
            config=self.config,
        )
        end_stage3 = time.time()
        print(
            "Time for Feature Projection: {} sec".format(
                end_stage3 - start_stage3
            )
        )

        if feature_edge_target_length is not None:
            from .feature_edges import densify_remeshed_feature_edges

            start_feature_edges = time.time()
            verts, faces = densify_remeshed_feature_edges(
                self.gt_mesh,
                verts,
                faces,
                target_length=feature_edge_target_length,
                resolution=resolution,
                feature_edges=feature_edges,
                angle_degrees=feature_edge_angle,
                match_tolerance=feature_edge_match_tolerance,
                max_splits=feature_edge_max_splits,
            )
            end_feature_edges = time.time()
            print(
                "Time for Feature-edge Densification: {} sec".format(
                    end_feature_edges - start_feature_edges
                )
            )

        return verts, faces

    def feature_optimize(
        self,
        points,
        triangles,
        resolution=256,
        projection_iterations=5,
        feature_edge_target_length=None,
        feature_edges=None,
        feature_edge_angle=30.0,
        feature_edge_match_tolerance=None,
        feature_edge_max_splits=100000,
        sdf_iterations=10,
        sdf_smoothing_step=0.2,
        sdf_projection_steps=3,
        quality_iterations=5,
        quality_step=0.2,
        flip_passes=2,
        sdf_mode="auto",
    ):
        """
        Jointly preserve original features and improve triangle quality.

        The pipeline performs SDF-constrained quality relocation, safe
        projection to the original surface, optional feature-chain
        densification, then explicit corner/curve-constrained relocation and
        quality-driven flips of non-feature edges. Feature snaps that conflict
        with the local triangulation are backtracked and locked to prevent
        flipped or near-degenerate output triangles.
        """
        if not self.use_stage3:
            raise RuntimeError(
                "Feature optimization requires PaMO(..., use_stage3=True)."
            )
        projection_iterations = int(projection_iterations)
        quality_iterations = int(quality_iterations)
        flip_passes = int(flip_passes)
        quality_step = float(quality_step)
        if projection_iterations <= 0:
            raise ValueError("Projection iterations must be positive.")
        if quality_iterations <= 0:
            raise ValueError("Feature quality iterations must be positive.")
        if flip_passes < 0:
            raise ValueError("Feature flip passes must be non-negative.")
        if not 0.0 < quality_step <= 1.0:
            raise ValueError("Feature quality step must be in (0, 1].")

        from .sdf_field import resolve_sdf_mode

        resolved_mode, mode_reason = resolve_sdf_mode(
            self.gt_mesh,
            sdf_mode,
        )
        if resolved_mode != "exact":
            raise ValueError(
                "Feature optimization v1 requires an exact-compatible "
                "watertight input. Repair envelopes contain newly filled "
                "regions which cannot be projected to the original surface "
                "without changing their intended geometry."
            )
        print(
            "Feature optimization SDF semantics: exact ({})".format(
                mode_reason
            )
        )

        start_sdf_quality = time.time()
        verts, faces = self.sdf_optimize(
            points,
            triangles,
            resolution=resolution,
            iterations=sdf_iterations,
            smoothing_step=sdf_smoothing_step,
            projection_steps=sdf_projection_steps,
            feature_angle=feature_edge_angle,
            sdf_mode="exact",
        )
        print(
            "Time for Feature-aware SDF Quality Stage: {} sec".format(
                time.time() - start_sdf_quality
            )
        )

        start_projection = time.time()
        verts, faces = pamo_safe_project.process(
            self.gt_mesh.vertices,
            self.gt_mesh.faces,
            verts,
            faces,
            projection_iterations,
            system=self.system,
            config=self.config,
        )
        print(
            "Time for Feature-safe Projection: {} sec".format(
                time.time() - start_projection
            )
        )

        if feature_edge_target_length is not None:
            from .feature_edges import densify_remeshed_feature_edges

            start_densification = time.time()
            verts, faces = densify_remeshed_feature_edges(
                self.gt_mesh,
                verts,
                faces,
                target_length=feature_edge_target_length,
                resolution=resolution,
                feature_edges=feature_edges,
                angle_degrees=feature_edge_angle,
                match_tolerance=feature_edge_match_tolerance,
                max_splits=feature_edge_max_splits,
            )
            print(
                "Time for Feature-chain Densification: {} sec".format(
                    time.time() - start_densification
                )
            )

        from .feature_optimize import optimize_feature_constrained_mesh

        start_joint_quality = time.time()
        verts, faces, _ = optimize_feature_constrained_mesh(
            self.gt_mesh,
            verts,
            faces,
            resolution=resolution,
            feature_edges=feature_edges,
            feature_angle_degrees=feature_edge_angle,
            match_tolerance=feature_edge_match_tolerance,
            iterations=quality_iterations,
            smoothing_step=quality_step,
            flip_passes=flip_passes,
        )
        print(
            "Time for Explicit Feature-constrained Quality Stage: {} sec"
            .format(time.time() - start_joint_quality)
        )
        return verts, faces

    def surface_sample_remesh(
        self,
        points,
        triangles,
        sample_count=10000,
        poisson_radius=None,
        oversample=4,
        seed=0,
        feature_edges=None,
        feature_edge_angle=30.0,
        flip_passes=5,
        relax_iterations=3,
        smoothing_step=0.2,
        barycentric_margin=0.08,
        minimum_source_quality=1e-4,
        minimum_source_area_ratio=0.25,
        maximum_edge_ratio=2.0,
        minimum_edge_ratio=0.5,
        split_passes=64,
        collapse_passes=24,
        protected_source_quality=0.8,
        maximum_normal_deviation_degrees=5.0,
        maximum_surface_deviation_ratio=0.05,
        minimum_collapse_quality=0.25,
        coplanar_angle_degrees=1.0,
    ):
        """
        Remesh from points sampled directly on the original triangle surface.

        Area sampling, grid-Poisson filtering, conflict-free quality flips, and
        inserted-vertex relaxation run with CUDA tensors. Samples retain their
        original face membership; original vertices and detected feature edges
        remain exact hard constraints.
        """
        from .surface_sample import surface_sample_remesh

        verts, faces, _ = surface_sample_remesh(
            self.gt_mesh,
            points,
            triangles,
            sample_count=sample_count,
            poisson_radius=poisson_radius,
            oversample=oversample,
            seed=seed,
            feature_edges=feature_edges,
            feature_angle_degrees=feature_edge_angle,
            flip_passes=flip_passes,
            relax_iterations=relax_iterations,
            smoothing_step=smoothing_step,
            barycentric_margin=barycentric_margin,
            minimum_source_quality=minimum_source_quality,
            minimum_source_area_ratio=minimum_source_area_ratio,
            maximum_edge_ratio=maximum_edge_ratio,
            minimum_edge_ratio=minimum_edge_ratio,
            split_passes=split_passes,
            collapse_passes=collapse_passes,
            protected_source_quality=protected_source_quality,
            maximum_normal_deviation_degrees=(
                maximum_normal_deviation_degrees
            ),
            maximum_surface_deviation_ratio=(
                maximum_surface_deviation_ratio
            ),
            minimum_collapse_quality=minimum_collapse_quality,
            coplanar_angle_degrees=coplanar_angle_degrees,
        )
        return verts, faces

    def run(
        self,
        points,
        triangles,
        ratio,
        tolerance=4,
        threshold=1e-3,
        iter=1000000,
        min_verts=10000000000,
        sdf_mode="auto",
    ):
        
        self.target_faces = max(int(ratio * len(triangles)), min_verts)
        print("Target faces : {}".format(self.target_faces))

        if self.use_stage1:
            remesh_resolution = 256
            if self.target_faces <= 1000:
                remesh_resolution = 128
            if self.target_faces <= 50:
                remesh_resolution = 64
            self.set_remesh_resolution(remesh_resolution)

        # scale the input mesh
        tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(points, triangles, self.band, self.margin)
        tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')

        # stage1 (Remeshing)
        if self.use_stage1:
            start_stage1 = time.time()
            verts, faces = self.remesh(
                tris,
                tris_min,
                tris_max,
                tris_mean,
                sdf_mode=sdf_mode,
            )
            end_stage1 = time.time()
            print(f"Time for Remeshing: {end_stage1 - start_stage1} sec")
        else:
            verts = points - torch.from_numpy(tris_mean).to(points.device)
            faces = triangles

        # stage2 (Simplification)
        start_stage2 =time.time()
        verts_undo = torch.empty(0, dtype=torch.int32, device='cuda')
        n_verts_undo = 0
        count = 0
        is_stuck = 0
        scale = max(max(verts[:,0].max()-verts[:,0].min(), verts[:,1].max()-verts[:,1].min()), verts[:,2].max()-verts[:,2].min())
        init = True
        for it in range(iter):
            num_faces_prev = faces.shape[0]
            # Simplify
            verts, faces, verts_occ, verts_map, verts_undo = self.func.apply(verts, faces, verts_undo, n_verts_undo, scale, threshold, is_stuck, init)
            init = False
            n_verts_undo = verts_undo.shape[0]
            
            # set verts and faces after 1 step of simplification
            verts = verts[verts_occ.view(-1).bool()]
            faces = faces[faces[:, 0] >= 0]
            faces[:,0] = verts_map[faces[:,0].long()].view(-1)
            faces[:,1] = verts_map[faces[:,1].long()].view(-1)
            faces[:,2] = verts_map[faces[:,2].long()].view(-1)
            
            num_faces_current = faces.shape[0]
            
            if num_faces_current <= self.target_faces or num_faces_current <= 10:
                break # simplified to target ratio
            
            if num_faces_current == num_faces_prev:
                count += 1
            else:
                count = 0
                is_stuck = 0

            if count >= 2:
                is_stuck = 1
                
            if count == tolerance:
                print("Not enough edges available to be collapsed.")
                break
        
        end_stage2 = time.time()
        verts = verts.cpu().numpy()+ tris_mean
        faces = faces.cpu().numpy()
        print(f"Time for Simplification: {end_stage2 - start_stage2} sec")
        
        # stage3 (Safe projection)
        if self.use_stage3 == True:
            stage2_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            verts, faces = pamo_safe_project.process(
                self.gt_mesh.vertices,
                self.gt_mesh.faces,
                stage2_mesh.vertices,
                stage2_mesh.faces,
                5,
                system=self.system,  # if provided, reuse the same system to avoid memory allocation
                config=self.config,  # if system is not provided, use this config to create a new system
            )
            
        return verts, faces


class PaSP(nn.Module):
    def __init__(self):
        super().__init__()
        sp = _C.CUDSP()

        class PaSPFunction(Function):
            @staticmethod
            def forward(ctx, points, triangles, scale, threshold, init):
                verts, faces, verts_occ, verts_map = sp.forward(points, triangles, scale, threshold, init)
                ctx.points = points
                ctx.triangles = triangles
                return verts, faces, verts_occ, verts_map

        self.func = PaSPFunction

    def run(self, points, triangles, threshold=0.001, iter=1000):
        verts = points
        faces = triangles
        scale = max(max(verts[:,0].max()-verts[:,0].min(), verts[:,1].max()-verts[:,1].min()), verts[:,2].max()-verts[:,2].min())
        init = True
        for it in range(iter):
            # if it < 20:
            #     t = threshold / (21 - it) * 2
            # else:
            #     t = threshold
            # print(t)
            num_faces = faces.shape[0]
            verts, faces, verts_occ, verts_map = self.func.apply(verts, faces, scale, threshold, init)
            verts = verts[verts_occ.view(-1).bool()]
            faces = faces[faces[:, 0] >= 0]
            faces[:,0] = verts_map[faces[:,0].long()].view(-1)
            faces[:,1] = verts_map[faces[:,1].long()].view(-1)
            faces[:,2] = verts_map[faces[:,2].long()].view(-1)
            init = False
            # if faces.shape[0] < 4500:
            #     break
            if faces.shape[0] == num_faces:
                print("Converged at iteration {}".format(it))
                break

            # v = verts.cpu().numpy()
            # f = faces.cpu().numpy()
            # output_mesh = trimesh.Trimesh(vertices=v, faces=f)
            # output_mesh.export('/home/sarahwei/code/simp_parallel/{}.obj'.format(it))

        
        return verts, faces
