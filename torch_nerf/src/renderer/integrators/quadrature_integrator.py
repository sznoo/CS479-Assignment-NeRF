"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase
import time


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.

        st = time.time()
        alpha = 1.0 - torch.exp(-sigma * delta)
        transmittance = torch.exp(torch.cumsum(-sigma*delta, dim=1))
        weights = alpha * transmittance
        weights_expand = weights.unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0)

        rgbs = (weights_expand * radiance).sum(dim=1) 
        # print(f">>>integrate_along_rays: {time.time() - st} device = {alpha.device}")

        return rgbs, weights

        # raise NotImplementedError("Task 3")
