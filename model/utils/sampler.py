from jaxtyping import Float, Int64, Shaped
from torch import Tensor
from einops import reduce
import torch
import torch.nn.functional as F


def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
):
# tuple[
#     Int64[Tensor, "*batch sample"],  # index
#     Float[Tensor, "*batch sample"],  # probability density
# ]
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    cdf = normalized_pdf.cumsum(dim=-1)
    samples = torch.rand((*batch, num_samples), device=pdf.device)
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
):
    # tuple[
    #     Int64[Tensor, "*batch sample"],  # index
    #     Float[Tensor, "*batch sample"],  # probability density
    # ]
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)


class DistributionSampler:
    def sample(
        self,
        pdf: Float[Tensor, "*batch bucket"],
        deterministic: bool,
        num_samples: int,
    ):
    # tuple[
    #     Int64[Tensor, "*batch sample"],  # index
    #     Float[Tensor, "*batch sample"],  # probability density
    # ]
        """Sample from the given probability distribution. Return sampled indices and
        their corresponding probability densities.
        """
        if deterministic:
            index, densities = gather_discrete_topk(pdf, num_samples)
        else:
            index, densities = sample_discrete_distribution(pdf, num_samples)
        return index, densities

    def gather(
        self,
        index: Int64[Tensor, "*batch sample"],
        target: Shaped[Tensor, "..."],  # *batch bucket *shape
    ) -> Shaped[Tensor, "..."]:  # *batch *shape
        """Gather from the target according to the specified index. Handle the
        broadcasting needed for the gather to work. See the comments for the actual
        expected input/output shapes since jaxtyping doesn't support multiple variadic
        lengths in annotations.
        """
        bucket_dim = index.ndim - 1
        while len(index.shape) < len(target.shape):
            index = index[..., None]
        broadcasted_index_shape = list(target.shape)
        broadcasted_index_shape[bucket_dim] = index.shape[bucket_dim]
        index = index.broadcast_to(broadcasted_index_shape)

        # Add the ability to broadcast.
        if target.shape[bucket_dim] == 1:
            index = torch.zeros_like(index)

        return target.gather(dim=bucket_dim, index=index)


class DifferentiableSampler:
    """Differentiable sampling methods for anchor position selection."""
    def __init__(self, method: str = "soft", temperature: float = 1.0):
        self.method = method
        self.temperature = temperature
    
    @staticmethod
    def soft_weighted_sum(
        pdfs: torch.Tensor,  # (B, N, H, W, D+1)
        anchor_pts: torch.Tensor,  # (B, N, H, W, D, 3)
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute weighted sum of anchor positions using pdf as weights.
        Fully differentiable - gradients flow back to pdfs.
        
        Args:
            pdfs: Probability distribution over depth bins (last bin is "empty")
            anchor_pts: 3D positions for each depth bin
            temperature: Softmax temperature (lower = sharper)
            
        Returns:
            Weighted positions (B, N, H, W, 3) and confidence scores
        """
        # Remove the "empty" bin from pdfs for position computation
        depth_pdfs = pdfs[..., :anchor_pts.shape[-2]]  # (B, N, H, W, D)
        empty_prob = pdfs[..., -1:]  # (B, N, H, W, 1)
        
        # Apply temperature scaling
        if temperature != 1.0:
            depth_pdfs = F.softmax(
                torch.log(depth_pdfs + 1e-8) / temperature, dim=-1
            )
        
        # Normalize depth pdfs (exclude empty bin for position computation)
        depth_pdfs_norm = depth_pdfs / (depth_pdfs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum: position = Σ(pdf_i × anchor_pts_i)
        # anchor_pts: (B, N, H, W, D, 3), depth_pdfs_norm: (B, N, H, W, D)
        weighted_positions = (
            anchor_pts * depth_pdfs_norm.unsqueeze(-1)
        ).sum(dim=-2, keepdim=True)  # (B, N, H, W, 1, 3)

        # Confidence = 1 - empty_prob (how confident we are there's a surface)
        confidence = 1.0 - empty_prob  # (B, N, H, W, 1)

        return weighted_positions, confidence, depth_pdfs_norm
    
    @staticmethod
    def gumbel_softmax_sample(
        logits: torch.Tensor,  # (B, N, H, W, D+1)
        anchor_pts: torch.Tensor,  # (B, N, H, W, D, 3)
        temperature: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Use Gumbel-Softmax for differentiable categorical sampling.
        
        Args:
            logits: Raw logits (before softmax)
            anchor_pts: 3D positions for each depth bin
            temperature: Gumbel temperature (lower = more discrete)
            hard: If True, use straight-through estimator
            
        Returns:
            Sampled positions with gradients
        """
        # Gumbel-Softmax on depth bins only (exclude empty)
        depth_logits = logits[..., :-1]  # (B, N, H, W, D)
        
        # Sample from Gumbel-Softmax
        gumbel_weights = F.gumbel_softmax(
            depth_logits, tau=temperature, hard=hard, dim=-1
        )  # (B, N, H, W, D)
        
        # Weighted sum with Gumbel weights
        sampled_positions = (
            anchor_pts * gumbel_weights.unsqueeze(-1)
        ).sum(dim=-2, keepdim=True)  # (B, N, H, W, 1, 3)

        # Confidence from gumbel weights (max weight per pixel)
        confidence = gumbel_weights.max(dim=-1, keepdim=True).values  # (B, N, H, W, 1)

        return sampled_positions, confidence, gumbel_weights
    
    @staticmethod
    def straight_through_sample(
        pdfs: torch.Tensor,
        anchor_pts: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Straight-through estimator: hard selection forward, soft gradients backward.
        """
        depth_pdfs = pdfs[..., :-1]  # Remove empty bin
        
        # Forward: hard selection (argmax or top-k)
        if num_samples == 1:
            hard_indices = depth_pdfs.argmax(dim=-1, keepdim=True)  # (B, N, H, W, 1)
        else:
            hard_indices = depth_pdfs.topk(num_samples, dim=-1).indices
        
        # Create one-hot for hard selection
        one_hot = torch.zeros_like(depth_pdfs).scatter_(-1, hard_indices, 1.0)
        
        # Straight-through: use hard one-hot forward, soft pdfs backward
        # This trick: one_hot - soft.detach() + soft
        soft_weights = depth_pdfs / (depth_pdfs.sum(dim=-1, keepdim=True) + 1e-8)
        st_weights = one_hot - soft_weights.detach() + soft_weights
        
        # Weighted sum with straight-through weights
        sampled_positions = (
            anchor_pts * st_weights.unsqueeze(-1)
        ).sum(dim=-2, keepdim=True)  # (B, N, H, W, 1, 3)

        # Confidence = max of soft_weights (how peaked the distribution is)
        confidence = soft_weights.max(dim=-1, keepdim=True).values  # (B, N, H, W, 1)

        return sampled_positions, confidence, st_weights

    def sample(
        self,
        pdfs: torch.Tensor,  # (B, N, H, W, D+1)
        anchor_pts: torch.Tensor,  # (B, N, H, W, D, 3)
        hard: bool = False,
    ) -> torch.Tensor:
        if self.method == "soft":
            return self.soft_weighted_sum(pdfs, anchor_pts, self.temperature)
        elif self.method == "gumbel":
            return self.gumbel_softmax_sample(pdfs, anchor_pts, self.temperature, hard)
        elif self.method == "straight":
            return self.straight_through_sample(pdfs, anchor_pts)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
