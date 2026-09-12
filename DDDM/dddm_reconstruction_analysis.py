import torch
from torch.func import jvp, vjp

def disable_checkpointing(model):
    for module in model.modules():
        if hasattr(module, "checkpoint"):
            module.checkpoint = False


def estimate_jacobian_spectral_norm(
    model,
    x_T,
    x_bar,
    T,
    model_kwargs=None,
    num_power_iter=10,
):


    if model_kwargs is None:
        model_kwargs = {}

    disable_checkpointing(model)    

    x_bar = x_bar.detach().requires_grad_(True)

    def F_fn(z):
        z = z.contiguous()
        return model(
            x_T,
            T,
            context=z,
            **model_kwargs,
        )

    v = torch.randn_like(x_bar)
    v = v / (torch.linalg.vector_norm(v) + 1e-12)

    for _ in range(num_power_iter):
        _, Jv = jvp(F_fn, (x_bar,), (v,))

        Jv_norm = torch.linalg.vector_norm(Jv)
        if Jv_norm < 1e-12:
            return x_bar.new_tensor(0.0)

        u = Jv / Jv_norm

        _, vjp_fn = vjp(F_fn, x_bar)
        v = vjp_fn(u)[0]

        v_norm = torch.linalg.vector_norm(v)
        if v_norm < 1e-12:
            return x_bar.new_tensor(0.0)

        v = v / v_norm

    _, Jv = jvp(F_fn, (x_bar,), (v,))

    return torch.linalg.vector_norm(Jv)


def hessian_vector_product_of_projection(
    model,
    x_T,
    x_bar,
    T,
    condition,
    u,
    v,
):
    
    disable_checkpointing(model)

    x_bar = x_bar.detach().requires_grad_(True)

    def scalar_projection(z):
        z = z.contiguous()
        F = model(
            x_T,
            T,
            context=z,
            **condition,
        )
        return torch.sum(F * u)

    grad = torch.autograd.grad(
        scalar_projection(x_bar),
        x_bar,
        create_graph=True,
    )[0]

    return torch.autograd.grad(
        torch.sum(grad * v),
        x_bar,
    )[0]


def estimate_hessian_bound(
    model,
    x_T,
    x_bar,
    T,
    model_kwargs=None,
    num_projections=4,
    num_power_iter=5,
):


    if model_kwargs is None:
        model_kwargs = {}

    x_bar = x_bar.detach().requires_grad_(True)

    with torch.no_grad():
        F = model(
            x_T,
            T,
            context=x_bar,
            **model_kwargs,
        )

    B_sq = x_bar.new_tensor(0.0)

    for _ in range(num_projections):

        u = torch.randn_like(F)
        u = u / (torch.linalg.vector_norm(u) + 1e-12)

        v = torch.randn_like(x_bar)
        v = v / (torch.linalg.vector_norm(v) + 1e-12)

        for _ in range(num_power_iter):

            Hv = hessian_vector_product_of_projection(
                model,
                x_T,
                x_bar,
                T,
                model_kwargs,
                u,
                v,
            )

            Hv_norm = torch.linalg.vector_norm(Hv)

            if Hv_norm < 1e-12:
                break

            v = Hv / Hv_norm

        Hv = hessian_vector_product_of_projection(
            model,
            x_T,
            x_bar,
            T,
            model_kwargs,
            u,
            v,
        )

        B_sq = torch.maximum(
            B_sq,
            torch.linalg.vector_norm(Hv),
        )

    return B_sq


def analyze_reconstruction(
    model,
    x_T,
    x_bar,
    T,
    sigma,
    model_kwargs=None,
):

    if model_kwargs is None:
        model_kwargs = {}

    disable_checkpointing(model)    

    x_bar = x_bar.detach().requires_grad_(True)

    F = model(
        x_T,
        T,
        context=x_bar,
        **model_kwargs,
    )

    # Actual DDDM reconstruction
    x_hat = x_T - F

    # r(z) = z - x_T + F(z)
    r = x_bar - x_T + F

    R = torch.linalg.vector_norm(r)

    L = estimate_jacobian_spectral_norm(
        model=model,
        x_T=x_T,
        x_bar=x_bar,
        T=T,
        model_kwargs=model_kwargs,
    )


    B_sq = estimate_hessian_bound(
        model=model,
        x_T=x_T,
        x_bar=x_bar,
        T=T,
        model_kwargs=model_kwargs,
    )

    sigma = sigma.mean().to(
        device=x_T.device,
        dtype=x_T.dtype,
    )

    mu = (
        (1.0 - L) ** 2
        - R * B_sq
    ) / sigma ** 2

    return {
        "x_hat": x_hat.detach(),
        "sigma": sigma.detach(),
        "R": R.detach(),
        "L": L.detach(),
        "B_sq": B_sq.detach(),
        "mu_bound": mu.detach(),
    }
