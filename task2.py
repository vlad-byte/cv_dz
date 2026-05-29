import torch


def forward_pass(x, gamma, beta, eps=1e-5):
    norm_dims = tuple(range(-gamma.dim(), 0))
    mean = x.mean(dim=norm_dims, keepdim=True)
    var = x.var(dim=norm_dims, keepdim=True, unbiased=False)
    x_centered = x - mean
    std = torch.sqrt(var + eps)
    x_hat = x_centered / std
    y = x_hat * gamma + beta
    return y, mean, var


def backward_pass(grad_y, x, gamma, beta, eps, mean, var):
    norm_dims = tuple(range(-gamma.dim(), 0))
    batch_dims = tuple(range(grad_y.dim() - gamma.dim()))
    std = torch.sqrt(var + eps)
    x_hat = (x - mean) / std
    g = grad_y * gamma
    mean_g = g.mean(dim=norm_dims, keepdim=True)
    mean_gx = (g * x_hat).mean(dim=norm_dims, keepdim=True)
    grad_x = (1.0 / std) * (g - mean_g - x_hat * mean_gx)
    grad_gamma = (grad_y * x_hat).sum(dim=batch_dims, keepdim=False)
    grad_beta = grad_y.sum(dim=batch_dims, keepdim=False)
    return grad_x, grad_gamma, grad_beta


class LNF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        y, mean, var = forward_pass(x, gamma, beta, eps)
        ctx.save_for_backward(x, gamma, beta, mean, var)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, gamma, beta, mean, var = ctx.saved_tensors
        grad_x, grad_gamma, grad_beta = backward_pass(
            grad_y, x, gamma, beta, ctx.eps, mean, var
        )
        return grad_x, grad_gamma, grad_beta, None


def test():
    eps = 1e-5
    batch, feat = 4, 10
    x = torch.randn(batch, feat, requires_grad=True)
    gamma = torch.randn(feat, requires_grad=True)
    beta = torch.randn(feat, requires_grad=True)
    ln_ref = torch.nn.LayerNorm(feat, eps=eps)
    with torch.no_grad():
        ln_ref.weight.copy_(gamma)
        ln_ref.bias.copy_(beta)
    y_ref = ln_ref(x)
    y = LNF.apply(x, gamma, beta, eps)
    torch.testing.assert_close(y, y_ref)
    print("Forward 2D passed")
    grad_out = torch.randn_like(y)
    y_ref.backward(grad_out, retain_graph=True)
    grad_x_ref = x.grad.clone()
    grad_gamma_ref = ln_ref.weight.grad.clone()
    grad_beta_ref = ln_ref.bias.grad.clone()
    x.grad = None
    gamma.grad = None
    beta.grad = None
    y.backward(grad_out)
    torch.testing.assert_close(x.grad, grad_x_ref)
    torch.testing.assert_close(gamma.grad, grad_gamma_ref)
    torch.testing.assert_close(beta.grad, grad_beta_ref)
    print("Backward 2D passed")
    N, H, W, C = 2, 3, 4, 5
    x4 = torch.randn(N, H, W, C, requires_grad=True)
    gamma4 = torch.randn(C, requires_grad=True)
    beta4 = torch.randn(C, requires_grad=True)
    ln_ref4 = torch.nn.LayerNorm(C, eps=eps)
    with torch.no_grad():
        ln_ref4.weight.copy_(gamma4)
        ln_ref4.bias.copy_(beta4)
    y_ref4 = ln_ref4(x4)
    y4 = LNF.apply(x4, gamma4, beta4, eps)
    torch.testing.assert_close(y4, y_ref4)
    grad_out4 = torch.randn_like(y4)
    y_ref4.backward(grad_out4, retain_graph=True)
    grad_x4_ref = x4.grad.clone()
    grad_gamma4_ref = ln_ref4.weight.grad.clone()
    grad_beta4_ref = ln_ref4.bias.grad.clone()
    x4.grad = None
    gamma4.grad = None
    beta4.grad = None
    y4.backward(grad_out4)
    torch.testing.assert_close(x4.grad, grad_x4_ref)
    torch.testing.assert_close(gamma4.grad, grad_gamma4_ref)
    torch.testing.assert_close(beta4.grad, grad_beta4_ref)
    print("4D C passed")
    x5 = torch.randn(N, H, W, C, requires_grad=True)
    gamma5 = torch.randn(H, W, C, requires_grad=True)
    beta5 = torch.randn(H, W, C, requires_grad=True)
    ln_ref5 = torch.nn.LayerNorm((H, W, C), eps=eps)
    with torch.no_grad():
        ln_ref5.weight.copy_(gamma5)
        ln_ref5.bias.copy_(beta5)
    y_ref5 = ln_ref5(x5)
    y5 = LNF.apply(x5, gamma5, beta5, eps)
    torch.testing.assert_close(y5, y_ref5)
    grad_out5 = torch.randn_like(y5)
    y_ref5.backward(grad_out5, retain_graph=True)
    grad_x5_ref = x5.grad.clone()
    grad_gamma5_ref = ln_ref5.weight.grad.clone()
    grad_beta5_ref = ln_ref5.bias.grad.clone()
    x5.grad = None
    gamma5.grad = None
    beta5.grad = None
    y5.backward(grad_out5)
    torch.testing.assert_close(x5.grad, grad_x5_ref)
    torch.testing.assert_close(gamma5.grad, grad_gamma5_ref)
    torch.testing.assert_close(beta5.grad, grad_beta5_ref)
    print("4D H,W,C passed")
    print("All tests passed")


if __name__ == "__main__":
    test()
