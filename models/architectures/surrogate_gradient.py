import torch


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spiking neurons
    Using fast sigmoid as described in Equation 8
    """

    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrogateGradient.scale*torch.abs(input)+1.0)**2
        return grad
