import torch


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spiking neurons
    Using fast sigmoid as described in Equation 8
    """
    @staticmethod
    def forward(ctx, input, beta=40.0):
        ctx.save_for_backward(input)
        ctx.beta = beta
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        beta = ctx.beta
        # Fast sigmoid surrogate gradient (Equation 8)
        grad_input = grad_output * input / (1 + beta * torch.abs(input))
        return grad_input, None
