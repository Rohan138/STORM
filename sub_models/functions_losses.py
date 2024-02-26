from tinygrad import Tensor

def linspace(start, end, steps): return Tensor.full(steps, start, requires_grad=False) + Tensor.arange(steps, requires_grad=False) * ((end-start) / (steps-1))

def symlog(x):
    Tensor.no_grad = True
    loss = Tensor.sign(x) * Tensor.log(1 + Tensor.abs(x))
    Tensor.no_grad = False
    return loss

def digitize(x: Tensor, bins: Tensor):
    return (x.unsqueeze(-1) - bins).relu().argmin(-1).contiguous().realize()

def symexp(x):
    Tensor.no_grad = True
    loss = Tensor.sign(x) * (Tensor.exp(Tensor.abs(x)) - 1)
    Tensor.no_grad = False
    return loss


def mse_loss(x, y):
    return Tensor.square(x - y).mean()


class SymLogLoss:
    def __call__(self, output, target):
        target = symlog(target)
        return 0.5 * mse_loss(output, target)


class SymLogTwoHotLoss():
    def __init__(self, num_classes, lower_bound, upper_bound):
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes - 1)

        self.bins = linspace(lower_bound, upper_bound, num_classes)

    def __call__(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = digitize(target, self.bins)
        diff = target - self.bins[index - 1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = Tensor.clip(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1 - weight) * Tensor.one_hot(
            index - 1, self.num_classes
        ) + weight * Tensor.one_hot(index, self.num_classes)

        loss = -target_prob * Tensor.log_softmax(output, axis=-1)
        loss = loss.sum(axis=-1)
        return loss.mean()

    def decode(self, output):
        return symexp(Tensor.softmax(output, dim=-1) @ self.bins)


if __name__ == "__main__":
    loss_func = SymLogTwoHotLoss(255, -20, 20)
    B = 2
    T = 4
    output = Tensor.randn(B, T, 255)
    target = Tensor.randint(B, T, low = -15, high = 15).float()
    print(target.numpy())
    loss = loss_func(output, target)
    print(loss.numpy())
