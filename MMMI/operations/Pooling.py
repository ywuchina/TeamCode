import torch


class Pooling(torch.nn.Module):
    def __init__(self, pool_type='max'):
        self.pool_type = pool_type
        super(Pooling, self).__init__()

    def forward(self, input):
        if self.pool_type == 'max':
            return torch.max(input, 2)[0].contiguous()
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            return torch.mean(input, 2).contiguous()

if __name__ == '__main__':
    input = torch.randn(10, 1024, 2048)
    out = Pooling()(input)
    print(out.shape)