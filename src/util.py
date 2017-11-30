import torch


def to_scalar(var):
    return var.view(-1).data.toist()[0]

def argmax(vec):
    _, idx = vec.max(1)
    return idx.data.cpu().numpy()

# vec: batch_size, tag_size
def log_sum_exp(vec, dim):
    # batch_size
    max_value, _ = vec.max(dim=dim)
    max_value_expand = max_value.unsqueeze(dim).expand_as(vec)
    return max_value + torch.log(torch.sum((vec - max_value_expand).exp_(), dim=dim))
    # batch_size

if __name__ == '__main__':
    t = torch.FloatTensor([[-1, 1, 2],[3,3,2]])
    print(log_sum_exp(t, 1))
    x = 1
