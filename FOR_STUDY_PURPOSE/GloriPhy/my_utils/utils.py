import torch
from torch.autograd import Variable
from types import SimpleNamespace
def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def spec_to_network_input(x, opts=None):
    if opts is None:
        opts = SimpleNamespace()
        opts.freq_size = 4096//8
        opts.normalization = True
        opts.x_image_channel = 2

    """Converts numpy to variable."""
    freq_size = opts.freq_size
    print(freq_size)
    # trim
    trim_size = freq_size // 2
    # up down 拼接
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    if opts.normalization:
        y_abs = torch.abs(y)
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y = torch.div(y, y_abs_max)

    if opts.x_image_channel == 2:
        y = torch.view_as_real(y)  # [B,H,W,2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        y = torch.angle(y)  # [B,H,W]
        y = torch.unsqueeze(y, 1)  # [B,H,W]

    #   # === 4. PAD time dimension to even ===
    # _, _, H, T = y.shape

    # y = adjust_time_dim(y)
    
    return y  # [B,2,H,W]
