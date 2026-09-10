def repeat_kv(x, group_size):
    B, n_kv_head, T, head_size = x.shape
    x = x.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
    x = x.reshape(B, n_kv_head * group_size, T, head_size)
    return x

