
def tonumpy(torch_array):
    return torch_array.detach().cpu().numpy()
