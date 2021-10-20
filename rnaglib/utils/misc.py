from torch.utils.data import Subset


def tonumpy(torch_array):
    """
    Routine to get any torch tensor back to numpy without cpu or gradient problems

    :param torch_array: A torch array

    :return: A numpy array
    """
    return torch_array.detach().cpu().numpy()


def get_dataset(loader):
    """
    Using torch subsets have a weird property that messes up the dataset accessing.
     This is the recommended fix in https://discuss.pytorch.org/t/how-to-get-a-part-of-datasets/82161/7

    :param dataset: either a dataset or a Subset
    :param node_sim:

    :return:
    """
    if isinstance(loader, Subset):
        return loader.dataset.dataset
    else:
        return loader.dataset
