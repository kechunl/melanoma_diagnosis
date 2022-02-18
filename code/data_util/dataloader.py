import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 

from histocartography.utils import set_graph_on_cuda


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLHeteroGraph': lambda x: dgl.batch(x).to(DEVICE),
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}

def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class MelanomaDataset(Dataset):
    """Melanoma dataset."""

    def __init__(
            self,
            mg_path: str = None,
            load_in_ram: bool = False,
            max_num_node: int = 100000,
    ):
        """
        Melanoma dataset constructor.

        Args:
            mg_path (str, optional): Melanocyte Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(MelanomaDataset, self).__init__()

        assert not (mg_path is None), "You must provide path to melanocyte graph."

        self.mg_path = mg_path
        self.load_in_ram = load_in_ram
        self.max_num_node = max_num_node

        if mg_path is not None:
            self._load_mg()

    def _load_mg(self):
        """
        Load melanocyte graphs
        """
        self.mg_fnames = glob(os.path.join(self.mg_path, '**', '*.bin'))
        self.mg_fnames.sort()
        self.num_mg = len(self.mg_fnames)

        if self.load_in_ram:
            melanocyte_graphs = [load_graphs(os.path.join(self.mg_path, fname)) for fname in self.mg_fnames]
            self.melanocyte_graphs = [entry[0][0] for entry in melanocyte_graphs]
            self.melanocyte_graph_labels = [entry[1]['label'].item() for entry in melanocyte_graphs]

            num_node = [entry[0][0].num_nodes() for entry in melanocyte_graphs]
            self.melanocyte_graphs = [self.melanocyte_graphs[i] for i in range(len(num_node)) if num_node[i]<self.max_num_node]
            self.melanocyte_graph_labels = [self.melanocyte_graph_labels[i] for i in range(len(num_node)) if num_node[i]<self.max_num_node]
            self.num_mg = len(self.melanocyte_graphs)

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """
        if self.load_in_ram:
            mg = self.melanocyte_graphs[index]
            label = self.melanocyte_graph_labels[index]-1
        else:
            mg, label = load_graphs(self.mg_fnames[index])
            label = label['label'].item()-1
            mg = mg[0]
        mg = set_graph_on_cuda(mg) if IS_CUDA else mg
        return mg, label

    def __len__(self):
        """Return the number of samples in the melanoma dataset."""
        return self.num_mg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    # import pdb; pdb.set_trace()
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a melanoma data loader.
    """

    dataset = MelanomaDataset(**kwargs)
    dataloader = dgl.dataloading.pytorch.GraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader