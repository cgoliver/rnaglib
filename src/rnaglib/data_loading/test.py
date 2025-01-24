from torch.utils.data import DataLoader

from rnaglib.representations import GraphRepresentation, RingRepresentation
from rnaglib.representations import PointCloudRepresentation, VoxelRepresentation
from rnaglib.data_loading import RNADataset, get_loader, FeaturesComputer
from rnaglib.kernels import node_sim
from rnaglib.data_loading import Collater
from rnaglib.dataset_transforms import split_dataset

features_computer = FeaturesComputer(nt_features='nt_code', nt_targets='binding_protein')
da = RNADataset(features_computer=features_computer)
rna_1 = da[3]
pdbid = da.available_pdbids[3]
rna_2 = da.get_pdbid(pdbid)
print(rna_2)

features_computer.remove_feature('binding_protein', input_feature=False)
features_computer.add_feature('binding_ion', input_feature=False)
dataset = RNADataset(features_computer=features_computer)
graph = dataset[0]['rna']
print(type(graph))
for node in graph.nodes(data=True):
    print(node)
    break

# PC TEST
pc_rep = PointCloudRepresentation()
da = RNADataset(representations=[pc_rep],
                features_computer=features_computer)
elt = da[0]
print(elt['point_cloud'].keys())
for k, v in elt['point_cloud'].items():
    print(k, type(v))

# VOXEL TEST
voxel_rep = VoxelRepresentation(spacing=2)
da = RNADataset(representations=[voxel_rep],
                features_computer=features_computer)
elt = da[0]
print(elt.keys())
for key, value in elt['voxel'].items():
    try:
        print(key, value.shape)
    except:
        print(key, value)

# GRAPH TEST
graph_rep = GraphRepresentation(framework='dgl')
da = RNADataset(representations=[graph_rep],
                features_computer=features_computer)
da.add_representation(voxel_rep)
da.add_representation(pc_rep)
print(da[0].keys())
print(da[0]['point_cloud'])
for key, value in da[0]['point_cloud'].items():
    try:
        print(key, value.shape)
    except:
        print(key, value)

print('graph : ', da[0]['graph'])

node_simfunc = node_sim.SimFunctionNode(method='R_1', depth=2)
ring_rep = RingRepresentation(node_simfunc=node_simfunc, max_size_kernel=None)
da.add_representation(ring_rep)

train_set, valid_set, test_set = split_dataset(dataset, split_train=0.7, split_valid=0.85)
collater = Collater(dataset=dataset)
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=2, num_workers=0, collate_fn=collater)

for batch in train_loader:
    print(batch.keys())

    for key, value in batch['point_cloud'].items():
        try:
            print(key, value.shape)
        except:
            print(key, value)
    break
