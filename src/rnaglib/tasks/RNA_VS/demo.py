import time
import torch
import torch.optim as optim

from rnaglib.tasks.RNA_VS.vs_task import VirtualScreening
from rnaglib.tasks.RNA_VS.evaluate import run_virtual_screen
from rnaglib.transforms import GraphRepresentation
from rnaglib.utils.misc import set_seed

seed = 1
set_seed(seed)

# Create a task, we need to choose a framework for the ligand representation
framework = 'pyg'
ta = VirtualScreening(root='RNA_VS', ligand_framework=framework, debug=False, precomputed=True)

# Adding representation for RNAs
ta.dataset.add_representation(GraphRepresentation(framework=framework))

# Splitting dataset
print("Splitting Dataset")
train, val, test = ta.get_split_loaders(recompute=False, batch_size=8)
print(len(train.dataset))
print(len(val.dataset))
print(len(test.dataset))

info = ta.describe()

# Create an encoding model. This model must implement a predict_ligands(pocket, ligands) method
if framework == 'pyg':
    from rnaglib.tasks.RNA_VS.model_pyg import RNAEncoder, LigandGraphEncoder, Decoder, VSModel
else:
    from rnaglib.tasks.RNA_VS.model_dgl import RNAEncoder, LigandGraphEncoder, Decoder, VSModel

model = VSModel(encoder=RNAEncoder(), lig_encoder=LigandGraphEncoder(), decoder=Decoder())
assert hasattr(model, 'predict_ligands') and callable(getattr(model, 'predict_ligands'))

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
epochs = 1
t0 = time.time()
for k in range(epochs):
    for i, batch in enumerate(train):
        pockets = batch['graph']
        in_pocket = torch.tensor(batch['in_pocket'])
        pockets.in_pocket = in_pocket

        ligands = batch['ligand']["ligands"]
        actives = batch['ligand']["actives"]
        actives = torch.tensor(actives, dtype=torch.float32)
        optimizer.zero_grad()
        out = model(pockets, ligands)
        loss = criterion(input=torch.flatten(out), target=actives)
        loss.backward()
        optimizer.step()
        # if i > 3:
        #     break
        if not i % 5:
            print(f'Epoch {k}, batch {i}/{len(train)}, '
                  f'loss: {loss.item():.4f}, time: {time.time() - t0:.1f}s')

model = model.eval()
print(f"Results for seed {seed}:")
final_vs = run_virtual_screen(model, test)
