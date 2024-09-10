from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ComposeFilters
from rnaglib.transforms import RibosomalFilter
from rnaglib.transforms import PDBIDNameTransform
from rnaglib.transforms import ResidueAttributeFilter


class ProteinBindingSiteDetection(ResidueClassificationTask):
    """Residue-level task where the job is to predict a binary variable
    at each residue representing the probability that a residue belongs to
    a protein-binding interface
    """

    target_var = "binding_protein"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def build_dataset(self, root):
        # get full database
        full_dataset = RNADataset(debug=self.debug)

        # build the filters
        ribo_filter = RibosomalFilter()
        non_bind_filter = ResidueAttributeFilter(attribute=self.target_var)
        filters = ComposeFilters([ribo_filter, non_bind_filter])

        # assign a name to each remaining RNA
        add_name = PDBIDNameTransform()

        # apply filters and transforms
        rnas = filters(full_dataset)
        rnas = add_name(rnas)

        # initialize final dataset
        features_computer = FeaturesComputer(nt_targets=self.target_var)
        dataset = RNADataset(
            rnas=[r["rna"] for r in rnas], features_computer=features_computer
        )
        return dataset
