import requests

from rnaglib.transforms import Transform


class RfamTransform(Transform):
    """Obtain the Rfam classification of an RNA and store
    as a graph attribute. If no annotation is found, stores ``None``.
    """

    name = "rfam"
    encoder = None

    def forward(self, rna_dict: dict) -> dict:
        base_url = "https://www.ebi.ac.uk/pdbe/api/nucleic_mappings/rfam/"
        pdbid = rna_dict["rna"].graph["pdbid"].lower()
        url = f"{base_url}{pdbid}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()

            if pdbid.lower() in data and "Rfam" in data[pdbid]:
                # Get the first Rfam accession number found
                rfam_acc = next(iter(data[pdbid]["Rfam"].keys()), "N/A")
            else:
                rfam_acc = None
        except requests.RequestException as e:
            rfam_acc = None
        except (KeyError, IndexError, ValueError) as e:
            rfam_acc = None

        rna_dict["rna"].graph[self.name] = rfam_acc
        return rna_dict
