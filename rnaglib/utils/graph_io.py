import sys
import os
import json
import pickle

import requests
import warnings

import tarfile
import zipfile

from networkx.readwrite import json_graph
import networkx as nx

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))


def dump_json(filename, graph):
    """
    Just a shortcut to dump a json graph more compactly

    :param filename: The dump name
    :param graph: The graph to dump
    """
    g_json = json_graph.node_link_data(graph)
    json.dump(g_json, open(filename, 'w'), indent=2)


def load_json(filename):
    """
    Just a shortcut to dump a json graph more compactly

    :param filename: The dump name

    :return: The loaded graph
    """
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph


def load_graph(filename):
    """
    This is a utility function that supports loading from json or pickle
    Sometimes, the pickle also contains rings in the form of a node dict,
    in which case the rings are added into the graph

    :param filename: json or pickle filename

    :return: networkx DiGraph object
    """
    if filename.endswith('json'):
        return load_json(filename)
    elif filename.endswith('p'):
        pickled = pickle.load(open(filename, 'rb'))
        # Depending on the data versionning, the object contained in the pickles is
        # - a graph with noderings in the nodes
        # - a dict {graph: , rings: }
        if isinstance(pickled, dict):
            graph = pickled['graph']
            # rings is a dict of dict {ring_type : {node : ring}}
            rings = pickled['rings']
            for ring_type, noderings in rings.items():
                nx.set_node_attributes(G=graph, name=f'{ring_type}_annots', values=noderings)
        else:
            graph = pickled
        return graph

    else:
        raise NotImplementedError('We have not implemented this data format yet')


def get_default_download_dir():
    """
    Get the absolute path to the download directory.
    """
    dirname = os.path.join(os.path.expanduser('~'), '.rnaglib/')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def download(url, path=None, overwrite=True, retries=5, verify_ssl=True, log=True):
    """
    Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    :param url: URL to download.
    :param path:  Destination path to store downloaded file. By default stores to the current directory
     with the same name as in url.
    :param overwrite: Whether to overwrite the destination file if it already exists.
        By default always overwrites the downloaded file.
    :param retries: The number of times to attempt downloading in case of failure or non 200 return codes.
    :param verify_ssl: bool, default True. Verify SSL certificates.
    :param log:  bool, default True Whether to print the progress for download
    :return: The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
                      'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print('Downloading %s from %s...' % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                total_length = int(r.headers.get('content-length'))
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, 'wb') as f:
                    dl = 0
                    for chunk in r.iter_content(chunk_size=1024):
                        dl += len(chunk)
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            done = int(50 * dl / total_length)
                            sys.stdout.write("\r[%s%s] %.2f %%" % \
                                             ('=' * done, ' ' * (50 - done),
                                              dl / total_length * 100))
                            sys.stdout.flush()
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    return fname


def download_name_generator(dirname=None,
                            release='iguana',
                            redundancy='NR',
                            chop=False,
                            annotated=False,
                            download_dir=get_default_download_dir()):
    """
    This builds the adress of some data based on its feature

    :param dirname: For custom data saving
    :param release: For versioning
    :param redundancy: Whether we want all RNA structures or just a filtered set
    :param chop: Whether we want all graphs or fixed size chopped parts of the whole ones
    :param annotated: Whether to include pre-computed annotation for each node with information
        to be used by kernel functions
    :return:  url address, path of the downloaded file, path for the extracted data, dirname to save it,
     hashing files if needed (annotated = True)
    """
    # Generic name
    chop_str = '_chops' if chop else ''
    annotated_str = '_annot' if annotated else ''
    tarball_name = f'{redundancy}{chop_str}{annotated_str}'

    # Find remote url and get download link
    url = f'http://rnaglib.cs.mcgill.ca/static/datasets/{release}/{tarball_name}.tar.gz'
    dl_path = os.path.join(download_dir, f'data/downloads/{tarball_name}.tar.gz')

    # Complete dl path depending on annotation and optionally get hashing too
    if annotated_str == '':
        data_path = os.path.join(download_dir, 'data/graphs/')
        hashing_info = None
    else:
        data_path = os.path.join(download_dir, 'data/annotated/')
        hashing_url = f'http://rnaglib.cs.mcgill.ca/static/datasets/{release}/{tarball_name}_hash.p'
        hashing_path = os.path.join(download_dir, f'data/hashing/{tarball_name}_hash.p')
        hashing_info = (hashing_url, hashing_path)
    dirname = tarball_name if dirname is None else dirname
    return url, dl_path, data_path, dirname, hashing_info


def download_graphs(redundancy='NR', chop=False, annotated=False, overwrite=False,
                    download_dir=get_default_download_dir(), verbose=False):
    """
    Based on the options, get the right data from the latest release and put it in download_dir.
    :param redundancy: Whether to include all RNAs or just a non-redundant set as defined by BGSU
    :param chop: Whether to use full graphs or their chopped versions
    :param annotated: Whether to include graphlet annotations in the graphs. This will also create a hashing directory and table
    :param overwrite: To overwrite existing data
    :param download_dir: Where to save this data. Defaults to ~/.rnaglib/
    :return: the path of the data along with its hashing.
    """
    # Get the correct names for the download option and download the correct files
    hashing_path = None
    download_dir = get_default_download_dir() if download_dir is None else download_dir
    url, dl_path, data_path, dirname, hashing = download_name_generator(redundancy=redundancy,
                                                                        chop=chop,
                                                                        annotated=annotated,
                                                                        download_dir=download_dir)
    full_data_path = os.path.join(data_path, dirname)
    if verbose:
        print(f'Getting dataset : {full_data_path}')
        if hashing is not None:
            print(f'Getting hashing : {hashing[1]}')
    if not os.path.exists(full_data_path) or overwrite:
        if not os.path.exists(dl_path) or overwrite:
            print('Required dataset not found, launching a download. This should take about a minute')
            download(path=dl_path,
                     url=url)
        # Expand the compressed files at the right location
        if dl_path.endswith('.zip'):
            with zipfile.ZipFile(dl_path, 'r') as zip_file:
                zip_file.extractall(path=data_path)
        elif '.tar' in url:
            with tarfile.open(dl_path) as tar_file:
                tar_file.extractall(path=data_path)
    else:
        if verbose:
            print(f'Dataset was found and not overwritten')
    if hashing is not None:
        hashing_url, hashing_path = hashing
        if not os.path.exists(hashing_path) or overwrite:
            download(path=hashing_path,
                     url=hashing_url)
        else:
            if verbose:
                print(f'Hashing was found and not overwritten')
    return full_data_path, hashing_path


def graph_from_pdbid(pdbid, graph_dir=None, graph_format='json'):
    """Fetch an annotated graph with a PDBID.

    :param pdbid: PDB id to fetch
    :param graph_dir: path containing annotated graphs
    :param graph_format: which format to load (JSON, or networkx)
    """

    if graph_format == 'nx':
        graph_name = os.path.join(pdbid.lower() + '.nx')
    elif graph_format == 'json':
        graph_name = os.path.join(pdbid.lower() + '.json')
    else:
        raise ValueError(f"Invalid graph format {graph_format}. Use NetworkX or JSON.")

    graph_path = None

    # Try in look into the existing data, we need to check for both annotated and graphs, as well as in each dl
    if graph_dir is None:
        dl_dir = get_default_download_dir()
        found = False
        for parent_dirname in {'annotated', 'graphs'}:
            parent_dir = os.path.join(dl_dir, 'data', parent_dirname)
            if found:
                break
            if os.path.exists(parent_dir):
                for data_dirname in os.listdir(parent_dir):
                    # No need to screen chops
                    if 'chop' in data_dirname:
                        continue
                    data_dir = os.path.join(parent_dir, data_dirname)
                    if graph_name in os.listdir(data_dir):
                        found = True
                        graph_path = os.path.join(data_dir, graph_name)
    else:
        graph_path = os.path.join(graph_dir, graph_name)

    if graph_path is None:
        print('The required pdb was not found in existing default downloads, '
              'please provide a path to look for the graph')
        return None
    graph = load_graph(graph_path)
    return graph


if __name__ == '__main__':
    # tmp_path = '../../examples/2du5.json'
    # g = load_json(tmp_path)
    # print(g.nodes())
    default = get_default_download_dir()
    print(default)
    graph_from_pdbid('4nlf')
