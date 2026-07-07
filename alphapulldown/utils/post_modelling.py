import os
import json
import gzip
import shutil
import logging
import pickle
from pathlib import Path

import numpy as np

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, dendrogram
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from alphapulldown.utils.distogram_parser import distogram_parser


def compress_file(file_path):
    """Compress a single file with gzip."""
    logging.info(f"Compressing file: {file_path}")
    gz_path = file_path + '.gz'
    try:
        with open(file_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)  # Remove the original file after compression
        logging.info(f"File compressed and original removed: {file_path}")
    except Exception as e:
        logging.error(f"Failed to compress file: {file_path} with error: {e}")
    return gz_path


def compress_result_pickles(output_path):
    """Compress all .pkl files in the output directory."""
    for file_name in os.listdir(output_path):
        if file_name.endswith('.pkl'):
            compress_file(os.path.join(output_path, file_name))


def remove_keys_from_pickle(file_path, keys_to_remove):
    """Remove specific keys from a .pkl file."""
    logging.info(f"Removing keys {keys_to_remove} from pickle file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Remove the specified keys
        for key in keys_to_remove:
            if key in data:
                logging.info(f"Removing key: {key}")
                del data[key]

        # Save the modified data back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        logging.info(f"Keys removed and file updated: {file_path}")
    except Exception as e:
        logging.error(f"Failed to remove keys from file: {file_path} with error: {e}")



def extract_contacts_from_pickle(file_path):

    datadict = {}

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if 'ptm' in data:
        ptm = float(data['ptm'])

    elif 'ranking_confidence' in data:
        ptm = float(data['ranking_confidence'])
    else:
        ptm = np.mean(data['plddt'], dtype=float)


    #expres_probs = 1 / (1 + np.exp(-data['experimentally_resolved']['logits']))   # shape: [37, L]
    # and per residue...
    #expres_probs = expres_probs.mean(axis=1)  # shape: [L]
    distogram = data.get('distogram', None)

    asym_id=[]
    for _idx,_seq in enumerate(data['seqs']):
        asym_id.extend([_idx+1]*len(_seq))
    asym_id = np.array(asym_id)

    bin_edges = distogram['bin_edges']
    x = distogram['logits']
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    probs = exp_x_shifted / np.sum(exp_x_shifted, axis=-1, keepdims=True)

    bin_idx=np.max(np.where(bin_edges<8))
    below8pbty = np.sum(probs, axis=2, where=(np.arange(probs.shape[-1])<bin_idx))

    import string
    chain_ids = string.ascii_uppercase
    chain_lens = []
    for _seq in enumerate(data['seqs']):
        chain_lens.append(len(_seq))
        
    # save flattened distogram
    output_dict = {'below8pbty':below8pbty, 's':chain_lens, 'asym_id':asym_id, 'chain_ids':chain_ids}
    out_fn = result_dir / f"flat_distogram.pkl"
    with open(out_fn, "wb") as f:
        pickle.dump(output_dict, f)
        
    resi_i,resi_j = np.where(below8pbty>0.8)
    requested_contacts=[]
    for i,j in zip(resi_i, resi_j):

        ci = int(asym_id[i])
        cj = int(asym_id[j])

        # skipp: close, diag, and symm
        if ci>=cj: continue

        reli = 1+i-sum(chain_lens[:ci])
        relj = 1+j-sum(chain_lens[:cj])

        requested_contacts.append(f"{reli}/{chain_ids[ci]} {relj}/{chain_ids[cj]} {below8pbty[i,j]}")

        print(f"{reli:-4d}/{chain_ids[ci]} {relj:-4d}/{chain_ids[cj]} {below8pbty[i,j]:5.2f}")

    contacts_file = result_dir / "contacts.json"
    print(f"Saving {contacts_file} with {len(requested_contacts)} contacts")
    with contacts_file.open('w') as ofile:
        ofile.write(json.dumps(requested_contacts))

    datadict['contacts']=requested_contacts


    # most-likely cross-chain contacts (<8A)
    resids1 = np.where(asym_id==1)
    resids2 = np.where(asym_id==2)
    resids1 = np.asarray(resids1).ravel()
    resids2 = np.asarray(resids2).ravel()

    target_crosspbty = np.max(below8pbty[resids1][:, resids2], axis=1).tolist()
    partner_corsspbty = np.max(below8pbty[resids2][:, resids1], axis=1).tolist()

    datadict['target_crosspbty'] = target_crosspbty
    datadict['partner_crosspbty'] = partner_corsspbty

    return datadict

# ===========================

def weighted_jaccard(objects):
    N = len(objects)
    dists = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            wj = np.minimum(objects[i], objects[j]).sum()/np.maximum(objects[i], objects[j]).sum()
            dists[i, j] = dists[j, i] = 1-wj

    return dists

def cluster_and_reorder(dist, t=0.2):
    dc = squareform(dist, checks=False)
    Z = linkage(dc, method='average')
    #dendrogram(Z)
    order = leaves_list(Z)
    labels = fcluster(Z, t=t, criterion='distance')  # cluster id per original index
    dist_reordered = dist[np.ix_(order, order)]


    sizes = np.bincount(labels)
    biggest = np.argmax(sizes)
    members = np.where(labels == biggest)[0]
    if len(members) > 1:
        sims = 1 - dist[np.ix_(members, members)]
        iu = np.triu_indices(len(members), k=1)
        consistency = sims[iu].mean()
    else:
        consistency = 0.0

    return dist_reordered, order.tolist(), labels, sizes.max() / len(labels), consistency

def clusters_from_labels(labels, order):
    """Group original indices by cluster, preserving dendrogram leaf order."""
    clusters = []
    seen = {}
    for i in order:
        c = labels[i]
        if c not in seen:
            seen[c] = len(clusters)
            clusters.append([])
        clusters[seen[c]].append(i)
    return clusters

# ===========================

def post_prediction_process(output_path, compress_pickles=False, remove_pickles=False, remove_keys=False, make_plots=True):
    """Process resulted files after the prediction."""
    print("=====> INFO: keeping distograms and derived data")
    keys_to_remove = ['aligned_confidence_probs', 'masked_msa']

    contact_dict = {}
    for file_name in os.listdir(output_path):
        if "result_model" in file_name and file_name.endswith('.pkl'):
            contact_dict[file_name] = extract_contacts_from_pickle(os.path.join(output_path, file_name))

    dists = weighted_jaccard([contact_dict[k]['target_crosspbty'] for k in contact_dict])
    corr_reordered, order, labels, largest_cluster_frac, consistency = cluster_and_reorder(dists, t=0.2)
    clusters = clusters_from_labels(labels, order)

    print(f" Clusters {'+'.join(['.'.join(map(str,_)) for _ in clusters])} # {len(clusters)} largest_frac {largest_cluster_frac:.2f} consistency {consistency:.3f}")

    contact_dict['largest_cluster_frac']=largest_cluster_frac
    contact_dict['consistency']=consistency

    with Path(output_path, "fancy_extras.pkl").open("wb") as ofile:
        pickle.dump(contact_dict, ofile)

    # Plot matrix
    if make_plots:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = plt.imshow(corr_reordered, cmap='viridis', vmin=0, vmax=1, label="Jaccard distance")
        plt.title(f"Clustered Jaccard distance Matrix")

        # Compute a font size that shrinks with matrix size
        fontsize = max(1, 10 - int(corr_reordered.shape[1] / 10))
        corrected_labels = []
        for cidx, clust in enumerate(clusters):
            for _i in clust:
                corrected_labels.append(f"#{cidx}")

        ax.set_yticks(list(range(len(corrected_labels))))
        ax.set_yticklabels(corrected_labels, fontsize=fontsize)
        ax.set_xticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        plt.subplots_adjust(left=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "clusters.png"))
        plt.close()

    do=distogram_parser()
    contacts = do.get_contacts(output_path, verbose=1)
    print(f"Saving {output_path}/contacts.json with {len(contacts)} contacts")
    with Path(output_path, 'contacts.json').open('w') as ofile:
        ofile.write(json.dumps(contacts))

    try:
        # Get the best model from ranking_debug.json
        with open(os.path.join(output_path, "ranking_debug.json"), 'r') as f:
            best_model = json.load(f)['order'][0]

        # Best pickle file based on the known naming convention
        best_pickle = f"result_{best_model}.pkl"

        logging.info(f"Identified best pickle file: {best_pickle}")

        if remove_keys:
            logging.info(f"Removing specified keys from all pickle files in {output_path}")
            for file_name in os.listdir(output_path):
                if file_name.endswith('.pkl'):
                    remove_keys_from_pickle(os.path.join(output_path, file_name), keys_to_remove)

        if compress_pickles and remove_pickles:
            # Compress only the best .pkl file and remove the others
            logging.info("Compressing and removing pickles based on conditions.")
            compress_file(os.path.join(output_path, best_pickle))
            remove_irrelevant_pickles(output_path, best_pickle)
        else:
            if compress_pickles:
                logging.info("Compressing all pickle files.")
                compress_result_pickles(output_path)
            if remove_pickles:
                logging.info("Removing all non-best pickle files.")
                remove_irrelevant_pickles(output_path, best_pickle)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please check your inputs.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def remove_irrelevant_pickles(output_path, best_pickle):
    """Remove all .pkl files that do not belong to the best model."""
    for file_name in os.listdir(output_path):
        file_path = os.path.join(output_path, file_name)
        if file_name.endswith('.pkl') and file_name != best_pickle:
            logging.info(f"Removing irrelevant pickle file: {file_path}")
            os.remove(file_path)
