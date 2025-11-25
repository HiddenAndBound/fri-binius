use anyhow::ensure;
use binius_field::{BinaryField128b, ExtensionField, Field};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use sha3::{
    Digest, Keccak256,
    digest::{consts::U32, generic_array::GenericArray},
};
use tracing::instrument;

/// Wrapper struct for Keccak-256 digests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hash(pub GenericArray<u8, U32>);

/// Merkle tree backed by contiguous layers (index 0 = root, last = leaves).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleTree {
    pub data: Vec<Vec<Hash>>,
}

/// Commitment that stores the Merkle root and the number of hashing rounds (tree depth).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorCommitment {
    pub root: Hash,
    pub depth: usize,
}

impl VectorCommitment {
    pub fn root(&self) -> Hash {
        self.root.clone()
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl MerkleTree {
    pub fn get_merkle_path(&self, leaf_index: usize) -> Vec<Hash> {
        get_merkle_path(&self.data, leaf_index)
    }

    pub fn get_root(&self) -> Hash {
        self.data[0][0]
    }
}

/// Hash arbitrary bytes using Keccak-256.
#[inline(always)]
pub fn hash(data: &[u8]) -> Hash {
    Hash(Keccak256::digest(data))
}

/// Hash a single field element by embedding it into `BinaryField128b` first.
//We assume that the highest level tower field is T_7, so we convert any towerfield element to T_7
pub fn hash_field<F>(data: &F) -> Hash
where
    BinaryField128b: ExtensionField<F>,
    F: Field,
{
    Hash(Keccak256::digest(
        BinaryField128b::from(*data).val().to_le_bytes(),
    ))
}

/// Hash a pair of field elements sequentially.
pub fn hash_tuple(data: &(BinaryField128b, BinaryField128b)) -> Hash {
    let mut hasher = Keccak256::new();
    hasher.update(data.0.val().to_le_bytes());
    hasher.update(data.1.val().to_le_bytes());
    Hash(hasher.finalize())
}
#[inline(always)]
pub fn hash_concatenation(data1: &Hash, data2: &Hash) -> Hash {
    let mut val = [0; 64];
    //faster than chain update
    for i in 0..32 {
        val[i] = data1.0[i];
        val[i | 32] = data2.0[i];
    }
    Hash(Keccak256::digest(val))
}

/// Build every layer of a Merkle tree from a power-of-two set of leaf hashes.
#[instrument(skip_all, name = "merklize", level = "debug")]
pub fn merklize(leaf_hashes: Vec<Hash>) -> MerkleTree {
    assert!(
        leaf_hashes.len().is_power_of_two(),
        "Leaf hashes are not power of 2, cannot make Merkle Tree"
    );

    let tree_depth = leaf_hashes.len().trailing_zeros() as usize;
    let mut layers: Vec<Vec<Hash>> = Vec::with_capacity(tree_depth + 1);
    layers.push(leaf_hashes);

    for _ in 0..tree_depth {
        let parent_layer = build_parent_layer(layers.last().unwrap());
        layers.push(parent_layer);
    }

    layers.reverse();

    MerkleTree { data: layers }
}

/// Return the sibling hashes from a leaf up to (but excluding) the root.
pub fn get_merkle_path(tree: &[Vec<Hash>], leaf_index: usize) -> Vec<Hash> {
    let leaf_depth = tree
        .len()
        .checked_sub(1)
        .expect("Merkle tree cannot be empty");

    assert!(
        leaf_index < tree[leaf_depth].len(),
        "Leaf index out of bounds"
    );

    let mut index = leaf_index;
    let mut path = Vec::with_capacity(leaf_depth);

    for depth in (1..=leaf_depth).rev() {
        let sibling = index ^ 1;
        path.push(tree[depth][sibling]);
        index >>= 1;
    }

    path
}

/// Recompute the Merkle root from a leaf hash and its path, asserting equality.
pub fn verify_merkle_path(
    commitment: &VectorCommitment,
    leaf_hash: Hash,
    leaf_index: usize,
    merkle_path: &[Hash],
) -> anyhow::Result<()> {
    let mut hash = leaf_hash;

    ensure!(
        merkle_path.len() == commitment.depth,
        "Merkle path length doesn't match claimed depth."
    );

    for d in 0..merkle_path.len() {
        let is_left_child = ((leaf_index >> d) & 1) == 0;
        hash = if is_left_child {
            hash_concatenation(&hash, &merkle_path[d])
        } else {
            hash_concatenation(&merkle_path[d], &hash)
        };
    }

    ensure!(
        hash == commitment.root,
        "Path at index {leaf_index} failed to verify."
    );
    Ok(())
}

/// Collapse pairs of field elements into leaf hashes.
#[instrument(skip_all, name = "compute_leaf_hashes", level = "debug")]
pub fn compute_leaf_hashes(vals: &[BinaryField128b]) -> Vec<Hash> {
    assert_eq!(
        vals.len() & 1,
        0,
        "Leaf construction requires an even number of field elements"
    );

    vals.par_chunks_exact(2)
        .map(|pair| {
            let mut hasher = Keccak256::new();
            hasher.update(pair[0].val().to_le_bytes());
            hasher.update(pair[1].val().to_le_bytes());

            Hash(hasher.finalize())
        })
        .collect()
}

fn build_parent_layer(child_layer: &[Hash]) -> Vec<Hash> {
    assert_eq!(
        child_layer.len() & 1,
        0,
        "Child layer must contain an even number of nodes"
    );
    child_layer
        .par_chunks_exact(2)
        .map(|pair| hash_concatenation(&pair[0], &pair[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn make_merkle_tree_test() {
        use rand::thread_rng;

        let leaf_hashes: Vec<Hash> = (0..1 << 8)
            .into_iter()
            .map(|_| {
                let random_elem = BinaryField128b::random(thread_rng());
                hash_field(&random_elem)
            })
            .collect();

        let _merkle_tree = merklize(leaf_hashes);
    }

    #[test]
    fn get_merkle_path_test() {
        use rand::thread_rng;

        let leaf_hashes: Vec<Hash> = (0..1 << 8)
            .into_iter()
            .map(|_| {
                let random_elem = BinaryField128b::random(thread_rng());
                hash_field(&random_elem)
            })
            .collect();

        let merkle_tree = merklize(leaf_hashes.clone());
        let commitment = VectorCommitment {
            root: merkle_tree.get_root(),
            depth: 8,
        };

        let idx = thread_rng().gen_range(0..1 << 8);
        let merkle_path = merkle_tree.get_merkle_path(idx);

        verify_merkle_path(&commitment, leaf_hashes[idx].clone(), idx, &merkle_path).unwrap();
    }
}
