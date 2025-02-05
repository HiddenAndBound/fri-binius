use std::collections::HashMap;

use binius_field::{
    BinaryField,
    BinaryField64b,
    BinaryField128b,
    ExtensionField,
    Field,
    TowerField,
};
use rayon::{iter::{ IntoParallelIterator, ParallelIterator }, slice::ParallelSlice};
use sha3::{ Digest, Keccak256, digest::{ consts::U32, generic_array::GenericArray } };
use tracing::instrument;

//Wrapper struct for hash digests
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hash(pub GenericArray<u8, U32>);

//Struct for Merkle Tree. Backing type chosen to be a hashmap for average case constant insertions and indexing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleTree {
    pub data: HashMap<usize, Vec<Hash>>,
}

//Struct for Merkle Tree. Backing type chosen to be a hashmap for average case constant insertions and indexing.
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
        self.data.get(&0).unwrap()[0].clone()
    }
}

#[inline(always)]
pub fn hash(data: &[u8]) -> Hash {
    Hash(Keccak256::digest(data))
}

//We assume that the highest level tower field is T_7, so we convert any towerfield element to T_7
pub fn hash_field<F>(data: &F) -> Hash where BinaryField128b: ExtensionField<F>, F: Field {
    Hash(Keccak256::digest(BinaryField128b::from(*data).val().to_le_bytes()))
}

pub fn hash_tuple(data: &(BinaryField128b, BinaryField128b)) -> Hash
{
    let mut hasher = Keccak256::new();
    hasher.update(data.0.val().to_le_bytes());
    hasher.update(data.1.val().to_le_bytes());
    Hash(hasher.finalize())
}
#[inline(always)]
pub fn hash_concatenation(data1: &Hash, data2: &Hash) -> Hash {
    let hasher = Keccak256::new();
    let mut val = [0; 64];
    //faster than chain update
    for i in 0..32 {
        val[i] = data1.0[i];
        val[i | 32] = data2.0[i];
    }
    Hash(Keccak256::digest(val))
}

#[instrument(skip_all, name = "make_merkle_tree", level = "debug")]
pub fn merklize(leaf_hashes: Vec<Hash>) -> MerkleTree {
    assert!(
        leaf_hashes.len().is_power_of_two(),
        "Leaf hashes are not power of 2, cannot make Merkle Tree"
    );

    let mut tree: HashMap<usize, Vec<Hash>> = HashMap::new();

    let tree_depth = leaf_hashes.len().trailing_zeros() as usize;

    tree.insert(tree_depth, leaf_hashes);

    for depth in (0..tree_depth).rev() {
        let lower_layer = tree.get(&(depth + 1)).unwrap();

        let current_layer_size = lower_layer.len() / 2;

        let current_layer: Vec<Hash> = (0..current_layer_size)
            .into_par_iter()
            .map(|i| hash_concatenation(&lower_layer[2 * i], &lower_layer[2 * i + 1]))
            .collect();

        tree.insert(depth, current_layer);
    }

    MerkleTree { data: tree }
}

pub fn get_merkle_path(tree: &HashMap<usize, Vec<Hash>>, leaf_index: usize) -> Vec<Hash> {
    let tree_depth = tree.len();

    // println!("{:?}", tree.keys());
    let mut indices = vec![leaf_index; tree_depth - 1];
    let mut path = Vec::new();
    for d in 0..tree_depth - 1 {
        if ((leaf_index >> d) & 1) == 0 {
            indices[d] = (leaf_index >> d) + 1;
        } else {
            indices[d] = (leaf_index >> d) - 1;
        }
    }

    for i in 0..indices.len() {
        path.push(match tree.get(&(tree_depth - i - 1)) {
            Some(tree_layer) => tree_layer[indices[i]].clone(),
            None => panic!("Tried to index {:?}, not found", tree_depth - i),
        });
    }

    path
}

pub fn verify_merkle_path(
    commitment: &VectorCommitment,
    leaf_hash: Hash,
    leaf_index: usize,
    merkle_path: &Vec<Hash>
) {
    let mut hash = leaf_hash;

    assert_eq!(merkle_path.len(), commitment.depth);

    for d in 0..merkle_path.len() {
        if ((leaf_index >> d) & 1) == 0 {
            hash = hash_concatenation(&hash, &merkle_path[d]);
        } else {
            hash = hash_concatenation(&merkle_path[d], &hash);
        }
    }

    assert_eq!(hash, commitment.root);
}

#[instrument(skip_all, name = "compute leaf hashes", level="debug")]
pub fn compute_leaf_hashes(vals: &Vec<BinaryField128b>)->Vec<Hash>{
    vals.par_chunks(2)
        .map(|pair| {
            let mut hasher = Keccak256::new();
            hasher.update(pair[0].val().to_le_bytes());
            hasher.update(pair[1].val().to_le_bytes());

            Hash(hasher.finalize())
        })
        .collect()
}
pub mod tests {
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

        let merkle_tree = merklize(leaf_hashes);
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

        verify_merkle_path(&commitment, leaf_hashes[idx].clone(), idx, &merkle_path);
    }
}
