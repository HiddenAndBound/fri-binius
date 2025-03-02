use std::{
    cell::UnsafeCell,
    cmp::max,
    thread::{self, available_parallelism},
};

use binius_field::{
    arithmetic_traits::{Square, TaggedSquare}, BinaryField, BinaryField128b, BinaryField1b, ExtensionField, Field, PackedBinaryField128x1b, PackedExtension, PackedField, RepackedExtension, TowerField
};
use binius_ntt::{DynamicDispatchNTT, MultithreadedNTT};
use rand::thread_rng;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use sha3::{Digest, Keccak256};
use tracing::instrument;

use crate::{
    utils::{
        TAU,
        channel::{self, Channel},
        code::{Code, LOG_RATE, RATE},
        merkle::{Hash, MerkleTree, VectorCommitment, compute_leaf_hashes, merklize},
        mle::{LagrangeBases, PackedMLE, compute_dot_product, compute_row_batch},
    },
    verifier::compute_eq_tower_ind,
};

pub struct FriCommitment {
    pub vector_commitment: VectorCommitment,
    pub packing_factor: usize,
}

#[instrument(skip_all, name = "commit", level = "debug")]
pub fn commit<F, P>(
    mle: &PackedMLE<F>,
    ntt: &MultithreadedNTT<P>,
) -> (FriCommitment, Code<BinaryField128b>, MerkleTree)
where
    BinaryField128b: ExtensionField<F> + ExtensionField<P> + PackedExtension<P>,
    F: BinaryField + TowerField + ExtensionField<P>,
    P:BinaryField + PackedField
{
    let code = Code::new_ext(&mle.coeffs, ntt);

    let leaf_hashes: Vec<Hash> = compute_leaf_hashes(&code.encoding);
    let merkle_tree = merklize(leaf_hashes);

    let vector_commitment = VectorCommitment {
        root: merkle_tree.get_root(),
        depth: (code.encoding.len().trailing_zeros() - 1) as usize,
    };

    let fri_commitment = FriCommitment {
        vector_commitment,
        packing_factor: <F as TowerField>::TOWER_LEVEL,
    };

    (fri_commitment, code, merkle_tree)
}

#[instrument(skip_all, name = "commit_fri_oracle", level = "debug")]
pub fn commit_oracle(code: &Code<BinaryField128b>) -> (VectorCommitment, MerkleTree) {
    let leaf_hashes: Vec<Hash> = compute_leaf_hashes(&code.encoding);
    let merkle_tree = merklize(leaf_hashes);

    let vector_commitment = VectorCommitment {
        root: merkle_tree.get_root(),
        depth: (code.encoding.len().trailing_zeros() - 1) as usize,
    };

    (vector_commitment, merkle_tree)
}

///We assume that each coefficient of mle actually represents a packed vector of F_2 elements equal to number of bits required to represent F or
///F's dimension as a vector space over F_2

#[instrument(skip_all, name = "prove", level = "debug")]
pub fn prove<F,P>(
    mle: &PackedMLE<F>,
    eval_point: &[BinaryField128b],
    eval: BinaryField128b,
    encoding: &Code<BinaryField128b>,
    commitment: &FriCommitment,
    merkle_tree: &MerkleTree,
    ntt: &MultithreadedNTT<P>,
    channel: &mut Channel,
) -> EvalProof
where
    BinaryField128b: ExtensionField<F> + ExtensionField<P> + PackedExtension<P>,
    F: BinaryField + TowerField,
    P:BinaryField
{
    //The statement should be observed
    channel.observe_fri_commitment(commitment);
    channel
        .observe_field_elems(eval_point)
        .expect("failed to observe eval_point");
    channel
        .observe_field_elem(eval)
        .expect("failed to observe eval");

    let (left, right) = eval_point.split_at(TAU);

    let (mut left_eq, mut right_eq) = (
        LagrangeBases::gen_from_point(left),
        LagrangeBases::gen_from_point(right),
    );

    let upper_partial_evals = get_partial_evals(mle, &right_eq);

    let tensor_batching_point = channel
        .get_random_points(TAU)
        .expect("unable to sample random point for tensor batching");

    let batching_eq = LagrangeBases::gen_from_point(&tensor_batching_point);

    let mut repacked_mle = mle.clone().repack_for_fri();

    let mut sum_check_claim = compute_row_batch(&batching_eq.vals, &upper_partial_evals);

    let mut tensored_eq = right_eq.row_batch(&batching_eq);

    let rounds = right.len();

    let mut sum_check_oracles = Vec::new();

    let mut fri_folded_codes: Vec<Code<BinaryField128b>> = Vec::new();
    let mut fri_oracles: Vec<VectorCommitment> = Vec::new();
    let mut fri_merkle_trees: Vec<MerkleTree> = Vec::new();

    let mut random_challenges = Vec::new();
    for round in 0..rounds {
        //Sum check Logic
        let poly = sum_check_round(&repacked_mle, &tensored_eq, sum_check_claim);

        channel.observe_field_elems(&poly.coeffs).expect(&format!(
            "failed to observe prover oracle in sum check: round {round}"
        ));

        let r = channel.get_random_point().expect(&format!(
            "failed to get verifier challenge in sumcheck: round {round}"
        ));

        sum_check_claim = poly.evaluate(r);

        //Folding the code with the sum check challenges.
        let folded_code: Code<BinaryField128b> = match round {
            0 => encoding.fold_code(r, round, ntt),
            _ => fri_folded_codes[round - 1].fold_code(r, round, ntt),
        };

        let (commitment, merkle_tree) = commit_oracle(&folded_code);

        channel.observe_vector_commitment(&commitment);

        fri_folded_codes.push(folded_code);
        fri_oracles.push(commitment);
        fri_merkle_trees.push(merkle_tree);
        sum_check_oracles.push(poly);
        repacked_mle = repacked_mle.fold_lo(&r);
        tensored_eq.fold_lo(&r);
        random_challenges.push(r);
    }

    //FRI
    let final_code_folded_value = fri_folded_codes[rounds - 1].idx(0);

    channel
        .observe_field_elem(final_code_folded_value)
        .expect("Failed to observe final sum check claim");

    let mut current_queries: Vec<usize> = channel
        .gen_queries(right.len() + LOG_RATE)
        .expect("Failed to generate FRI queries.")
        .iter()
        .map(|i| i >> 1)
        .collect();

    let mut round_merkle_paths: Vec<Vec<Vec<Hash>>> = Vec::new();
    let mut round_queried_symbols: Vec<Vec<(BinaryField128b, BinaryField128b)>> = Vec::new();
    for round in 0..rounds {
        let (merkle_paths, queried_symbols) = match round {
            0 => (
                current_queries
                    .iter()
                    .map(|i| merkle_tree.get_merkle_path(*i))
                    .collect(),
                current_queries
                    .iter()
                    .map(|i| (encoding.encoding[i << 1], encoding.encoding[(i << 1) | 1]))
                    .collect(),
            ),
            _ => (
                current_queries
                    .iter()
                    .map(|i| fri_merkle_trees[round - 1].get_merkle_path(*i))
                    .collect(),
                current_queries
                    .iter()
                    .map(|i| {
                        (
                            fri_folded_codes[round - 1].encoding[i << 1],
                            fri_folded_codes[round - 1].encoding[(i << 1) | 1],
                        )
                    })
                    .collect(),
            ),
        };
        round_merkle_paths.push(merkle_paths);
        round_queried_symbols.push(queried_symbols);

        current_queries.iter_mut().for_each(|i| {
            *i >>= 1;
        });
    }

    EvalProof::new(
        upper_partial_evals,
        sum_check_oracles,
        fri_oracles,
        final_code_folded_value,
        round_queried_symbols,
        round_merkle_paths,
    )
}

pub struct EvalProof {
    pub upper_partial_evals: Vec<BinaryField128b>,
    pub sum_check_oracles: Vec<Univariate>,
    pub final_folded_value: BinaryField128b,
    pub fri_oracles: Vec<VectorCommitment>,
    pub fri_queried_symbols: Vec<Vec<(BinaryField128b, BinaryField128b)>>,
    pub fri_merkle_paths: Vec<Vec<Vec<Hash>>>,
}

impl EvalProof {
    pub fn new(
        upper_partial_evals: Vec<BinaryField128b>,
        sum_check_oracles: Vec<Univariate>,
        fri_oracles: Vec<VectorCommitment>,
        final_folded_value: BinaryField128b,
        fri_queried_symbols: Vec<Vec<(BinaryField128b, BinaryField128b)>>,
        fri_merkle_paths: Vec<Vec<Vec<Hash>>>,
    ) -> EvalProof {
        EvalProof {
            upper_partial_evals,
            sum_check_oracles,
            fri_oracles,
            final_folded_value,
            fri_queried_symbols,
            fri_merkle_paths,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Univariate {
    pub coeffs: Vec<BinaryField128b>,
}

impl Univariate {
    pub fn new(coeffs: Vec<BinaryField128b>) -> Univariate {
        Univariate { coeffs }
    }

    pub fn evaluate(&self, r: BinaryField128b) -> BinaryField128b {
        let mut eval = BinaryField128b::ZERO;

        for val in self.coeffs.iter().rev() {
            eval = *val + eval * r
        }
        eval
    }
}

#[instrument(skip_all, name = "get partial evals", level = "debug")]
pub fn get_partial_evals<F>(mle: &PackedMLE<F>, eq: &LagrangeBases) -> Vec<BinaryField128b>
where
    F: BinaryField + TowerField,
    BinaryField128b: ExtensionField<F>,
{
    (0..1 << TAU)
        .into_par_iter()
        .map(|k| {
            let vars = mle.variables;

            let res: BinaryField128b = (0..1 << (vars - TAU))
                .into_iter()
                .map(move |j| mle.packed_idx(k.clone() | (j << TAU)) * eq.idx(j))
                .sum();

            res
        })
        .collect()
}

#[instrument(skip_all, name = "sum check", level = "debug")]
pub fn sum_check_round(mle:&PackedMLE<BinaryField128b>, eq:&LagrangeBases, sum_check_claim:BinaryField128b)->Univariate{

    let half_size = mle.len()/2;

    let eval_at_0 = (0..half_size)
        .into_par_iter()
        .map(|i| (mle.idx(i << 1) * eq.idx(i << 1)))
        .sum();

    let eval_at_1 = sum_check_claim - eval_at_0;

    let eval_at_inf: BinaryField128b = (0..half_size)
        .into_par_iter()
        .map(|i| {
            (mle.idx(i << 1) + mle.idx((i << 1) | 1))
                * (eq.idx(i << 1) + eq.idx((i << 1) | 1))
        })
        .sum();

    Univariate::new(vec![
        eval_at_0,
        eval_at_0 + eval_at_1 + eval_at_inf,
        eval_at_inf,
    ])
}