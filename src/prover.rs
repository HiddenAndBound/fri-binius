use crate::{
    Result,
    utils::{
        TAU,
        channel::Channel,
        code::{Code, LOG_RATE},
        merkle::{Hash, MerkleTree, VectorCommitment, compute_leaf_hashes, merklize},
        mle::{LagrangeBases, PackedMLE, compute_row_batch},
    },
};
use binius_field::{
    BinaryField, BinaryField128b, ExtensionField, Field, PackedExtension, PackedField, TowerField,
};
use binius_ntt::MultithreadedNTT;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::instrument;

#[instrument(skip_all, name = "commit", level = "debug")]
pub fn commit<F, P>(
    mle: &PackedMLE<F>,
    ntt: &MultithreadedNTT<P>,
) -> (FriCommitment, Code<BinaryField128b>, MerkleTree)
where
    BinaryField128b: ExtensionField<F> + ExtensionField<P> + PackedExtension<P>,
    F: BinaryField + TowerField + ExtensionField<P>,
    P: BinaryField + PackedField,
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
pub fn prove<F, P>(
    mle: &PackedMLE<F>,
    eval_point: &[BinaryField128b],
    eval: BinaryField128b,
    encoding: &Code<BinaryField128b>,
    commitment: &FriCommitment,
    merkle_tree: &MerkleTree,
    ntt: &MultithreadedNTT<P>,
    channel: &mut Channel,
) -> Result<EvalProof>
where
    BinaryField128b: ExtensionField<F> + ExtensionField<P> + PackedExtension<P>,
    F: BinaryField + TowerField,
    P: BinaryField,
{
    //The statement should be observed
    channel.observe_fri_commitment(commitment);
    channel.observe_field_elems(eval_point)?;
    channel.observe_field_elem(eval)?;

    let (_, right) = eval_point.split_at(TAU);
    let mut right_eq = LagrangeBases::gen_from_point(right);

    // Get partial evaluations for the binding of the latter variables.
    let upper_partial_evals = get_partial_evals(mle, &right_eq);
    let tensor_batching_point = channel.get_random_points(TAU)?;

    let batching_eq = LagrangeBases::gen_from_point(&tensor_batching_point);

    // After appropriately row-wise batching the ring-switch claims, reinterpret the received MLE as having coefficients in the 128 degree extension.
    let mut repacked_mle = mle.clone().repack_for_fri();

    let mut sum_check_claim = compute_row_batch(&batching_eq.vals, &upper_partial_evals);

    let mut tensored_eq = right_eq.row_batch(&batching_eq);

    let rounds = right.len();

    let mut proof_state = ProofState::default();

    let final_code_folded_value = commit_phase(
        rounds,
        encoding,
        ntt,
        channel,
        &mut proof_state,
        &mut repacked_mle,
        &mut tensored_eq,
        &mut sum_check_claim,
    )?;

    channel.observe_field_elem(final_code_folded_value)?;

    let (round_queried_symbols, round_merkle_paths) =
        query_phase(rounds, encoding, merkle_tree, channel, &proof_state)?;

    Ok(EvalProof::new(
        upper_partial_evals,
        proof_state,
        final_code_folded_value,
        round_queried_symbols,
        round_merkle_paths,
    ))
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
    fn new(
        upper_partial_evals: Vec<BinaryField128b>,
        proof_state: ProofState,
        final_folded_value: BinaryField128b,
        fri_queried_symbols: Vec<Vec<(BinaryField128b, BinaryField128b)>>,
        fri_merkle_paths: Vec<Vec<Vec<Hash>>>,
    ) -> EvalProof {
        EvalProof {
            upper_partial_evals,
            sum_check_oracles: proof_state.sum_check_oracles,
            fri_oracles: proof_state.fri_oracles,
            final_folded_value,
            fri_queried_symbols,
            fri_merkle_paths,
        }
    }
}

pub struct FriCommitment {
    pub vector_commitment: VectorCommitment,
    pub packing_factor: usize,
}

#[derive(Default)]
struct ProofState {
    fri_folded_codes: Vec<Code<BinaryField128b>>,
    fri_oracles: Vec<VectorCommitment>,
    fri_merkle_trees: Vec<MerkleTree>,
    sum_check_oracles: Vec<Univariate>,
    random_challenges: Vec<BinaryField128b>,
}

impl ProofState {
    fn update(
        &mut self,
        folded_code: Code<BinaryField128b>,
        commitment: VectorCommitment,
        merkle_tree: MerkleTree,
        poly: Univariate,
        challenge: BinaryField128b,
    ) {
        self.fri_folded_codes.push(folded_code);
        self.fri_oracles.push(commitment);
        self.fri_merkle_trees.push(merkle_tree);
        self.sum_check_oracles.push(poly);
        self.random_challenges.push(challenge);
    }
}

fn commit_phase<P>(
    rounds: usize,
    encoding: &Code<BinaryField128b>,
    ntt: &MultithreadedNTT<P>,
    channel: &mut Channel,
    proof_state: &mut ProofState,
    repacked_mle: &mut PackedMLE<BinaryField128b>,
    tensored_eq: &mut LagrangeBases,
    sum_check_claim: &mut BinaryField128b,
) -> Result<BinaryField128b>
where
    BinaryField128b: ExtensionField<P> + PackedExtension<P>,
    P: BinaryField,
{
    for round in 0..rounds {
        //Sum check Logic
        let poly = sum_check_round(repacked_mle, tensored_eq, *sum_check_claim);

        channel.observe_field_elems(&poly.coeffs)?;

        let r = channel.get_random_point()?;

        *sum_check_claim = poly.evaluate(r);

        //Folding the code with the sum check challenges.
        let folded_code: Code<BinaryField128b> = match round {
            0 => encoding.fold_code(r, round, ntt),
            _ => proof_state.fri_folded_codes[round - 1].fold_code(r, round, ntt),
        };

        let (commitment, merkle_tree) = commit_oracle(&folded_code);

        channel.observe_vector_commitment(&commitment);

        proof_state.update(folded_code, commitment, merkle_tree, poly, r);
        *repacked_mle = repacked_mle.fold_lo(&r);
        tensored_eq.fold_lo(&r);
    }

    Ok(proof_state.fri_folded_codes[rounds - 1].idx(0))
}

fn query_phase(
    rounds: usize,
    encoding: &Code<BinaryField128b>,
    merkle_tree: &MerkleTree,
    channel: &mut Channel,
    proof_state: &ProofState,
) -> Result<(
    Vec<Vec<(BinaryField128b, BinaryField128b)>>,
    Vec<Vec<Vec<Hash>>>,
)> {
    let mut current_queries: Vec<usize> = channel
        .gen_queries(rounds + LOG_RATE)?
        .iter()
        .map(|i| i >> 1)
        .collect();

    let mut round_merkle_paths: Vec<Vec<Vec<Hash>>> = Vec::with_capacity(rounds);
    let mut round_queried_symbols: Vec<Vec<(BinaryField128b, BinaryField128b)>> =
        Vec::with_capacity(rounds);

    for round in 0..rounds {
        let (tree, code) = match round {
            0 => (merkle_tree, encoding),
            _ => (
                &proof_state.fri_merkle_trees[round - 1],
                &proof_state.fri_folded_codes[round - 1],
            ),
        };

        let (merkle_paths, queried_symbols) = gather_round_queries(tree, code, &current_queries);

        round_merkle_paths.push(merkle_paths);
        round_queried_symbols.push(queried_symbols);

        current_queries.iter_mut().for_each(|i| {
            *i >>= 1;
        });
    }

    Ok((round_queried_symbols, round_merkle_paths))
}

fn gather_round_queries(
    tree: &MerkleTree,
    code: &Code<BinaryField128b>,
    queries: &[usize],
) -> (Vec<Vec<Hash>>, Vec<(BinaryField128b, BinaryField128b)>) {
    let merkle_paths = queries.iter().map(|i| tree.get_merkle_path(*i)).collect();

    let queried_symbols = queries
        .iter()
        .map(|i| (code.encoding[i << 1], code.encoding[(i << 1) | 1]))
        .collect();

    (merkle_paths, queried_symbols)
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
pub fn sum_check_round(
    mle: &PackedMLE<BinaryField128b>,
    eq: &LagrangeBases,
    sum_check_claim: BinaryField128b,
) -> Univariate {
    let half_size = mle.len() / 2;
    let (eval_at_0, eval_at_inf) = (0..half_size)
        .into_par_iter()
        .map(|i| {
            (
                mle.idx(i << 1) * eq.idx(i << 1),
                (mle.idx(i << 1) + mle.idx((i << 1) | 1)) * (eq.idx(i << 1) + eq.idx((i << 1) | 1)),
            )
        })
        .reduce(
            || (BinaryField128b::ZERO, BinaryField128b::ZERO),
            |(acc_0, acc_1), (e_0, e_1)| (acc_0 + e_0, acc_1 + e_1),
        );

    let eval_at_1 = sum_check_claim - eval_at_0;
    Univariate::new(vec![
        eval_at_0,
        eval_at_0 + eval_at_1 + eval_at_inf,
        eval_at_inf,
    ])
}
