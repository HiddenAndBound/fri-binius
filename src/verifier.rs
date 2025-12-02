use anyhow::{Ok, ensure};
use binius_field::{BinaryField, BinaryField128b, ExtensionField, Field};
use binius_ntt::MultithreadedNTT;
use itertools::multizip;
use tracing::instrument;

use crate::{
    Result,
    prover::{EvalProof, FriCommitment},
    utils::{
        TAU,
        channel::Channel,
        code::{LOG_RATE, fold},
        merkle::{hash_tuple, verify_merkle_path},
        mle::{compute_row_batch, switch_view},
    },
};
#[instrument(skip_all, name = "verify", level = "debug")]
pub fn verify<P>(
    commitment: &FriCommitment,
    eval_point: &[BinaryField128b],
    eval: BinaryField128b,
    eval_proof: EvalProof,
    ntt: &MultithreadedNTT<P>,
    channel: &mut Channel,
) -> Result<()>
where
    BinaryField128b: ExtensionField<P>,
    P: BinaryField,
{
    channel.observe_fri_commitment(commitment);
    channel.observe_field_elems(eval_point)?;
    channel.observe_field_elem(eval)?;

    let (left, right) = eval_point.split_at(TAU);

    let left_eq = compute_eq_table(left);

    let mut derived_eval = BinaryField128b::ZERO;

    for i in 0..1 << TAU {
        derived_eval += left_eq[i] * eval_proof.upper_partial_evals[i];
    }

    ensure!(derived_eval == eval);

    let tensor_batching_point = channel.get_random_points(TAU)?;

    let batching_eq = compute_eq_table(&tensor_batching_point);
    let mut sum_check_claim = compute_row_batch(&batching_eq, &eval_proof.upper_partial_evals);

    let rounds = right.len();

    ensure!(rounds == eval_proof.sum_check_oracles.len());

    let mut random_point = Vec::new();
    for round in 0..rounds {
        let oracle = &eval_proof.sum_check_oracles[round];

        ensure!(
            oracle.evaluate(BinaryField128b::ZERO) + oracle.evaluate(BinaryField128b::ONE)
                == sum_check_claim,
            "Sum of oracle evaluations failed on round {round}"
        );

        channel.observe_field_elems(&oracle.coeffs)?;

        let r = channel.get_random_point()?;

        let current_oracle = &eval_proof.fri_oracles[round];
        channel.observe_vector_commitment(current_oracle);
        sum_check_claim = oracle.evaluate(r);
        random_point.push(r);
    }

    channel.observe_field_elem(eval_proof.final_folded_value)?;

    let mut current_queries: Vec<usize> = channel
        .gen_queries(right.len() + LOG_RATE)?
        .iter()
        .map(|i| i >> 1)
        .collect();

    
    let mut folded_symbols = Vec::new();

    for round in 0..rounds {
        // Choose the commitment: root for round 0, previous oracle thereafter.
        let oracle = match round {
            0 => &commitment.vector_commitment,
            _ => &eval_proof.fri_oracles[round - 1],
        };

        folded_symbols = multizip((
            current_queries.iter_mut(),             // queries we mutate in-place
            &eval_proof.fri_queried_symbols[round], // (s0, s1) pairs
            &eval_proof.fri_merkle_paths[round],    // Merkle paths
        ))
        .enumerate()
        .map(|(i, (query, &(s0, s1), merkle_path))| {
            let hash = hash_tuple(&(s0, s1));

            match round {
                // First round: no consistency check yet.
                0 => (),
                // Later rounds: check consistency, then step up the tree.
                _ => {
                    let expected = match *query & 1 {
                        1 => s1,
                        _ => s0,
                    };
                    ensure!(
                        folded_symbols[i] == expected,
                        "Symbol not consistent at query {i} in round {round}"
                    );
                    *query >>= 1; // move to parent index for next round
                }
            }

            // Membership proof against the chosen oracle
            verify_merkle_path(oracle, hash, *query, merkle_path)?;

            // Fold this pair for use in the next round
            Ok(fold(random_point[round], round, *query, s0, s1, ntt))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    }

    for symbol in folded_symbols {
        assert_eq!(symbol, eval_proof.final_folded_value)
    }

    Ok(())
}

pub fn compute_eq_table(r: &[BinaryField128b]) -> Vec<BinaryField128b> {
    //Initialize eq with (1- r[0]) and r[0]
    let mut eq = [BinaryField128b::ONE - r[0], r[0]].to_vec();
    //Iterate over the length of the r vector
    for k in 1..r.len() {
        let temp = eq;
        //initialize table of double size with zero

        eq = vec![BinaryField128b::ZERO; temp.len() * 2];

        for i in 0..temp.len() {
            eq[i | (1 << k)] = temp[i] * r[k];
            eq[i] = temp[i] - eq[i | (1 << k)];
        }
    }
    eq
}

pub fn compute_eq_tower_ind(
    r_init: &[BinaryField128b],
    r_sum: &[BinaryField128b],
    eq_batch: &[BinaryField128b],
) -> BinaryField128b {
    assert_eq!(r_init.len(), r_sum.len());

    let mut eval = vec![BinaryField128b::ZERO; 128];

    eval[0] = BinaryField128b::ONE;

    for i in 0..r_init.len() {
        for _ in 0..128 {
            let temp = eval[i] * r_init[i];
            eval[i] += temp
        }

        eval = switch_view(&eval);

        for _ in 0..128 {
            let temp = eval[i] * r_sum[i];
            eval[i] += temp
        }

        eval = switch_view(&eval);
    }

    compute_row_batch(eq_batch, &eval)
}
