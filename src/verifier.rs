use binius_field::{ BinaryField, BinaryField128b, ExtensionField, Field, TowerField };
use binius_ntt::MultithreadedNTT;

use crate::{
    prover::{ EvalProof, FriCommitment },
    utils::{
        channel::{ self, Channel },
        code::{ fold, LOG_RATE },
        merkle::{ hash_field, hash_tuple, verify_merkle_path, VectorCommitment },
        mle::LagrangeBases, TAU,
    },
};

//The generic parameter defines the packing factor.
pub fn verify(
    commitment: &FriCommitment,
    eval_point: &[BinaryField128b],
    eval: BinaryField128b,
    eval_proof: EvalProof,
    ntt: &MultithreadedNTT<BinaryField128b>,
    channel: &mut Channel
)
{
    channel.observe_fri_commitment(commitment);
    channel.observe_field_elems(eval_point).expect("failed to observe eval_point");
    channel.observe_field_elem(eval).expect("failed to observe eval");

    let (left, right) = eval_point.split_at(TAU);

    let left_eq = compute_eq_table(left);

    let mut derived_eval = BinaryField128b::ZERO;

    for i in 0..1 << TAU {
        derived_eval += left_eq[i] * eval_proof.upper_partial_evals[i];
    }

    assert_eq!(derived_eval, eval);

    let tensor_batching_point = channel
        .get_random_points(TAU)
        .expect("unable to sample random point for tensor batching");

    let batching_eq = compute_eq_table(&tensor_batching_point);
    let mut sum_check_claim = BinaryField128b::ZERO;

    for i in 0..1 << TAU {
        sum_check_claim += batching_eq[i] * eval_proof.upper_partial_evals[i];
    }

    let rounds = right.len();

    assert_eq!(rounds, eval_proof.sum_check_oracles.len());

    let mut random_point = Vec::new();
    for round in 0..rounds {
        let oracle = eval_proof.sum_check_oracles[round];

        assert_eq!(
             oracle.evaluate(BinaryField128b::ZERO) + oracle.evaluate(BinaryField128b::ONE),
            sum_check_claim,
            "Sum of oracle evaluations failed on round {round}"
        );

        channel.observe_field_elems(&oracle.coeffs).unwrap();

        let r = channel.get_random_point().unwrap();

        let current_oracle = &eval_proof.fri_oracles[round];
        channel.observe_vector_commitment(current_oracle);
        sum_check_claim = oracle.evaluate(r);
        random_point.push(r);
    }

    channel
        .observe_field_elem(eval_proof.final_folded_value)
        .expect("Failed to observe final sum check claim");

    let mut current_queries: Vec<usize> = channel
        .gen_queries(right.len() + LOG_RATE)
        .expect("Failed to generate FRI queries.")
        .iter()
        .map(|i| i >> 1)
        .collect();

    let mut folded_symbols = Vec::new();
    for round in 0..rounds {
        let round_folded_symbols = match round {
            0 => {
                (0..current_queries.len())
                    .into_iter()
                    .map(|i| {
                        let queried_symbols = eval_proof.fri_queried_symbols[round][i];
                        let merkle_path = &eval_proof.fri_merkle_paths[round][i];

                        let hash = hash_tuple(&queried_symbols);
                        verify_merkle_path(
                            &commitment.vector_commitment,
                            hash,
                            current_queries[i],
                            &merkle_path
                        );
                        fold(
                            random_point[round],
                            round,
                            current_queries[i],
                            queried_symbols.0,
                            queried_symbols.1,
                            ntt
                        )
                    })
                    .collect()
            }
            _ => {
                (0..current_queries.len())
                    .into_iter()
                    .map(|i| {
                        let queried_symbols = eval_proof.fri_queried_symbols[round][i];
                        let merkle_path = &eval_proof.fri_merkle_paths[round][i];
                        let hash = hash_tuple(&queried_symbols);
                    
                        if current_queries[i] & 1 == 1 {
                            assert_eq!(
                                folded_symbols[i], queried_symbols.1,
                                "Symbol not consistent at query {i} in round {round}"
                            );
                        } else {
                            assert_eq!(
                                folded_symbols[i], queried_symbols.0,
                                "Symbol not consistent at query {i} in round {round}"
                            );
                        }

                        current_queries[i] >>= 1;

                        verify_merkle_path(
                            &eval_proof.fri_oracles[round - 1],
                            hash,
                            current_queries[i],
                            &merkle_path
                        );

                        fold(
                            random_point[round],
                            round,
                            current_queries[i],
                            queried_symbols.0,
                            queried_symbols.1,
                            ntt
                        )
                        
                        

                    })
                    .collect()
            }
        };

       
        folded_symbols = round_folded_symbols
    }
    for symbol in folded_symbols{
        assert_eq!(symbol, eval_proof.final_folded_value)
    }
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
