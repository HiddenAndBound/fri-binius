

pub fn verify(
    eval: BinaryField128b,
    point: &Vec<BinaryField128b>,
    commit: &Commitment,
    eval_proof: EvalProof,
    channel: &mut Channel,
) {
    let initial_evals = eval_proof.eval_claim;

    assert_eq!(initial_evals.len(), 1 << commit.packing_factor);
    let packing_eq = LagrangeBases::gen_from_point(&point[..commit.packing_factor].to_vec());

    assert_eq!(
        eval,
        linear_combination(&initial_evals, &packing_eq.vals),
        "Initial claim is incorrect"
    );

    let mut initial_tensor_eval = [BinaryField128b::ZERO; 128];

    for i in 0..64 {
        initial_tensor_eval[i] = initial_evals[i]
    }

    let mut current_sum = PackedAlgebra128::new(initial_tensor_eval);

    let rounds = commit.vars - commit.packing_factor;
    let norms = compute_norms(rounds);
    let mut random_challenges = Vec::new();
    for i in 0..rounds {
        let poly = &eval_proof.sumcheck_polys[i];
        assert_eq!(poly.len(), 2, "Polynomial is not of claimed degree");
        assert_eq!(
            current_sum,
            poly[0] + (poly[0] + poly[1]).scalar_mul(&point[commit.packing_factor + i]),
            "Sum check verification failed at round {i}"
        );

        channel.reseed(&PackedAlgebra128::unpack(poly.clone()));

        let r = channel.get_random_point();

        current_sum = poly[0] + (poly[0] + poly[1]) * r;

        random_challenges.push(r);
    }

    assert_eq!(
        current_sum, eval_proof.final_sum_check_eval,
        "Final Sum check claim not matching."
    );

    let mut current_queries: Vec<usize> =
        gen_queries((1 << (commit.vars - commit.packing_factor)) * RATE, channel)
            .iter()
            .map(|i| i >> 1)
            .collect();

    let mut folded_symbols: Vec<BinaryField128b> =
        vec![BinaryField128b::ZERO; current_queries.len()];

    let mut code = eval_proof.early_stop;
    println!("code length {:?}", code.0.len());
    for round in 0..rounds {
        if round == 0 && round + 7 <= rounds{
            let queried_locations = &eval_proof.round_queried_symbols[0];

            let query_merkle_paths = &eval_proof.round_queried_merkle_path[0];

            for query_idx in 0..current_queries.len() {
                let (left_hash, right_hash) = (
                    hash_128(&queried_locations[query_idx].0),
                    hash_128(&queried_locations[query_idx].1),
                );
                verify_merkle_path(
                    &commit.commit,
                    left_hash,
                    current_queries[query_idx] << 1,
                    &query_merkle_paths[query_idx].0,
                );
                verify_merkle_path(
                    &commit.commit,
                    right_hash,
                    (current_queries[query_idx] << 1) | 1,
                    &query_merkle_paths[query_idx].1,
                );

                let query_folded_symbol = fold(
                    random_challenges[0],
                    round,
                    current_queries[query_idx],
                    queried_locations[query_idx].0,
                    queried_locations[query_idx].1,
                    &norms,
                );
                folded_symbols[query_idx] = query_folded_symbol;
            }

            if round + 7 == rounds{
                for query_idx in 0..current_queries.len(){
                    assert_eq!(folded_symbols[query_idx], code.0[current_queries[query_idx]])
                }
            }
    
        } else if round > 0 && round + 7 <= rounds{
            let queried_locations = &eval_proof.round_queried_symbols[round];
            let query_merkle_paths = &eval_proof.round_queried_merkle_path[round];

            let oracle_commit = &eval_proof.oracle_commitments[round - 1];

            for query_idx in 0..current_queries.len() {
                if current_queries[query_idx] & 1 == 1 {
                    assert_eq!(
                        folded_symbols[query_idx], queried_locations[query_idx].1,
                        "Symbol not consistent at query {query_idx}"
                    );
                } else {
                    assert_eq!(
                        folded_symbols[query_idx], queried_locations[query_idx].0,
                        "Symbol not consistent at query {query_idx}"
                    );
                }

                current_queries[query_idx] >>= 1;

                let (left_hash, right_hash) = (
                    hash_128(&queried_locations[query_idx].0),
                    hash_128(&queried_locations[query_idx].1),
                );
                let query_folded_symbol = fold(
                    random_challenges[round],
                    round,
                    current_queries[query_idx],
                    queried_locations[query_idx].0,
                    queried_locations[query_idx].1,
                    &norms,
                );
                folded_symbols[query_idx] = query_folded_symbol;
            }
            if round + 7 == rounds{
                for query_idx in 0..current_queries.len(){
                    assert_eq!(folded_symbols[query_idx], code.0[current_queries[query_idx]])
                }
            }
        } else{
            code = fold_code(random_challenges[round], round, &code, &norms);
        }

    }
    for elem in &code.0{
        assert_eq!(eval_proof.final_sum_check_eval, PackedAlgebra128::from(elem));
    }
}