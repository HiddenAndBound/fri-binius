use std::{cell::UnsafeCell, cmp::max, thread::{self, available_parallelism}};

use binius_field::BinaryField128b;
use tracing::instrument;

#[instrument(skip_all, name = "commit_oracle", level = "debug")]
pub fn commit_oracle(code: &Code<BinaryField128b>) -> (VectorCommitment, MerkleTree) {
    let mut leaf_hashes: Vec<Hash> = vec![hash(&code.0[0].0.to_le_bytes()); code.0.len()];
    
    code
        .0
        .par_iter()
        .map(|val| hash(&val.0.to_le_bytes()))
        .collect_into_vec(&mut leaf_hashes);

    let merkle_tree = merklize(leaf_hashes);

    let merkle_root = merkle_tree.get_root();

    (
        VectorCommitment::new(merkle_root, code.0.len()),
        merkle_tree,
    )
}

#[instrument(skip_all, name = "commit", level = "debug")]
pub fn commit(mle: &MLE<BinaryField64b>) -> (Commitment, MerkleTree, Code<BinaryField64b>) {
    let encoding = encode(&mle.poly);

    let mut leaf_hashes: Vec<Hash> = vec![hash_128(&(encoding[0].into())); encoding.len()];
    
    encoding
        .par_iter()
        .copied()
        .map(|val| hash_128(&(val.into())))
        .collect_into_vec(&mut leaf_hashes);


    let merkle_tree = merklize(leaf_hashes);

    let merkle_root = merkle_tree.get_root();

    (
        Commitment::new(merkle_root, mle.variables, mle.packing_factor),
        merkle_tree,
        Code(encoding),
    )
}




#[instrument(skip_all, level = "debug")]
pub fn eval_proof<F>(
    mle: &MLE<F>,
    point: &Vec<BinaryField128b>,
    encoding: &Code<F>,
    merkle_tree: &MerkleTree,
    channel: &mut Channel,
) -> EvalProof 
where F:BinaryField
+ TowerField
+ Mul<BinaryField128b, Output = BinaryField128b>
+ Add<BinaryField128b, Output = BinaryField128b>
+ Copy
+ Mul<BinaryField32b, Output = F>
+ AddAssign<<F as Mul<BinaryField32b>>::Output>,
BinaryField128b: ExtensionField<F>
{
    let rounds = mle.variables - mle.packing_factor;
    let norms = compute_norms(rounds);
    let mut current_encoding = Code(Vec::new());
    let mut current_merkle_tree = merkle_tree.clone();

    let mut folded_handle = MLE::new(
        mle.poly
            .par_iter()
            .map(|coeff| BinaryField128b::from(*coeff))
            .collect(),
        true,
    );

    let mut oracle_commitments = Vec::new();
    let mut oracle_merkle_trees = Vec::new();
    let mut oracles = Vec::new();

    let threads = match available_parallelism() {
        Ok(n_threads) => n_threads.get(),
        Err(err) => panic!("{:?}", err),
    };

    let mut round_merkle_paths = Vec::new();
    let mut round_queried_locations = Vec::new();
    let mut polys: Vec<Vec<PackedAlgebra128>> = Vec::new();

    let mut eq = LagrangeBases::gen_from_point(&point[mle.packing_factor..].to_vec());

    let fold = mle.fold_as_unpacked_hi(&eq);

    let mut evals = [BinaryField128b::ZERO; 128];

    for i in 0..F::N_BITS {
        evals[i] = fold.idx(i)
    }

    let mut current_sum = PackedAlgebra128::new(evals);

    eq.fold_in();

    //Tensor sum check
    for round in 0..rounds {

        let halfsize = 1 << (rounds - round - 1);

        let chunk_size = max(halfsize / threads, 1);
        let chunks = halfsize / chunk_size;
        let handle_cell = UnsafeCell::new(&folded_handle);
        let eq_cell = UnsafeCell::new(&eq);
        let current_round_point = &point[mle.packing_factor + round];

        let mut eval_acc = tensor_sum_check_round(round, rounds, &mle, &folded_handle, &eq);

        let eval_0 = PackedAlgebra128(eval_acc);
        let eval_1 = (current_sum
            - eval_0.scalar_mul(&(BinaryField128b::ONE + current_round_point)))
        .scalar_mul(&current_round_point.invert().unwrap());

        let coeffs = vec![eval_0, eval_1];

        channel.reseed(&PackedAlgebra128::unpack(coeffs.clone()));

        let r = channel.get_random_point();
        
        let fold_coeffs = MLE::<BinaryField128b>::fold(&folded_handle, &r);
        folded_handle = MLE::new(fold_coeffs.poly, true);

        if round == 0 && round + 7 <= rounds{
            current_encoding = fold_code(r, round, &encoding, &norms);

            let (current_commitment, current_merkle_tree) = commit_oracle(&current_encoding);

            oracles.push(current_encoding.clone());
            oracle_commitments.push(current_commitment);
            oracle_merkle_trees.push(current_merkle_tree);
        } else if round > 0 && round + 7 <= rounds{
            current_encoding = fold_code::<BinaryField128b>(r, round, &current_encoding, &norms);
            let (current_commitment, current_merkle_tree) = commit_oracle(&current_encoding);

            oracles.push(current_encoding.clone());
            oracle_commitments.push(current_commitment);
            oracle_merkle_trees.push(current_merkle_tree);
        }

        eq.fold_in();

        current_sum = coeffs[0] + (coeffs[0] + coeffs[1]) * r;

        polys.push(coeffs);
    }


    let mut current_queries: Vec<usize> = gen_queries(mle.len() * RATE, channel)
        .iter()
        .map(|i| i >> 1)
        .collect();

    //FRI
    for round in 0..rounds {
        if round == 0 && round + 7 <= rounds {
            let merkle_paths: Vec<(Vec<Hash>, Vec<Hash>)> = current_queries
                .iter()
                .map(|i| {
                    (
                        merkle_tree.get_merkle_path(*i << 1),
                        merkle_tree.get_merkle_path((*i << 1) | 1),
                    )
                })
                .collect();

            let queried_locations: Vec<(BinaryField128b, BinaryField128b)> = current_queries
                .iter()
                .map(|i| (encoding.0[*i << 1].into(), encoding.0[(*i << 1) | 1].into()))
                .collect();

            round_merkle_paths.push(merkle_paths);
            round_queried_locations.push(queried_locations);
        } else if round > 0 && round + 7 <= rounds {
            let merkle_paths: Vec<(Vec<Hash>, Vec<Hash>)> = current_queries
                .iter()
                .map(|i| {
                    (
                        oracle_merkle_trees[round - 1].get_merkle_path(*i << 1),
                        oracle_merkle_trees[round - 1].get_merkle_path((*i << 1) | 1),
                    )
                })
                .collect();
            let queried_locations: Vec<(BinaryField128b, BinaryField128b)> = current_queries
                .iter()
                .map(|i| {
                    (
                        oracles[round - 1].0[*i << 1].into(),
                        oracles[round - 1].0[(*i << 1) | 1].into(),
                    )
                })
                .collect();

            round_merkle_paths.push(merkle_paths);
            round_queried_locations.push(queried_locations);
        }

        current_queries = current_queries.iter().map(|i| i >> 1).collect();
    }

    println!("code length {:?}", current_encoding.0.len());
    println!("rounds: {rounds}");
    if rounds < 7{
        current_encoding = Code(encoding.0.par_iter().copied().map(|val| BinaryField128b::from(val)).collect());
    }

    EvalProof::new(
        fold.poly,
        polys,
        oracle_commitments,
        PackedAlgebra128::from(&folded_handle.idx(0)),
        round_queried_locations,
        round_merkle_paths,
        current_encoding
    )
}

#[instrument(skip_all, name = "tensor_sum_check_round", level = "debug")]
pub fn tensor_sum_check_round<F>(round: usize, rounds:usize, handle: &MLE<F>, folded_handle: &MLE<BinaryField128b>, eq: &LagrangeBases) -> [BinaryField128b; 128]
where F:BinaryField
+ TowerField
+ Mul<BinaryField128b, Output = BinaryField128b>
+ Add<BinaryField128b, Output = BinaryField128b>
+ Copy
+ Mul<BinaryField32b, Output = F>
+ AddAssign<<F as Mul<BinaryField32b>>::Output>,
BinaryField128b: ExtensionField<F>{
    let halfsize = 1 << (rounds - round - 1);

        let threads = match available_parallelism() {
            Ok(n_threads) => n_threads.get(),
            Err(err) => panic!("{:?}", err),
        };
        let chunk_size = max((halfsize + 1) / threads, 1);
        let chunks = halfsize/ chunk_size;
        let handle_cell = UnsafeCell::new(&folded_handle);
        let eq_cell = UnsafeCell::new(&eq);
        let mut eval_acc = [BinaryField128b::ZERO; 128];

        if round == 0 {
            thread::scope(|scope| {
                let mut thread_results = Vec::new();
                
            
                for chunk in 0..chunks {
                 
                    let handle = unsafe{& *handle_cell.get()};

                    let eq_ref = unsafe{& *eq_cell.get()};

                    
                    thread_results.push(scope.spawn(move || {
                        let mut acc = [BinaryField128b::ZERO; 128];
                        for i in chunk * chunk_size..(chunk + 1) * chunk_size {
                            for b in 0..F::N_BITS {
                                acc[b] += handle.packed_idx((i << 8) | b) * eq_ref.idx(i)
                            }
                        }
                        acc
                    }))
                }

                for res in thread_results {
                    match res.join() {
                        Ok(result) => {
                            for i in 0..F::N_BITS {
                                eval_acc[i] += result[i]
                            }
                        }
                        Err(error) => panic!("{:?}", error),
                    }
                }
            });
        } else {
            thread::scope(|scope| {
                let mut thread_results = Vec::new();

                for chunk in 0..chunks {
                    let handle = unsafe{& *handle_cell.get()};

                    let eq_ref = unsafe{& *eq_cell.get()};
                    thread_results.push(scope.spawn(move || {
                        let mut acc = [BinaryField128b::ZERO; 128];

                        for i in chunk * chunk_size..(chunk + 1) * chunk_size {
                            for b in 0..128 {
                                acc[b] += handle.packed_idx((i << 8) | b) * eq_ref.idx(i)
                            }
                        }
                        acc
                    }))
                }

                for res in thread_results {
                    match res.join() {
                        Ok(result) => {
                            for i in 0..128 {
                                eval_acc[i] += result[i]
                            }
                        }
                        Err(error) => panic!("{:?}", error),
                    }
                }
            });
        }
        eval_acc
}
