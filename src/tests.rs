use std::time::Instant;

use binius_field::{ BinaryField128b, BinaryField64b, Field };
use binius_ntt::{ MultithreadedNTT, SingleThreadedNTT };
use rand::thread_rng;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tracing_profile::init_tracing;

use crate::{
    prover::{ commit, prove },
    utils::{ channel::Channel, code::LOG_RATE, mle::{ self, LagrangeBases, PackedMLE } },
    verifier::verify,
};

#[test]
fn fri_test() {
    init_tracing().expect("failed to initialize tracing");
    for l in 10..40 {
        println!("--------------|| length 2^{:?} ||-------------- \n\n", l + 6);
        let time = Instant::now();
        let poly: Vec<BinaryField64b> = (0..1 << l)
            .into_par_iter()
            .map(|_| BinaryField64b::random(thread_rng()))
            .collect();
        println!("Time: {:?} \n", time.elapsed());

        let poly = PackedMLE::new(poly, true);
        println!("Making ntt precomputations");
        let time = Instant::now();
        let ntt = SingleThreadedNTT::<BinaryField128b>
            ::new(l + LOG_RATE)
            .unwrap()
            .multithreaded();
        println!("Time: {:?} \n", time.elapsed());

        println!("Committing");

        let time = Instant::now();
        let (commitment, encoded_poly, merkle_tree) = commit(&poly, &ntt);
        println!("Time: {:?} \n", time.elapsed());

        let point: Vec<BinaryField128b> = (0..l + 6)
            .into_iter()
            .map(|_| BinaryField128b::random(thread_rng()))
            .collect();

        let bases = LagrangeBases::gen_from_point(&point);
        // println!("{:?}", base);
        let eval = poly.get_bound_elem(0, &bases);

        let mut channel = Channel::new();

        println!("Generating proof");
        let time = Instant::now();

        let eval_proof = prove(
            &poly,
            &point,
            eval,
            &encoded_poly,
            &commitment,
            &merkle_tree,
            &ntt,
            &mut channel
        );
        println!("Time: {:?}\n", time.elapsed());

        let time = Instant::now();

        println!("Verifying");

        let mut channel = Channel::new();

        verify(&commitment, &point, eval, eval_proof, &ntt, &mut channel);
        println!("Time: {:?} \n", time.elapsed());
    }
}
