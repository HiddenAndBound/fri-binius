use binius_field::{BinaryField32b, BinaryField64b, BinaryField128b, Field};
use binius_ntt::SingleThreadedNTT;
use rand::thread_rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing_profile::init_tracing;

use fri_binius::{
    Result,
    prover::{commit, prove},
    utils::{
        channel::Channel,
        code::LOG_RATE,
        mle::{LagrangeBases, PackedMLE},
    },
    verifier::verify,
};

fn main() -> Result<()> {
    init_tracing().expect("failed to initialize tracing");
    for l in 10..40 {
        println!(
            "--------------|| length 2^{:?} ||-------------- \n\n",
            l + 6
        );
        let poly: Vec<BinaryField64b> = (0..1 << l)
            .into_par_iter()
            .map(|_| BinaryField64b::random(thread_rng()))
            .collect();

        let poly = PackedMLE::new(poly, true);

        let ntt = SingleThreadedNTT::<BinaryField32b>::new(l + LOG_RATE)
            .unwrap()
            .multithreaded();

        let (commitment, encoded_poly, merkle_tree) =
            commit::<BinaryField64b, BinaryField32b>(&poly, &ntt);

        let point: Vec<BinaryField128b> = (0..l + 6)
            .into_iter()
            .map(|_| BinaryField128b::random(thread_rng()))
            .collect();

        let bases = LagrangeBases::gen_from_point(&point);
        let eval = poly.get_bound_elem(0, &bases);

        let mut channel = Channel::new();

        let eval_proof = prove(
            &poly,
            &point,
            eval,
            &encoded_poly,
            &commitment,
            &merkle_tree,
            &ntt,
            &mut channel,
        )?;

        let mut channel = Channel::new();

        verify(&commitment, &point, eval, eval_proof, &ntt, &mut channel)?;
    }

    Ok(())
}
