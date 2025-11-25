use binius_field::{BinaryField, BinaryField128b, ExtensionField, PackedExtension, TowerField};
use binius_ntt::{AdditiveNTT, MultithreadedNTT};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

pub const RATE: usize = 4;
pub const LOG_RATE: usize = 2;
//Struct containing the Reed-Solomon encoding of a message of packed elements. We assume the elements of the message contain packed base field elements.
#[derive(Clone, Debug, Default)]
pub struct Code<F: BinaryField> {
    pub encoding: Vec<F>,
}

impl Code<BinaryField128b> {
    fn repack_message<F>(message: &[F]) -> Vec<BinaryField128b>
    where
        BinaryField128b: ExtensionField<F>,
        F: BinaryField + TowerField,
    {
        message
            .par_chunks(<BinaryField128b as ExtensionField<F>>::DEGREE)
            .map(|base_elems| {
                BinaryField128b::from_bases(base_elems).expect("failed to repack base elements")
            })
            .collect()
    }

    fn encode_with_transform<F, N, T>(
        message: &[F],
        ntt: &MultithreadedNTT<N>,
        mut transform: T,
    ) -> Code<BinaryField128b>
    where
        BinaryField128b: ExtensionField<F>,
        F: BinaryField + TowerField,
        N: BinaryField,
        T: FnMut(&MultithreadedNTT<N>, &mut Vec<BinaryField128b>, u32),
    {
        let repacked_message = Self::repack_message(message);
        let mut encoding = Vec::with_capacity(repacked_message.len() * RATE);

        for round in 0..RATE as u32 {
            let mut temp = repacked_message.clone();
            transform(ntt, &mut temp, round);
            encoding.append(&mut temp);
        }

        Code { encoding }
    }

    pub fn new<F>(message: &[F], ntt: &MultithreadedNTT<BinaryField128b>) -> Code<BinaryField128b>
    where
        BinaryField128b: ExtensionField<F>,
        F: BinaryField + TowerField,
    {
        Self::encode_with_transform(message, ntt, |ntt, temp, round| {
            ntt.forward_transform(temp, round, 0)
                .expect("NTT forward transform failed")
        })
    }

    pub fn new_ext<F, P>(message: &[F], ntt: &MultithreadedNTT<P>) -> Code<BinaryField128b>
    where
        BinaryField128b: ExtensionField<F> + ExtensionField<P> + PackedExtension<P>,
        F: BinaryField + TowerField + ExtensionField<P>,
        P: BinaryField,
    {
        Self::encode_with_transform(message, ntt, |ntt, temp, round| {
            ntt.forward_transform_ext::<BinaryField128b>(temp, round)
                .expect("extended NTT forward transform failed");
        })
    }

    pub fn fold_code<P>(
        &self,
        r: BinaryField128b, //folding challenge
        round: usize,
        ntt: &MultithreadedNTT<P>,
    ) -> Code<BinaryField128b>
    where
        BinaryField128b: ExtensionField<P>,
        P: BinaryField,
    {
        let encoding: Vec<BinaryField128b> = self
            .encoding
            .par_chunks_exact(2)
            .enumerate()
            .map(|(i, pair)| fold(r, round, i, pair[0], pair[1], ntt))
            .collect();

        Code { encoding }
    }

    pub fn idx(&self, idx: usize) -> BinaryField128b {
        self.encoding[idx]
    }
}

#[inline(always)]
pub fn fold<P>(
    r: BinaryField128b,
    round: usize,
    idx: usize,
    val0: BinaryField128b,
    val1: BinaryField128b,
    ntt: &MultithreadedNTT<P>,
) -> BinaryField128b
where
    BinaryField128b: ExtensionField<P>,
    P: BinaryField,
{
    //twiddle for inverse ntt component of the fold i.e the twiddle for the butterfly unit
    //at the index if we were to apply the inverse ntt

    let twiddle = ntt.get_subspace_eval(round, idx);
    let (mut x0, mut x1) = (val0, val1);
    x1 += x0;
    x0 += x1 * twiddle;

    x0 + r * (x0 + x1)
}

#[cfg(test)]
mod tests {
    use crate::utils::mle::LagrangeBases;

    use super::*;
    use binius_field::Field;
    use binius_ntt::SingleThreadedNTT;
    use rand::thread_rng;
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
    #[test]
    fn test_fold() {
        let l = 11;
        let poly: Vec<BinaryField128b> = (0..1 << l)
            .into_par_iter()
            .map(|_| BinaryField128b::random(thread_rng()))
            .collect();

        let ntt = SingleThreadedNTT::<BinaryField128b>::new(l + 2)
            .unwrap()
            .multithreaded();

        let code = Code::new(&poly, &ntt);

        let r: Vec<BinaryField128b> = (0..l)
            .into_iter()
            .map(|_| BinaryField128b::random(thread_rng()))
            .collect();

        let r_eq = LagrangeBases::gen_from_point(&r);

        let poly_eval: BinaryField128b = poly
            .par_iter()
            .zip(r_eq.vals)
            .map(|(coeff, eq_val)| *coeff * eq_val)
            .sum();

        let mut folded_code = code.fold_code(r[0], 0, &ntt);
        for round in 1..l {
            folded_code = folded_code.fold_code(r[round], round, &ntt)
        }
        assert_eq!(poly_eval, folded_code.idx(0));
        assert!(
            !folded_code.encoding.is_empty(),
            "folding produced empty code"
        );
    }

    #[test]
    fn test_ntt() {
        let l = 11;
        let mut poly = (0..1 << l)
            .into_par_iter()
            .map(|_| BinaryField128b::random(thread_rng()))
            .collect::<Vec<_>>();

        let ntt = SingleThreadedNTT::<BinaryField128b>::new(13)
            .unwrap()
            .multithreaded();

        ntt.forward_transform_ext(&mut poly, 0)
            .expect("extended forward transform failed");
    }
}
