use binius_field::{ BinaryField, BinaryField1b, BinaryField128b, ExtensionField, Field };
use binius_ntt::{ AdditiveNTT, Error, MultithreadedNTT };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

pub const RATE: usize = 4;

//Struct containing the Reed-Solomon encoding of a message of packed elements. We assume the elements of the message contain packed base field elements.
pub struct Code<F: BinaryField> {
    encoding: Vec<F>,
}

impl<F: BinaryField> Code<F> {
    pub fn new(message: &Vec<F>, ntt: &MultithreadedNTT<F>) -> Result<Code<F>, Error> {
        //Padding the message and transforming it is better asymptotically.
        let mut encoding = vec![F::ZERO; message.len() * RATE];
        encoding[..message.len()].copy_from_slice(&message);

        match ntt.forward_transform(&mut encoding, 0, 0) {
            Ok(()) => Ok(Code { encoding }),
            Err(error) => Err(error),
        }
    }
}

//Folded codes once folded will have symbols in top level field as challenges are sampled from there.
impl Code<BinaryField128b> {
    //ntt taken as argument in order to compute necessary twiddle factor for the fold.
    pub fn fold_code<F>(
        r: BinaryField128b, //folding challenge
        round: usize,
        code: &Code<F>,
        ntt: &MultithreadedNTT<F>
    )
        -> Code<BinaryField128b>
        where F: BinaryField, BinaryField128b: ExtensionField<F>
    {
        let encoding: Vec<BinaryField128b> = (0..code.encoding.len() >> 1)
            .into_par_iter()
            .map(|i| { fold(r, round, i, code.encoding[i << 1], code.encoding[(i << 1) | 1], ntt) })
            .collect();
        Code { encoding }
    }
}

#[inline(always)]
pub fn fold<F>(
    r: BinaryField128b,
    round: usize,
    idx: usize,
    val0: F,
    val1: F,
    ntt: &MultithreadedNTT<F>
)
    -> BinaryField128b
    where F: BinaryField, BinaryField128b: ExtensionField<F>
{
    let twiddle = ntt.get_subspace_eval(round, idx);
    let (mut x0, mut x1) = (val0, val1);
    x1 += x0;
    x0 += x1 * twiddle;

    (BinaryField1b::ONE + r) * x0 + r * x1
}
