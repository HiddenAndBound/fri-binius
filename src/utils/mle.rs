use binius_field::{
    BinaryField, BinaryField1b, BinaryField128b, ExtensionField, Field, TowerField,
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};
use tracing::instrument;

use crate::utils::TAU;

// We use this struct to represent both the case when coefficients are from an extension field but represent packed elements, and when the coefficients of the MLE are truly in the extension field.
#[derive(Clone, Debug, Default)]
pub struct PackedMLE<F>
where
    F: BinaryField + TowerField,
    BinaryField128b: ExtensionField<F>,
{
    pub packing_factor: usize,
    pub variables: usize,
    pub coeffs: Vec<F>,
}

impl<F> PackedMLE<F>
where
    F: BinaryField + TowerField,
    BinaryField128b: ExtensionField<F>,
{
    pub fn new(coeffs: Vec<F>, packed: bool) -> PackedMLE<F> {
        match packed {
            true => PackedMLE {
                packing_factor: F::N_BITS.trailing_zeros() as usize,
                variables: (coeffs.len().trailing_zeros() + F::N_BITS.trailing_zeros()) as usize,
                coeffs,
            },

            false => PackedMLE {
                packing_factor: 0,
                variables: coeffs.len().trailing_zeros() as usize,
                coeffs,
            },
        }
    }

    #[inline(always)]
    //Indexes the vector as an unpacked vector.
    pub fn packed_idx(&self, idx: usize) -> BinaryField1b {
        let out_idx = idx >> self.packing_factor;
        let in_idx = (out_idx << self.packing_factor) ^ idx;
        <F as ExtensionField<BinaryField1b>>::iter_bases(&self.coeffs[out_idx])
            .nth(in_idx)
            .unwrap()
    }

    pub fn repack_for_fri(self) -> PackedMLE<BinaryField128b> {
        PackedMLE::<BinaryField128b>::new(
            self.coeffs
                .par_chunks(<BinaryField128b as ExtensionField<F>>::DEGREE)
                .map(|base_elems| BinaryField128b::from_bases(base_elems).unwrap())
                .collect(),
            false,
        )
    }
    pub fn get_bound_elem(&self, idx: usize, eq: &LagrangeBases) -> BinaryField128b {
        assert!(idx < 1 << (self.variables - eq.vars), "idx out of bounds");
        let mut res = BinaryField128b::ZERO;
        if eq.vars > 0 {
            let sub_idx = idx << eq.vars;
            for sub_cube in 0..eq.vals.len() {
                res += self.packed_idx(sub_idx | sub_cube) * eq.idx(sub_cube);
            }
        } else {
            res = self.packed_idx(idx).into();
        }
        res
    }

    pub fn fold_as_unpacked_hi(&self, eq: &LagrangeBases) -> PackedMLE<BinaryField128b> {
        PackedMLE::<BinaryField128b>::new(
            (0..1 << (self.variables - eq.vars))
                .into_par_iter()
                .map(|i| {
                    assert!(i < 1 << (self.variables - eq.vars), "idx out of bounds");
                    let mut res = BinaryField128b::ZERO;
                    let shift = self.variables - eq.vars;
                    for sub_cube in 0..eq.vals.len() {
                        res += self.packed_idx(i | (sub_cube << shift)) * eq.idx(sub_cube);
                    }
                    res
                })
                .collect(),
            false,
        )
    }

    pub fn fold_as_unpacked_lo(&self, eq: &LagrangeBases) -> PackedMLE<BinaryField128b> {
        PackedMLE::<BinaryField128b>::new(
            (0..1 << (self.variables - eq.vars))
                .into_par_iter()
                .map(|i| self.get_bound_elem(i, eq))
                .collect(),
            false,
        )
    }

    pub fn batch_rows(&self, eq: &LagrangeBases) -> PackedMLE<BinaryField128b> {
        PackedMLE::<BinaryField128b>::new(
            (0..1 << (self.variables - TAU))
                .into_par_iter()
                .map(|k| {
                    let res = (0..1 << TAU)
                        .into_iter()
                        .map(|i| self.packed_idx(i | (k << TAU)) * eq.idx(i))
                        .sum();
                    res
                })
                .collect(),
            false,
        )
    }

    pub fn idx(&self, idx: usize) -> F {
        self.coeffs[idx]
    }

    pub fn fold_lo(&self, r: &BinaryField128b) -> PackedMLE<BinaryField128b> {
        let half_len = self.coeffs.len() >> 1;
        let fold = (0..half_len)
            .into_par_iter()
            .map(|i| *r * (self.coeffs[i << 1] + self.coeffs[(i << 1) | 1]) + self.coeffs[i << 1])
            .collect();

        PackedMLE::<BinaryField128b>::new(fold, false)
    }

    pub fn fold_hi(&self, r: &BinaryField128b) -> PackedMLE<BinaryField128b> {
        let half_len = self.coeffs.len() >> 1;
        let fold = (0..half_len)
            .into_par_iter()
            .map(|i| *r * (self.coeffs[i] + self.coeffs[i | (1 << i)] + self.coeffs[i]))
            .collect();

        PackedMLE::<BinaryField128b>::new(fold, false)
    }

    pub fn fold_as_packed_hi(&self, eq: &LagrangeBases) -> PackedMLE<BinaryField128b> {
        PackedMLE::<BinaryField128b>::new(
            (0..self.len() >> eq.vars)
                .into_iter()
                .map(|i| {
                    assert!(i < self.len() >> eq.vars, "idx out of bounds");
                    let mut res = BinaryField128b::ZERO;
                    let shift = (self.len().trailing_zeros() as usize) - eq.vars;
                    for sub_cube in 0..eq.vals.len() {
                        res += eq.idx(sub_cube) * self.idx(i | (sub_cube << shift));
                    }
                    res
                })
                .collect(),
            false,
        )
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }
}

pub struct LagrangeBases {
    pub vals: Vec<BinaryField128b>,
    pub vars: usize,
}

impl LagrangeBases {
    pub fn new() -> LagrangeBases {
        LagrangeBases {
            vals: vec![BinaryField128b::ONE],
            vars: 0,
        }
    }

    pub fn packed_idx(&self, idx: usize) -> BinaryField1b {
        let out_idx = idx >> 7;
        let in_idx = (out_idx << 7) ^ idx;
        <BinaryField128b as ExtensionField<BinaryField1b>>::iter_bases(&self.vals[out_idx])
            .nth(in_idx)
            .unwrap()
    }

    pub fn from_mle(mle: PackedMLE<BinaryField128b>) -> LagrangeBases {
        let vars = mle.coeffs.len().trailing_zeros() as usize;
        LagrangeBases {
            vals: mle.coeffs,
            vars,
        }
    }

    pub fn gen_from_point(point: &[BinaryField128b]) -> LagrangeBases {
        LagrangeBases {
            vals: compute_eq(point),
            vars: point.len(),
        }
    }

    pub fn tensor(&mut self, point: &BinaryField128b) {
        let mut vals = vec![BinaryField128b::ZERO; 1 << (self.vars)];

        (self.vals.par_iter_mut(), vals.par_iter_mut())
            .into_par_iter()
            .for_each(|(l, r)| {
                *r = *point * *l;
                *l += *r;
            });

        self.vals.extend(vals);
        self.vars += 1;
    }

    pub fn fold_lo(&mut self, r: &BinaryField128b) {
        let half_len = self.vals.len() >> 1;
        let fold: Vec<BinaryField128b> = (0..half_len)
            .into_par_iter()
            .map(|i| *r * (self.vals[i << 1] + self.vals[(i << 1) | 1]) + self.vals[i << 1])
            .collect();

        self.vals = fold;
        self.vars -= 1;
    }

    #[instrument(skip_all, name = "row batch eq", level = "debug")]
    pub fn row_batch(&mut self, eq: &LagrangeBases) -> LagrangeBases {
        let vals = (0..self.vals.len())
            .into_par_iter()
            .map(|i| {
                <BinaryField128b as ExtensionField<BinaryField1b>>::iter_bases(&self.idx(i))
                    .zip(eq.vals.iter())
                    .map(|(b, e)| b * *e)
                    .sum()
            })
            .collect();
        LagrangeBases {
            vals,
            vars: self.vars,
        }
    }

    //Compute eq table with one variable dropped for Gruen's optimisation.
    pub fn fold_in(&mut self) {
        if self.vals.len() > 1 {
            let fold = (0..self.vals.len() / 2)
                .into_par_iter()
                .map(|i| self.idx(i << 1) + self.idx((i << 1) | 1))
                .collect();
            self.vals = fold;
            self.vars -= 1;
        }
    }

    pub fn idx(&self, idx: usize) -> BinaryField128b {
        self.vals[idx]
    }
}

fn compute_eq(r: &[BinaryField128b]) -> Vec<BinaryField128b> {
    let mut bases: Vec<BinaryField128b> = vec![BinaryField128b::ZERO; 1 << r.len()];
    let mut size = 1;
    bases[0] = BinaryField128b::ONE;

    for r in r.iter() {
        let (bases_left, bases_right) = bases.split_at_mut(size);
        let (bases_right, _) = bases_right.split_at_mut(size);

        bases_left
            .par_iter_mut()
            .zip_eq(bases_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * *r;
                *x -= *y;
            });

        size *= 2;
    }

    bases
}

pub fn compute_dot_product(
    scalars: &[BinaryField128b],
    vals: &[BinaryField128b],
) -> BinaryField128b {
    vals.iter()
        .zip(scalars.iter())
        .map(|(val, scalar)| *scalar * *val)
        .sum()
}

//Assumes input is in column view.
pub fn compute_row_batch(scalars: &[BinaryField128b], vals: &[BinaryField128b]) -> BinaryField128b {
    let row_view = switch_view(vals);

    row_view
        .iter()
        .zip(scalars.iter())
        .map(|(val, scalar)| *scalar * *val)
        .sum()
}

//Switches view of an algebra element from column to row and vice versa.
pub fn switch_view(vals: &[BinaryField128b]) -> Vec<BinaryField128b> {
    let mut row_view = Vec::new();

    for i in 0..128 {
        let mut row_bases = Vec::new();
        for j in 0..128 {
            row_bases.push(
                <BinaryField128b as ExtensionField<BinaryField1b>>::iter_bases(&vals[j])
                    .nth(i)
                    .unwrap(),
            );
        }
        row_view.push(BinaryField128b::from_bases(&row_bases).unwrap())
    }

    row_view
}
