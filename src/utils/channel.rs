use anyhow::{Context, Result, bail};
use binius_field::{BinaryField, BinaryField128b, TowerField, serialize_canonical};
use binius_utils::serialization::DeserializeBytes;
use sha3::{Digest, Keccak256};

use crate::prover::FriCommitment;

use super::merkle::VectorCommitment;

/// Fiat–Shamir transcript helper for deriving deterministic challenges.
pub struct Channel {
    state: Keccak256,
    round_idx: usize,
}

impl Channel {
    pub fn new() -> Self {
        Self {
            state: Keccak256::new(),
            round_idx: 0,
        }
    }

    fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.state.update(bytes);
    }

    fn sample_digest(&self, counter: usize) -> [u8; 32] {
        let mut sponge = self.state.clone();
        sponge.update(counter.to_le_bytes());
        sponge.finalize().into()
    }

    pub fn observe_field_elem<F: BinaryField + TowerField>(&mut self, elem: F) -> Result<()> {
        let mut buffer = Vec::new();
        serialize_canonical(elem, &mut buffer).context("serialize field element")?;
        self.absorb_bytes(&buffer);
        Ok(())
    }

    pub fn observe_field_elems<F: BinaryField + TowerField>(&mut self, elems: &[F]) -> Result<()> {
        elems
            .iter()
            .try_for_each(|elem| self.observe_field_elem(*elem))
            .context("serialize field element collection")
    }

    pub fn observe_vector_commitment(&mut self, commitment: &VectorCommitment) {
        self.absorb_bytes(&commitment.root().0);
        self.absorb_bytes(&commitment.depth().to_le_bytes());
    }

    pub fn observe_fri_commitment(&mut self, commitment: &FriCommitment) {
        self.observe_vector_commitment(&commitment.vector_commitment);
        self.absorb_bytes(&commitment.packing_factor.to_le_bytes())
    }

    pub fn get_random_point(&mut self) -> Result<BinaryField128b> {
        let digest = self.sample_digest(self.round_idx);
        self.round_idx = self
            .round_idx
            .checked_add(1)
            .context("channel counter overflow")?;

        BinaryField128b::deserialize(digest.as_slice()).context("draw random point from channel")
    }

    pub fn get_random_points(&mut self, n: usize) -> Result<Vec<BinaryField128b>> {
        (0..n)
            .map(|_| self.get_random_point())
            .collect::<Result<Vec<_>>>()
            .context("draw random points from channel")
    }

    /// Number of queries hard coded for 96 bits of security and R=4 as per [DP24].
    pub fn gen_queries(&mut self, log_max_len: usize) -> Result<Vec<usize>> {
        let shift = u32::try_from(log_max_len).context("log_max_len does not fit in u32")?;
        let domain_size = 1usize
            .checked_shl(shift)
            .context("log_max_len too large for usize")?;

        if domain_size == 0 {
            bail!("domain size must be positive");
        }

        if domain_size < 144 {
            // Domain is small; query every element once.
            return Ok((0..domain_size).collect());
        }

        let bit_mask = u128::try_from(domain_size - 1).context("domain size exceeds 2^128")?;
        let random_elems = self.get_random_points(144)?;
        Ok(random_elems
            .iter()
            .map(|elem| (elem.val() & bit_mask) as usize)
            .collect())
    }
}
