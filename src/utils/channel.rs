use std::io::BufWriter;

use binius_field::{
    error,
    serialize_canonical,
    BinaryField,
    BinaryField128b,
    TowerField,
};
use binius_utils::serialization::{ self, DeserializeBytes, Error, SerializeBytes };
use sha3::{ Digest, Keccak256 };

use super::merkle::Commitment;

//Instantiation of Fiat-Shamir transform
pub struct Channel {
    state: Keccak256,
    round_idx: usize,
}

impl Channel {
    pub fn new() -> Channel {
        let mut state = Keccak256::new();
        Channel {
            state,
            round_idx: 0,
        }
    }
    pub fn observe_field_elem<F: BinaryField + TowerField>(
        &mut self,
        elem: F
    ) -> Result<(), Error> {
        let mut buffer = &mut Vec::new();
        match serialize_canonical(elem, &mut buffer) {
            Ok(()) => {
                self.state.update(buffer);
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    pub fn observe_field_elems<F: BinaryField + TowerField>(
        &mut self,
        elems: &[F]
    ) -> Result<(), Error> {
        //Conveniently a collection of the same result types can be converted to a result type indicating an error in any iteration
        match
            elems
                .iter()
                .map(|elem| self.observe_field_elem(*elem))
                .collect()
        {
            Ok(()) => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub fn observe_commitment(&mut self, commitment: Commitment) {
        self.state.update(commitment.root().0);
        self.state.update(commitment.depth().to_le_bytes());
    }

    pub fn get_random_point(
        &mut self
    ) -> Result<BinaryField128b, Error> {
        self.state.update(self.round_idx.to_le_bytes());
        let mut out = self.state.clone().finalize();
        match BinaryField128b::deserialize(out.as_slice()) {
            Ok(random_point) => {
                self.round_idx += 1;
                Ok(random_point)
            }
            Err(error) => Err(error),
        }
    }
    pub fn get_random_points(&mut self, n:usize)->Result<Vec<BinaryField128b>, Error> {
        match (0..n).into_iter().map(|_| self.get_random_point()).collect(){
            Ok(random_points)=>Ok(random_points),
            Err(error) => Err(error)
        }
    }

    //Number of queries hard coded for 96 bits of security and R=4 as per [DP24]
    pub fn gen_queries<F: BinaryField + TowerField>(&mut self, log_max_len:usize)->Result<Vec<usize>, Error>{

        match self.get_random_points(144) {
            Ok(random_elems)=>{
                let bit_mask = (1<<log_max_len) - 1;
                let queries = random_elems.iter().map(|elem| (elem.val()&bit_mask) as usize).collect();
                Ok(queries)
            },
            Err(error)=>Err(error)
        }
    }
}
