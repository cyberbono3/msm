use ark_ec::CurveGroup;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;


pub fn binary_msm<F, G>(
    scalars: &[u8],
    points: &[Vec<G::Affine>],
) -> G 
where G: CurveGroup<ScalarField = F>
{
    assert_eq!(points.len(), scalars.len());
    points
        .par_iter()
        .zip(scalars.par_iter())
        .filter(|(_, &scalar)| scalar as usize != 0)
        .map(|(point, scalar)| point[(*scalar) as usize - 1])
        .fold(G::zero, |acc, new| acc + new)
        .reduce(G::zero, |acc, new| acc + new)
}

#[cfg(test)]
mod test {

    use super::*;
    use ark_bls12_381::{G1Affine, G1Projective};
    use ark_std::rand::Rng;
    use ark_std::{test_rng, UniformRand, Zero};
    use itertools::Itertools;

    fn binary_rep(iter: impl Iterator<Item = bool>) -> u8 {
        iter.take(8)
            .fold(0, |s, b| (s << 1) + if b { 1 } else { 0 })
    }

    #[test]
    fn test_binary_rep_all_false() {
        // Input: 0000_0000
        let input = vec![false, false, false, false, false, false, false, false];
        let result = binary_rep(input.into_iter());
        assert_eq!(result, 0b0000_0000);
    }

    #[test]
    fn test_binary_rep_all_true() {
        // Input: 1111_1111
        let input = vec![true, true, true, true, true, true, true, true];
        let result = binary_rep(input.into_iter());
        assert_eq!(result, 0b1111_1111);
    }

    #[test]
    fn test_binary_rep_mixed() {
        // Input: 1010_0101
        let input = vec![true, false, true, false, false, true, false, true];
        let result = binary_rep(input.into_iter());
        assert_eq!(result, 0b1010_0101);
    }

    fn make_scalars(scalars: impl Iterator<Item = bool>, chunk_size: usize) -> Vec<u8> {
        scalars
            .chunks(chunk_size)
            .into_iter()
            .map(|chunk| binary_rep(chunk))
            .collect()
    }

    pub fn process_chunk<F, G: CurveGroup<ScalarField = F>>(
        chunk: &[G::Affine],
        chunk_size: usize,
    ) -> Vec<G::Affine> {
        let proj = (1..(1 << chunk_size))
            .map(|i| {
                (0..chunk_size)
                    .zip(chunk.iter().rev())
                    .filter(|(idx, _)| (1 << idx) & i != 0)
                    .map(|(_, b): (_, &G::Affine)| *b)
                    .fold(G::zero(), |acc, new| acc + new)
            })
            .collect_vec();
        G::normalize_batch(&proj)
    }

    fn make_points<F, G: CurveGroup<ScalarField = F>>(
        points: &[G::Affine],
        chunk_size: usize,
    ) -> Vec<Vec<G::Affine>> {
        points
            .par_chunks(chunk_size)
            .map(|chunk: &[G::Affine]| process_chunk::<F, G>(chunk, chunk_size))
            .collect()
    }


    #[test]
    fn test_msm_basic() {
        let gen = &mut test_rng();
        let bool_vec = (0..50).map(|_| gen.gen_bool(0.5)).collect_vec();
        let scalars: Vec<u8> = make_scalars(bool_vec.clone().into_iter(), 8);
        let points_vec = (0..50).map(|_| G1Affine::rand(gen)).collect_vec();

        let points = make_points::<_, G1Projective>(&points_vec, 8);

        let result = binary_msm::<_, G1Projective>(&scalars, &points);

        let expected = bool_vec
            .iter()
            .zip_eq(points_vec.iter())
            .filter(|(&c, _)| c)
            .map(|(_, b)| b)
            .fold(G1Projective::zero(), |acc, new| acc + new);
        assert_eq!(result.into_affine(), expected.into_affine());
    }
}
