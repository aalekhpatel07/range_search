use fst::Automaton;

pub struct RangeSearch<'a, const N: usize, D> {
    query: &'a [u8],
    max_distance: D,
    distance_fn: fn(u8, u8) -> D,
}

impl<'a, const N: usize, D> RangeSearch<'a, N, D> {
    /// Create a RangeSearch automaton that accepts byte vectors of a given dimension
    /// only if they are at most a given distance apart where the distances are calculated
    /// according to a given distance function.
    ///
    /// Note: The stream of vectors returned aren't returned in the order of distances but the order
    /// in which they were inserted in the underlying `fst::Fst`.
    /// # Examples
    ///
    /// - An automaton that accepts only (144,)-u8 vectors if they are within a squared L2-distance
    ///   of 100_000.0 apart of a given query vector.
    /// ```rust
    /// use range_search::RangeSearch;
    /// use fst::{Automaton, set::{Set, SetBuilder}, IntoStreamer, Streamer};
    ///
    /// let query = [0x01; 144];
    /// let aut = RangeSearch::<144, _>::new_l2(&query, 100_000.0).unwrap();
    ///
    /// // Suppose you have some fst::Set of vectors.
    /// let set = SetBuilder::memory().into_set();
    ///
    /// // Iterate over the set of vectors close enough to the query vector.
    /// let mut stream = set.search(aut).into_stream();
    /// while let Some(hit) = stream.next() {
    ///     eprintln!("found a vector within the given range of the query vector {hit:#?}");
    /// }
    /// ```
    pub fn new(
        query: &'a [u8],
        max_distance: D,
        distance_fn: fn(u8, u8) -> D,
    ) -> fst::Result<Self> {
        if query.len() as u64 != { N as u64 } {
            return Err(fst::Error::Fst(fst::raw::Error::WrongType {
                expected: { N as u64 },
                got: query.len() as u64,
            }));
        }
        Ok(Self {
            query,
            max_distance,
            distance_fn,
        })
    }
}

macro_rules! impl_hamming_for {
    ($t:ty) => {
        impl<'a, const N: usize> RangeSearch<'a, N, $t> {
            pub fn new_hamming(query: &'a [u8], max_distance: $t) -> fst::Result<Self> {
                Self::new(query, max_distance, |from, to| {
                    (from ^ to).count_ones() as $t
                })
            }
        }
    };
}

impl_hamming_for! { u8 }
impl_hamming_for! { u16 }
impl_hamming_for! { u32 }
impl_hamming_for! { u64 }

impl_hamming_for! { i8 }
impl_hamming_for! { i16 }
impl_hamming_for! { i32 }
impl_hamming_for! { i64 }

macro_rules! impl_l2_for {
    ($t:ty) => {
        impl<'a, const N: usize> RangeSearch<'a, N, $t> {
            pub fn new_l2(query: &'a [u8], max_distance_squared: $t) -> fst::Result<Self> {
                Self::new(query, max_distance_squared, |from, to| {
                    (from as $t - to as $t) * (from as $t - to as $t)
                })
            }
        }
    };
}

impl_l2_for! { f32 }
impl_l2_for! { f64 }

impl_l2_for! { u16 }
impl_l2_for! { u32 }
impl_l2_for! { u64 }
impl_l2_for! { u128 }

impl_l2_for! { i16 }
impl_l2_for! { i32 }
impl_l2_for! { i64 }
impl_l2_for! { i128 }

impl<const N: usize, D> Automaton for RangeSearch<'_, N, D>
where
    D: std::ops::Add<D, Output = D> + Copy + PartialEq + PartialOrd + Default,
{
    type State = (D, usize);

    fn start(&self) -> Self::State {
        (Default::default(), 0)
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        let step_by = (self.distance_fn)(self.query[state.1], byte);
        (state.0 + step_by, state.1 + 1)
    }

    fn is_match(&self, state: &Self::State) -> bool {
        state.1 == { N } && state.0 <= self.max_distance
    }

    fn can_match(&self, state: &Self::State) -> bool {
        // our scanned vector is larger than what we're expecting.
        if state.1 > { N } {
            return false;
        }
        // can only match if we didn't already accumulate enough distance.
        state.0 <= self.max_distance
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use arbitrary::{Arbitrary, Unstructured};
    use fst::set::SetBuilder;
    use fst::{IntoStreamer, MapBuilder, Streamer};
    use rand::{prelude::*, rng};

    fn generate_data<const SIZE: usize>(count: usize) -> Vec<[u8; SIZE]> {
        let mut rng = rng();
        let mut buffer = [0u8; 1024];
        let mut vectors = vec![];
        for _ in 0..count {
            rng.fill_bytes(&mut buffer);
            let data: [u8; SIZE] = Arbitrary::arbitrary(&mut Unstructured::new(&buffer)).unwrap();
            vectors.push(data);
        }
        vectors
    }

    #[test]
    fn automaton_works() {
        let mut builder = SetBuilder::memory();
        let mut data = vec![[0u8, 255, 255, 255], [255, 255, 255, 255], [1, 1, 1, 1]];
        data.sort();

        for vector in data {
            builder.insert(vector).unwrap();
        }

        let set = builder.into_set();

        let query = [0, 0, 0, 0];

        let aut = RangeSearch::<4, i32>::new_hamming(&query, 5).unwrap();
        let stream = set.search(aut).into_stream();
        let observed_matches = stream.into_bytes();
        assert_eq!(observed_matches, vec![vec![1, 1, 1, 1]]);
    }

    pub fn naive_distance_l2(v1: &[u8], v2: &[u8]) -> f32 {
        let mut dist: f32 = 0.;
        for (&b1, &b2) in v1.iter().zip(v2) {
            dist += ((b1 as f32) - (b2 as f32)) * ((b1 as f32) - (b2 as f32))
        }
        dist
    }

    #[test]
    fn automaton_correctness_l2() {
        // generate 1M vectors.
        let mut data: Vec<[u8; 144]> = generate_data::<144>(1_000_000);
        data.sort();
        eprintln!("finished sorting...");
        let data_with_ids: Vec<_> = data.into_iter().enumerate().collect();

        let mut builder = MapBuilder::memory();
        for (idx, vector) in data_with_ids.iter() {
            builder.insert(*vector, *idx as u64).unwrap();
        }
        let map = builder.into_map();
        eprintln!("finished generating map. begin searches.");
        let query = generate_data::<144>(1)[0];

        let max_distance: f32 = 1_500_000.0;
        let aut = RangeSearch::<144, f32>::new_l2(&query, max_distance).unwrap();
        let mut stream = map.search(aut).into_stream();
        let mut hit_indices = HashSet::new();

        while let Some((key, idx)) = stream.next() {
            let distance = naive_distance_l2(&query, key);
            assert!(distance <= max_distance);
            hit_indices.insert(idx);
        }
        eprintln!("hits: {}", hit_indices.len());
        let non_hits: Vec<_> = data_with_ids
            .iter()
            .filter_map(|(idx, vector)| match hit_indices.contains(&(*idx as u64)) {
                false => Some(*vector),
                true => None,
            })
            .collect();

        eprintln!("non hits: {}", non_hits.len());
        for non_hit in non_hits {
            let distance = naive_distance_l2(&query, &non_hit);
            assert!(distance > max_distance);
        }
    }

    #[test]
    fn get_single_hit_l2() {
        let count: usize = 100_000;
        const SIZE: usize = 144;
        // generate vectors.
        let mut data: Vec<[u8; SIZE]> = generate_data::<SIZE>(count);
        eprintln!("generated {count} vectors of dimension {SIZE}...");
        data.sort();
        eprintln!("sorted {count} vectors of dimension {SIZE}...");

        let mut builder = SetBuilder::memory();
        for vector in data.iter() {
            builder.insert(*vector).unwrap();
        }
        let set = builder.into_set();
        eprintln!("finished generating set. begin searches.");

        let num_searches: usize = 100;
        // generate search queries.
        let queries: Vec<[u8; SIZE]> = generate_data::<SIZE>(num_searches);
        eprintln!("generated {num_searches} query vectors of dimension {SIZE}...");

        let max_distance: u64 = 1_000_000;
        let mut hits = 0;
        let mut search_times = Vec::with_capacity(num_searches);
        let mut seen = 0;
        for query in queries {
            let aut = RangeSearch::<SIZE, u64>::new_l2(&query, max_distance).unwrap();
            let mut stream = set.search_with_state(aut).into_stream();
            let search_time = std::time::Instant::now();
            if let Some((_hit, state)) = stream.next() {
                eprintln!("hit state: {state:?}");
                // found some hit.
                hits += 1;
            }
            seen += 1;
            let elapsed = search_time.elapsed();
            eprintln!("seen so far: {}", seen);
            search_times.push(elapsed.as_nanos());
        }
        eprintln!(
            "search times (ns): max={} min={} mean={} total={} hits={}",
            search_times.iter().max().unwrap(),
            search_times.iter().min().unwrap(),
            search_times.iter().copied().sum::<u128>() / search_times.len() as u128,
            num_searches,
            hits,
        );
        eprintln!(
            "total hits: {hits} at max_distance={max_distance:.4} out of {num_searches} queries against {count} vectors."
        );
    }

    #[test]
    fn get_single_hit_hamming() {
        let count: usize = 100_000;
        const SIZE: usize = 32;
        // generate vectors.
        let mut data: Vec<[u8; SIZE]> = generate_data::<SIZE>(count);
        eprintln!("generated {count} vectors of dimension {SIZE}...");
        data.sort();
        eprintln!("sorted {count} vectors of dimension {SIZE}...");

        let mut builder = SetBuilder::memory();
        for vector in data.iter() {
            builder.insert(*vector).unwrap();
        }
        let set = builder.into_set();
        eprintln!("finished generating set. begin searches.");

        let num_searches: usize = 100;
        // generate search queries.
        let queries: Vec<[u8; SIZE]> = generate_data::<SIZE>(num_searches);
        eprintln!("generated {num_searches} query vectors of dimension {SIZE}...");

        let max_distance: i32 = 100;
        let mut hits = 0;
        let mut search_times = Vec::with_capacity(num_searches);
        let mut seen = 0;
        for query in queries {
            let aut = RangeSearch::<SIZE, i32>::new_hamming(&query, max_distance).unwrap();
            let mut stream = set.search_with_state(aut).into_stream();
            let search_time = std::time::Instant::now();
            if let Some((_hit, state)) = stream.next() {
                eprintln!("hit state: {state:?}");
                // found some hit.
                hits += 1;
            }
            seen += 1;
            let elapsed = search_time.elapsed();
            eprintln!("seen so far: {}", seen);
            search_times.push(elapsed.as_nanos());
        }
        eprintln!(
            "search times (ns): max={} min={} mean={} total={} hits={}",
            search_times.iter().max().unwrap(),
            search_times.iter().min().unwrap(),
            search_times.iter().copied().sum::<u128>() / search_times.len() as u128,
            num_searches,
            hits,
        );
        eprintln!(
            "total hits: {hits} at max_distance={max_distance:.4} out of {num_searches} queries against {count} vectors."
        );
    }
}
