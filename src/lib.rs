use fst::Automaton;

pub struct RangeSearch<'a, const N: usize> {
    query: &'a [u8],
    max_distance: f32,
    distance_fn: Box<dyn Fn(u8, u8) -> f32>
}

impl<'a, const N: usize> RangeSearch<'a, N> 
{
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
    /// fn l2_step(from: u8, to: u8) -> f32 {
    ///     (from as f32 - to as f32) * (from as f32 - to as f32)
    /// }
    /// 
    /// let query = [0x01; 144];
    /// let aut = RangeSearch::<144>::new(&query, 100_000.0, l2_step);
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
    pub fn new<F>(query: &'a [u8], max_distance: f32, distance_fn: F) -> Self 
    where 
        F: Fn(u8, u8) -> f32 + 'static
    {
        Self {
            query,
            max_distance,
            distance_fn: Box::new(distance_fn),
        }
    }

    pub fn new_hamming(query: &'a [u8], max_distance: usize) -> Self {
        Self {
            query,
            max_distance: max_distance as f32,
            distance_fn: Box::new(hamming_step),
        }
    }

    pub fn new_l2(query: &'a [u8], max_distance_squared: f32) -> Self {
        Self {
            query,
            max_distance: max_distance_squared,
            distance_fn: Box::new(l2_step),
        }
    }
}


pub fn hamming_step(from: u8, to: u8) -> f32 {
    (from ^ to).count_ones() as f32
}

pub fn l2_step(from: u8, to: u8) -> f32 {
    (from as f32 - to as f32) * (from as f32 - to as f32)
}

impl<const N: usize> Automaton for RangeSearch<'_, N> {
    type State = (f32, usize, bool);

    fn start(&self) -> Self::State {
        (self.max_distance, 0, self.query.len() != { N })
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        let step_by = (self.distance_fn)(self.query[state.1], byte);
        (state.0 - step_by, state.1 + 1, false)
    }

    fn is_match(&self, state: &Self::State) -> bool {
        if state.2 {
            return false;
        }
        state.1 == { N } && state.0.is_sign_positive()
    }

    fn can_match(&self, state: &Self::State) -> bool {
        // we already know the query vector is of an incompatible
        // size.
        if state.2 {
            return false;
        }
        // our scanned vector is larger than what we're expecting.
        if state.1 > { N } {
            return false;
        }
        // can only match if there is any budget left at all.
        state.0.is_sign_positive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use fst::{IntoStreamer, MapBuilder, Streamer};
    use fst::set::SetBuilder;
    use rand::{prelude::*, rng};
    use arbitrary::{Arbitrary, Unstructured};

    fn generate_data<const SIZE: usize>(count: usize) -> Vec<[u8; SIZE]> {
        let mut rng = rng();
        let mut buffer=  [0u8; 1024];
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

        let aut = RangeSearch::<4>::new_hamming(
            &query, 
            5,
        );
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
        let aut = RangeSearch::<144>::new(
            &query, 
            max_distance,
            l2_step
        );
        let mut stream = map.search(aut).into_stream();
        let mut hit_indices = HashSet::new();

        while let Some((key, idx)) = stream.next() {
            let distance=  naive_distance_l2(&query, key);
            assert!(distance <= max_distance);
            hit_indices.insert(idx);
        }
        eprintln!("hits: {}", hit_indices.len());
        let non_hits: Vec<_> = 
            data_with_ids
            .iter()
            .filter_map(|(idx, vector)| {
                match hit_indices.contains(&(*idx as u64)) {
                    false => Some(*vector),
                    true => None
                }
            })
            .collect();
    
        eprintln!("non hits: {}", non_hits.len());
        for non_hit in non_hits {
            let distance=  naive_distance_l2(&query, &non_hit);
            assert!(distance > max_distance);
        }
    }

    #[test]
    fn get_single_hit() {
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
        
        let num_searches: usize = 1_000;
        // generate search queries.
        let queries: Vec<[u8; SIZE]> = generate_data::<SIZE>(num_searches);
        eprintln!("generated {num_searches} query vectors of dimension {SIZE}...");

        let max_distance: f32 = 1_000_000.0;
        let mut hits = 0;
        let mut search_times = Vec::with_capacity(num_searches);
        let mut seen = 0;
        for query in queries {
            let aut = RangeSearch::<144>::new_l2(
                &query, 
                max_distance,
            );
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
        eprintln!("total hits: {hits} at max_distance={max_distance:.4} out of {num_searches} queries against {count} vectors.");

    }
}
