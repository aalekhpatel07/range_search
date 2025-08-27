use fst::raw::{Builder, Fst};
use fst::{Automaton, IntoStreamer, Streamer};
mod automata;

pub use automata::*;


/// A builder for creating a vector set.
///
/// This is not your average everyday builder. It has two important qualities
/// that make it a bit unique from what you might expect:
///
/// 1. All keys must be added in lexicographic order. Adding a key out of order
///    will result in an error.
/// 2. The representation of a vector set is streamed to *any* `io::Write` as it is
///    built. For an in memory representation, this can be a `Vec<u8>`.
///
/// Point (2) is especially important because it means that a vector set can be
/// constructed *without storing the entire vector set in memory*. Namely, since it
/// works with any `io::Write`, it can be streamed directly to a file.
///
/// With that said, the builder does use memory, but **memory usage is bounded
/// to a constant size**. The amount of memory used trades off with the
/// compression ratio. Currently, the implementation hard codes this trade off
/// which can result in about 5-20MB of heap usage during construction. (N.B.
/// Guaranteeing a maximal compression ratio requires memory proportional to
/// the size of the set, which defeats the benefit of streaming it to disk.
/// In practice, a small bounded amount of memory achieves close-to-minimal
/// compression ratios.)
///
/// The algorithmic complexity of set construction is `O(n)` where `n` is the
/// number of vectors added to the vector set.
///
/// # Example: build in memory
///
/// This shows how to use the builder to construct a vector set in memory. Note that
/// `VectorSet::from_iter` provides a convenience function that achieves this same
/// goal without needing to explicitly use `SetBuilder`.
/// 
/// ```rust
/// use fst::{IntoStreamer, Streamer};
/// use vector_set::{VectorSetBuilder, VectorSet};
///
/// let mut build = VectorSetBuilder::<4, Vec<u8>>::memory();
/// build.insert(&[1, 1, 2, 3]).unwrap();
/// build.insert(&[1, 2, 2, 3]).unwrap();
/// build.insert(&[2, 255, 2, 3]).unwrap();
///
/// // You could also call `finish()` here, but since we're building the vector set in
/// // memory, there would be no way to get the `Vec<u8>` back.
/// let bytes = build.into_inner().unwrap();
///
/// // At this point, the set has been constructed, but here's how to read it.
/// let vset = VectorSet::<4, _>::new(bytes).unwrap();
/// let mut stream = vset.into_stream();
/// let mut keys = vec![];
/// while let Some(key) = stream.next() {
///     keys.push(key.to_vec());
/// }
/// assert_eq!(keys, vec![
///     vec![1, 1, 2, 3], 
///     vec![1, 2, 2, 3], 
///     vec![2, 255, 2, 3],
/// ]);
/// ```
///
/// # Example: stream to file
///
/// This shows how to stream construction of a set to a file.
///
/// ```rust,no_run
/// use std::fs::File;
/// use std::io;
///
/// use fst::{IntoStreamer, Streamer, Set, SetBuilder};
/// use vector_set::{VectorSetBuilder, VectorSet};
///
/// let mut wtr = io::BufWriter::new(File::create("set.fst").unwrap());
/// let mut build = VectorSetBuilder::<4, _>::new(wtr).unwrap();
/// build.insert(&[1, 1, 2, 3]).unwrap();
/// build.insert(&[1, 2, 2, 3]).unwrap();
/// build.insert(&[2, 255, 2, 3]).unwrap();
///
/// // If you want the writer back, then call `into_inner`. Otherwise, this
/// // will finish construction and call `flush`.
/// build.finish().unwrap();
///
/// // At this point, the set has been constructed, but here's how to read it.
/// // NOTE: Normally, one would memory map a file instead of reading its
/// // entire contents on to the heap.
/// let vset = VectorSet::<4, _>::new(std::fs::read("set.fst").unwrap()).unwrap();
/// let mut stream = vset.into_stream();
/// let mut keys = vec![];
/// while let Some(key) = stream.next() {
///     keys.push(key.to_vec());
/// }
/// assert_eq!(keys, vec![
///     vec![1, 1, 2, 3], 
///     vec![1, 2, 2, 3], 
///     vec![2, 255, 2, 3],
/// ]);
/// ```
pub struct VectorSetBuilder<const N: usize, W>(fst::raw::Builder<W>);

impl<const N: usize> VectorSetBuilder<N, Vec<u8>> {

    /// Create a builder that builds a vector set in memory.
    #[inline]
    pub fn memory() -> Self {
        Self(Builder::memory())
    }

    /// Finishes the construction of the set and returns it.
    #[inline]
    pub fn into_vector_set(self) -> VectorSet<N, Vec<u8>> {
        VectorSet(self.0.into_fst())
    }

}

/// A specialized stream for mapping vector set streams (`&[u8]`) to streams used
/// by raw fsts (`(&[u8], Output)`).
///
/// If this were iterators, we could use `iter::Map`, but doing this on streams
/// requires HKT, so we need to write out the monomorphization ourselves.
struct StreamZeroOutput<S>(S);

impl<'a, S: Streamer<'a>> Streamer<'a> for StreamZeroOutput<S> {
    type Item = (S::Item, fst::raw::Output);

    fn next(&'a mut self) -> Option<(S::Item, fst::raw::Output)> {
        self.0.next().map(|key| (key, fst::raw::Output::zero()))
    }
}


impl<const N: usize, W: std::io::Write> VectorSetBuilder<N, W> {
    /// Create a builder that builds a set of vectors by writing it to `wtr` in a
    /// streaming fashion.
    pub fn new(wtr: W) -> fst::Result<VectorSetBuilder<N, W>> {
        fst::raw::Builder::new_type(wtr, 0).map(VectorSetBuilder)
    }

    /// Insert a new vector into the set.
    ///
    /// If a key is inserted that is less than any previous key added, then
    /// an error is returned. Similarly, if there was a problem writing to
    /// the underlying writer, an error is returned.
    pub fn insert<K: AsRef<[u8]>>(&mut self, key: K) -> fst::Result<()> {
        let key = key.as_ref();
        if key.len() != N {
            return Err(fst::Error::Fst(fst::raw::Error::WrongType { expected: {N as u64}, got: key.len() as u64 }));
        }
        self.0.add(key)
    }

    /// Calls insert on each item in the iterator.
    ///
    /// If an error occurred while adding an element, processing is stopped
    /// and the error is returned.
    pub fn extend_iter<T, I>(&mut self, iter: I) -> fst::Result<()>
    where
        T: AsRef<[u8]>,
        I: IntoIterator<Item = T>,
    {
        for key in iter {
            self.insert(key)?;
        }
        Ok(())
    }

    /// Calls insert on each item in the stream.
    ///
    /// Note that unlike `extend_iter`, this is not generic on the items in
    /// the stream.
    pub fn extend_stream<'f, I, S>(&mut self, stream: I) -> fst::Result<()>
    where
        I: for<'a> IntoStreamer<'a, Into = S, Item = &'a [u8]>,
        S: 'f + for<'a> Streamer<'a, Item = &'a [u8]>,
    {
        self.0.extend_stream(StreamZeroOutput(stream.into_stream()))
    }

    /// Finishes the construction of the set and flushes the underlying
    /// writer. After completion, the data written to `W` may be read using
    /// one of `Set`'s constructor methods.
    pub fn finish(self) -> fst::Result<()> {
        self.0.finish()
    }

    /// Just like `finish`, except it returns the underlying writer after
    /// flushing it.
    pub fn into_inner(self) -> fst::Result<W> {
        self.0.into_inner()
    }

    /// Gets a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        self.0.get_ref()
    }

    /// Returns the number of bytes written to the underlying writer
    pub fn bytes_written(&self) -> u64 {
        self.0.bytes_written()
    }

}

pub struct VectorSet<const N: usize, D>(Fst<D>);

#[allow(clippy::should_implement_trait)]
impl<const N: usize> VectorSet<N, Vec<u8>>
{
    /// Create a vector set from an iterator of vectors
    /// of the assumed size.
    pub fn from_iter<I, K>(iter: I) -> fst::Result<VectorSet<N, Vec<u8>>> 
    where 
        I: Iterator<Item=K>,
        K: AsRef<[u8]>
    {
        let mut builder=  VectorSetBuilder::memory();
        for item in iter {
            let key = item.as_ref();
            builder.insert(key)?;
        }
        Ok(builder.into_vector_set())
    }
}

impl<const N: usize, D> VectorSet<N, D>
where D: AsRef<[u8]>
{

    /// Creates a vector set from its representation as a raw byte sequence.
    ///
    /// This accepts anything that can be cheaply converted to a `&[u8]`. The
    /// caller is responsible for guaranteeing that the given bytes refer to
    /// a valid FST. While memory safety will not be violated by invalid input,
    /// a panic could occur while reading the FST at any point.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vector_set::VectorSet;
    ///
    /// // File written from a build script using SetBuilder.
    /// # const IGNORE: &str = stringify! {
    /// static FST: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vector_set.fst"));
    /// # };
    /// # static FST: &[u8] = &[];
    ///
    /// const DIM: usize = 4;
    /// let set = VectorSet::<DIM, _>::new(FST).unwrap();
    /// ```
    pub fn new(data: D) -> fst::Result<VectorSet<N, D>> {
        fst::raw::Fst::new(data).map(VectorSet)
    }

    /// Get a stream of vectors stored in the set 
    /// that are at most the given distance away from the 
    /// query vector.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use vector_set::VectorSet;
    /// use fst::IntoStreamer;
    /// 
    /// let mut data: Vec<[u8; 144]> = vec![
    ///     [0; 144],
    ///     [1; 144],
    ///     [2; 144],
    /// ];
    /// data.sort();
    ///
    /// let vset = VectorSet::<144, _>::from_iter(data.iter()).unwrap();
    /// let stream = vset.neighbors_within_range_l2(&[3u8; 144], 144.0 * 2.0 * 2.0).into_stream();
    /// let neighbors= stream.into_byte_keys();
    /// assert_eq!(neighbors, vec![
    ///     vec![1; 144],
    ///     vec![2; 144],
    /// ]);
    /// ```
    pub fn neighbors_within_range_l2<'own, 'q>(&'own self, query: &'q [u8], max_distance_squared: f32) -> fst::raw::Stream<'own, VectorSetAutomata<'q, N>>
    where 'own: 'q
    {
        let aut = VectorSetAutomata::<N>::new(query, Budget::L2(max_distance_squared));
        self.0.search(aut).into_stream()
    }

    /// Get a stream of vectors stored in the set 
    /// that are at most the given distance away from the 
    /// query vector.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use vector_set::VectorSet;
    /// use fst::IntoStreamer;
    /// 
    /// let mut data: Vec<[u8; 2]> = vec![
    ///     [255, 255],
    ///     [0, 0]
    /// ];
    /// data.sort();
    ///
    /// let vset = VectorSet::<2, _>::from_iter(data.iter()).unwrap();
    /// let stream = vset.neighbors_within_range_hamming(&[1, 1], 2).into_stream();
    /// let neighbors = stream.into_byte_keys();
    /// assert_eq!(neighbors, vec![
    ///     vec![0, 0],
    /// ]);
    /// ```
    pub fn neighbors_within_range_hamming<'own, 'q>(&'own self, query: &'q [u8], max_distance: i64) -> fst::raw::Stream<'own, VectorSetAutomata<'q, N>>
    where 'own: 'q
    {
        let aut = VectorSetAutomata::<N>::new(query, Budget::Hamming(max_distance));
        self.0.search(aut).into_stream()
    }
}

impl<'s, 'a, const N: usize, D: AsRef<[u8]>> IntoStreamer<'a> for &'s VectorSet<N, D> {
    type Item = &'a [u8];
    type Into = Stream<'s>;

    #[inline]
    fn into_stream(self) -> Stream<'s> {
        Stream(self.0.stream())
    }
}

/// A lexicographically ordered stream of keys from a vector set.
///
/// The `A` type parameter corresponds to an optional automaton to filter
/// the stream. By default, no filtering is done.
///
/// The `'s` lifetime parameter refers to the lifetime of the underlying vector set.
pub struct Stream<'s, A = fst::automaton::AlwaysMatch>(fst::raw::Stream<'s, A>)
where
    A: Automaton;

impl<'a, 's, A: Automaton> Streamer<'a> for Stream<'s, A> {
    type Item = &'a [u8];

    fn next(&'a mut self) -> Option<&'a [u8]> {
        self.0.next().map(|(key, _)| key)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use fst::raw::Builder;
    use fst::IntoStreamer;

    #[test]
    fn automaton_works() {
        let mut builder = Builder::memory();
        let mut data = vec![
            [0u8, 255, 255, 255],
            [255, 255, 255, 255],
            [1, 1, 1, 1],
        ];
        data.sort();

        for vector in data {
            builder.add(vector).unwrap();
        }

        let fst = builder.into_fst();

        let query = [0, 0, 0, 0];

        let aut = VectorSetAutomata::<4>::new(&query, Budget::Hamming(5));
        let stream = fst.search(aut).into_stream();
        let observed_matches = stream.into_byte_keys();
        assert_eq!(observed_matches, vec![vec![1, 1, 1, 1]]);
    }

    #[test]
    fn vectorset_works() {
        let mut data: Vec<[u8; 144]> = vec![
            [0; 144],
            [1; 144],
            [2; 144],
        ];
        data.sort();

        let vset = VectorSet::<144, _>::from_iter(data.iter()).unwrap();
        let stream = vset.neighbors_within_range_l2(&[3u8; 144], 144.0 * 2.0 * 2.0).into_stream();
        let neighbors= stream.into_byte_keys();
        assert_eq!(neighbors, vec![
            data[1],
            data[2],
        ]);
    }
}
