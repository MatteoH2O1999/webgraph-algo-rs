use crate::{prelude::*, utils::MmapSlice};
use anyhow::{ensure, Context, Result};
use common_traits::*;
use rayon::prelude::*;
use std::{
    hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash},
    marker::PhantomData,
    sync::atomic::Ordering,
};
use sux::prelude::*;

type HashResult = u64;

fn min_alignment(bits: usize) -> String {
    if bits % 128 == 0 {
        "u128"
    } else if bits % 64 == 0 {
        "u64"
    } else if bits % 32 == 0 {
        "u32"
    } else if bits % 16 == 0 {
        "u16"
    } else {
        "u8"
    }
    .to_string()
}

/// Builder for [`HyperLogLogCounterArray`].
///
/// Create a builder with [`HyperLogLogCounterArrayBuilder::new`], edit parameters with
/// its methods, then call [`HyperLogLogCounterArrayBuilder::build`] on it to create
/// the [`HyperLogLogCounterArray`] as a [`Result`].
///
/// ```
/// # use webgraph_algo::utils::HyperLogLogCounterArrayBuilder;
/// # use crate::webgraph_algo::prelude::Counter;
/// # use anyhow::Result;
/// # fn main() -> Result<()> {
/// // Create a HyperLogLogCounterArray with 10 counters, each with
/// // 16 registers and an upper bound on the number of elements equal to 30
/// // and using a backend of usize.
/// // Type of the counter is usually inferred if the counter is used,
/// // otherwise it must be specified.
/// let counter_array = HyperLogLogCounterArrayBuilder::new()
///     .log_2_num_registers(6)
///     .num_elements_upper_bound(30)
///     .build(10)?;
/// counter_array.get_counter(0).add(42);
///
/// assert_eq!(counter_array.into_vec().len(), 10);
///
/// let counter_array = HyperLogLogCounterArrayBuilder::new()
///     .log_2_num_registers(6)
///     .num_elements_upper_bound(30)
///     .build::<usize>(10)?;
///
/// assert_eq!(counter_array.into_vec().len(), 10);
///
/// // The backend can also be changed to other unsigned types.
/// // Note that the type must be able to hold the result of the hash function.
/// let counter_array = HyperLogLogCounterArrayBuilder::new()
///     .log_2_num_registers(6)
///     .num_elements_upper_bound(30)
///     .word_type::<u64>()
///     .build::<usize>(10)?;
///
/// assert_eq!(counter_array.into_vec().len(), 10);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HyperLogLogCounterArrayBuilder<H: BuildHasher, W: Word + IntoAtomic> {
    log_2_num_registers: usize,
    num_elements: usize,
    mmap_options: TempMmapOptions,
    hasher_builder: H,
    word: PhantomData<W>,
}

impl HyperLogLogCounterArrayBuilder<BuildHasherDefault<DefaultHasher>, usize> {
    /// Creates a new builder for an [`HyperLogLogCounterArray`] with a default word type
    /// of [`usize`].
    pub fn new() -> Self {
        Self::new_with_word_type()
    }
}

impl<W: Word + IntoAtomic> HyperLogLogCounterArrayBuilder<BuildHasherDefault<DefaultHasher>, W> {
    /// Creates a new builder for an [`HyperLogLogCounterArray`] with a word type of `W`.
    pub fn new_with_word_type() -> Self {
        Self {
            log_2_num_registers: 4,
            num_elements: 1,
            hasher_builder: BuildHasherDefault::<DefaultHasher>::default(),
            mmap_options: TempMmapOptions::Default,
            word: PhantomData,
        }
    }
}

impl<H: BuildHasher, W: Word + IntoAtomic> HyperLogLogCounterArrayBuilder<H, W> {
    /// Sets the counters desired relative standard deviation.
    ///
    /// ## Note
    /// This is a high-level alternative to [`Self::with_log_2_num_registers`].
    /// Calling one after the other invalidates the work done by the first one.
    ///
    /// # Arguments
    /// * `rsd`: the relative standard deviation to be attained.
    pub fn rsd(self, rsd: f64) -> Self {
        self.log_2_num_registers(HyperLogLogCounterArray::log_2_number_of_registers(rsd))
    }

    /// Sets the log₂*m* number of registers for the array of counters.
    ///
    /// ## Note
    /// This is a low-level alternative to [`Self::with_rsd`].
    /// Calling one after the other invalidates the work done by the first one.
    ///
    /// # Arguments
    /// * `log_2_num_registers`: the logarithm of the number of registers per counter.
    pub fn log_2_num_registers(mut self, log_2_num_registers: usize) -> Self {
        self.log_2_num_registers = log_2_num_registers;
        self
    }

    /// Sets the upper bound on the number of distinct elements to be added to the
    /// counters.
    ///
    /// # Arguments
    /// * `num_elements`: an upper bound on the number of distinct elements.
    pub fn num_elements_upper_bound(mut self, num_elements: usize) -> Self {
        self.num_elements = num_elements;
        self
    }

    /// Sets the hasher builder to be used by the counters.
    ///
    /// # Arguments
    /// * `hasher_builder`: the builder of the hasher used by the array that implements
    ///   [`BuildHasher`].
    pub fn hasher_builder<H2: BuildHasher>(
        self,
        hasher_builder: H2,
    ) -> HyperLogLogCounterArrayBuilder<H2, W> {
        HyperLogLogCounterArrayBuilder {
            log_2_num_registers: self.log_2_num_registers,
            num_elements: self.num_elements,
            mmap_options: self.mmap_options,
            hasher_builder,
            word: PhantomData,
        }
    }

    /// Sets the memory options for the couters.
    ///
    /// # Arguments
    /// * `options`: the memory options for the backend of the counter array.
    pub fn mem_options(mut self, options: TempMmapOptions) -> Self {
        self.mmap_options = options;
        self
    }

    /// Sets the word type to be used by the counters.
    pub fn word_type<W2: Word + IntoAtomic>(self) -> HyperLogLogCounterArrayBuilder<H, W2> {
        HyperLogLogCounterArrayBuilder {
            log_2_num_registers: self.log_2_num_registers,
            num_elements: self.num_elements,
            mmap_options: self.mmap_options,
            hasher_builder: self.hasher_builder,
            word: PhantomData,
        }
    }

    /// Builds the counter array with the specified len, consuming the builder.
    ///
    /// The type of objects the counters keep track of is defined here by `T`, but
    /// it is usually inferred by the compiler.
    ///
    /// # Arguments
    /// * `len`: the length of the counter array in counters.
    pub fn build<T>(self, len: usize) -> Result<HyperLogLogCounterArray<T, W, H>> {
        let num_counters = len;
        let log_2_num_registers = self.log_2_num_registers;
        let num_elements = self.num_elements;
        let hasher_builder = self.hasher_builder;
        let mmap_options = self.mmap_options;

        // This ensures counters are at least 16-bit-aligned.
        ensure!(
            log_2_num_registers >= 4,
            "the logarithm of the number of registers per counter should be at least 4. Got {}",
            log_2_num_registers
        );

        let number_of_registers = 1 << log_2_num_registers;
        let register_size =
            HyperLogLogCounterArray::register_size_from_number_of_elements(num_elements);
        let sentinel_mask = 1 << ((1 << register_size) - 2);
        let alpha = match log_2_num_registers {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / number_of_registers as f64),
        };
        let num_registers_minus_1 = (number_of_registers - 1).try_into().unwrap_or_else(|_| {
            panic!(
                "should be able to convert {} from usize to the hash result type {}",
                number_of_registers - 1,
                std::any::type_name::<HashResult>()
            )
        });

        let counter_size_in_bits = number_of_registers * register_size;

        // This ensures counters are always aligned to W
        ensure!(
            counter_size_in_bits % W::BITS == 0,
            "W should allow counters to be aligned. Use {} or smaller words",
            min_alignment(counter_size_in_bits)
        );

        let mut msb = BitFieldVec::new(register_size, number_of_registers);
        let mut lsb = BitFieldVec::new(register_size, number_of_registers);
        let msb_w = W::ONE << (register_size - 1);
        let lsb_w = W::ONE;
        for i in 0..number_of_registers {
            msb.set(i, msb_w);
            lsb.set(i, lsb_w);
        }

        let required_words = std::cmp::max(
            1,
            (number_of_registers * num_counters * register_size + W::BITS - 1) / W::BITS,
        );
        let bits_vec =
            MmapSlice::from_closure(|| W::AtomicType::new(W::ZERO), required_words, mmap_options)
                .with_context(|| "Could not create bits for hyperloglog array as MmapSlice")?;
        debug_assert!(
            number_of_registers * num_counters * register_size <= bits_vec.len() * W::BITS
        );
        let bits = unsafe {
            AtomicBitFieldVec::from_raw_parts(
                bits_vec,
                register_size,
                number_of_registers * num_counters,
            )
        };

        let mut residual_mask = W::MAX;
        debug_assert_eq!(residual_mask.count_ones() as usize, W::BITS);
        if counter_size_in_bits % W::BITS != 0 {
            let residual_bits =
                ((counter_size_in_bits / W::BITS) + 1) * W::BITS - counter_size_in_bits;
            debug_assert!(residual_bits < W::BITS);
            residual_mask >>= residual_bits;
        }

        Ok(HyperLogLogCounterArray {
            bits,
            num_counters,
            num_registers: number_of_registers,
            num_registers_minus_1,
            log_2_num_registers,
            register_size,
            alpha_m_m: alpha * (number_of_registers as f64).powi(2),
            sentinel_mask,
            hasher_builder,
            msb_mask: msb,
            lsb_mask: lsb,
            residual_mask,
            _phantom_data: PhantomData,
        })
    }
}

impl<W: Word + IntoAtomic> Default
    for HyperLogLogCounterArrayBuilder<BuildHasherDefault<DefaultHasher>, W>
{
    fn default() -> Self {
        Self::new_with_word_type()
    }
}

/// An abstracted array of [`HyperLogLogCounter`].
///
/// This array is created using an [`AtomicBitFieldVec`] as a backend in order to avoid
/// wasting memory.
///
/// Individual counters can be accessed with the [`Self::get_counter`] method or concretized
/// as a [`Vec`] of [`HyperLogLogCounter`].
pub struct HyperLogLogCounterArray<
    T,
    W: Word + IntoAtomic = usize,
    H: BuildHasher = BuildHasherDefault<DefaultHasher>,
> {
    /// The bits of the registers
    bits: AtomicBitFieldVec<W, MmapSlice<W::AtomicType>>,
    /// The number of counters
    num_counters: usize,
    /// The number of registers per counter
    num_registers: usize,
    /// The number of registers per counter minus 1
    num_registers_minus_1: HashResult,
    /// The *log<sub>2</sub>* of the number of registers per counter
    log_2_num_registers: usize,
    /// The size in bits of each register
    register_size: usize,
    /// The correct value for αm<sup>2</sup>
    alpha_m_m: f64,
    /// The mask OR'd with the output of the hash function so that the number of trailing zeroes is not
    /// too large of a value
    sentinel_mask: HashResult,
    /// The builder of the hashers
    hasher_builder: H,
    /// A mask containing a one in the most significant bit of each register
    msb_mask: BitFieldVec<W>,
    /// A mask containing a one in the least significant bit of each register
    lsb_mask: BitFieldVec<W>,
    /// A mask with the residual bits of a counter set to 1
    residual_mask: W,
    _phantom_data: PhantomData<T>,
}

impl HyperLogLogCounterArray<()> {
    /// Returns the logarithm of the number of registers per counter that are necessary to attain a
    /// given relative stadard deviation.
    ///
    /// # Arguments
    /// * `rsd`: the relative standard deviation to be attained.
    pub fn log_2_number_of_registers(rsd: f64) -> usize {
        ((1.106 / rsd).pow(2.0)).log2().ceil() as usize
    }

    /// Returns the relative standard deviation corresponding to a given number of registers per counter.
    ///
    /// # Arguments
    /// * `log_2_num_registers`: the logarithm of the number of registers per counter.
    pub fn relative_standard_deviation(log_2_num_registers: usize) -> f64 {
        let tmp = match log_2_num_registers {
            4 => 1.106,
            5 => 1.070,
            6 => 1.054,
            7 => 1.046,
            _ => 1.04,
        };
        tmp / ((1 << log_2_num_registers) as f64).sqrt()
    }

    /// Returns the register size in bits, given an upper bound on the number of distinct elements.
    ///
    /// # Arguments
    /// * `n`: an upper bound on the number of distinct elements.
    pub fn register_size_from_number_of_elements(n: usize) -> usize {
        std::cmp::max(
            5,
            (((n as f64).ln() / 2.0.ln()) / 2.0.ln()).ln().ceil() as usize,
        )
    }
}

impl<T, W: Word + IntoAtomic, H: BuildHasher> HyperLogLogCounterArray<T, W, H>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Resets all counters by writing zeroes in all registers.
    pub fn clear(&mut self) {
        self.bits.reset_atomic(Ordering::Relaxed)
    }
}

impl<T, W: Word + IntoAtomic, H: BuildHasher> HyperLogLogCounterArray<T, W, H> {
    /// Returns the concretized [`HyperLogLogCounter`] with the specified index.
    ///
    /// # Arguments
    /// * `index`: the index of the counter to concretize.
    #[inline(always)]
    pub fn get_counter(&self, index: usize) -> HyperLogLogCounter<T, W, H> {
        assert!(index < self.num_counters);
        HyperLogLogCounter {
            counter_array: self,
            offset: index * self.num_registers,
            cached_bits: None,
            thread_helper: None,
        }
    }

    /// Creates a thread helper for a counter of this array.
    pub fn get_thread_helper(&self) -> ThreadHelper<W> {
        ThreadHelper {
            acc: Vec::with_capacity(self.words_per_counter()),
            mask: Vec::with_capacity(self.words_per_counter()),
        }
    }

    /// Returns the number of words `W` per counter.
    #[inline(always)]
    pub fn words_per_counter(&self) -> usize {
        self.msb_mask.as_slice().len()
    }

    /// Swaps the undelying bits with those of aother equivalent array.
    ///
    /// # Arguments
    /// * `other`: the array to swap bits with
    pub fn swap_with(&mut self, other: &mut Self) {
        assert_eq!(self.num_counters, other.num_counters);
        assert_eq!(self.num_registers, other.num_registers);
        assert_eq!(self.register_size, other.register_size);
        std::mem::swap(&mut self.bits, &mut other.bits);
    }

    /// Returns the register size.
    #[inline(always)]
    pub fn register_size(&self) -> usize {
        self.register_size
    }

    /// Returns the number of registers per counter.
    #[inline(always)]
    pub fn num_registers(&self) -> usize {
        self.num_registers
    }

    /// Returns the log₂ of the number of registers per counter.
    #[inline(always)]
    pub fn log_2_num_registers(&self) -> usize {
        self.log_2_num_registers
    }
}

impl<T: Sync, W: Word + IntoAtomic, H: BuildHasher + Sync> HyperLogLogCounterArray<T, W, H> {
    /// Creates a [`Vec`] where `v[i]` is the [`HyperLogLogCounter`] with index `i`.
    pub fn into_vec(&self) -> Vec<HyperLogLogCounter<T, W, H>> {
        let mut vec = Vec::with_capacity(self.num_counters);
        (0..self.num_counters)
            .into_par_iter()
            .map(|i| self.get_counter(i))
            .collect_into_vec(&mut vec);
        vec
    }
}

/// Utility struct for parallel optimization.
pub struct ThreadHelper<W: Word + IntoAtomic> {
    acc: Vec<W>,
    mask: Vec<W>,
}

/// Concretized counter for [`HyperLogLogCounterArray`].
///
/// Each counter holds only basic information in order to reduce memory usage.
/// In particular each counter holds a shared reference to the parent [`HyperLogLogCounterArray`]
/// and the offset of the first register of the counter.
///
/// In alternative, the counter may make a local copy of the registers using the
/// [`Self::cache`] method.
pub struct HyperLogLogCounter<'a, T, W: Word + IntoAtomic, H: BuildHasher> {
    /// The reference to the parent [`HyperLogLogCounterArray`].
    counter_array: &'a HyperLogLogCounterArray<T, W, H>,
    /// The offset of the first register of the counter.
    offset: usize,
    /// The cached counter bits. Remeinder bits are to be considered noise and not used.
    /// The boolean value is [`true`] if the cache has been modified and needs to be
    /// committed to the backend.
    cached_bits: Option<(BitFieldVec<W>, bool)>,
    /// Reference to an already allocated cache to help reduce allocations in parallel
    /// executions.
    thread_helper: Option<&'a mut ThreadHelper<W>>,
}

impl<'a, T, W: Word + IntoAtomic, H: BuildHasher> HyperLogLogCounter<'a, T, W, H> {
    /// Returns the index of the current counter
    #[inline(always)]
    pub fn counter_index(&self) -> usize {
        // self.offset / self.counter_array.num_registers
        self.offset >> self.counter_array.log_2_num_registers
    }

    /// Returns whether the counter's cache has been modified and should be committed to the backend.
    #[inline(always)]
    pub fn is_changed(&self) -> bool {
        if let Some((_, changed)) = self.cached_bits {
            changed
        } else {
            false
        }
    }

    /// Returns whether the counter is cached or not.
    #[inline(always)]
    pub fn is_cached(&self) -> bool {
        self.cached_bits.is_some()
    }

    /// Performs a multiple precision subtraction, leaving the result in the first operand.
    /// The operands MUST have the same length.
    ///
    /// # Arguments
    /// * `x`: the first operand. This will contain the final result.
    /// * `y`: the second operand that will be subtracted from `x`.
    #[inline(always)]
    fn subtract(x: &mut [W], y: &[W]) {
        debug_assert_eq!(x.len(), y.len());
        let mut borrow = false;

        for (x_word, &y) in x.iter_mut().zip(y.iter()) {
            let mut x = *x_word;
            if !borrow {
                borrow = x < y;
            } else if x != W::ZERO {
                x = x.wrapping_sub(W::ONE);
                borrow = x < y;
            } else {
                x = x.wrapping_sub(W::ONE);
            }
            *x_word = x.wrapping_sub(y);
        }
    }

    /// Merges `other` into `self` inplace using words instead of registers and returns
    /// whether `self` was modified.
    ///
    /// `other` is not modified but `self` can be.
    ///
    /// # Arguments
    /// * `other`: the counter to merge into `self`.
    ///
    /// # Safety
    ///
    /// Calling this method while reading (ie. with [`Self::cache`] on the same counter from
    /// another instance) or writing (ie. with [`Self::commit_changes`]) from the same memory
    /// zones in the backend [`HyperLogLogCounterArray`] is [undefined behavior].
    ///
    /// Calling this method on the same counters at the same time in
    /// different directions without first calling [`Self::cache`] as
    /// is shown below is [undefined behavior]:
    /// ```no_run
    /// # use rayon::join;
    /// # use webgraph_algo::utils::HyperLogLogCounterArrayBuilder;
    /// # use webgraph_algo::prelude::Counter;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let counters = HyperLogLogCounterArrayBuilder::new()
    ///     .rsd(0.1)
    ///     .num_elements_upper_bound(10)
    ///     .build(2)?;
    /// let mut c1 = counters.get_counter(0);
    /// let mut c2 = counters.get_counter(1);
    /// let c1_shared = counters.get_counter(0);
    /// let c2_shared = counters.get_counter(1);
    /// # counters.get_counter(0).add(0);
    ///
    /// // This is undefined behavior
    /// join(|| unsafe {c1.merge_unsafe(&c2_shared)}, || unsafe {c2.merge_unsafe(&c1_shared)});
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// On the other hand, once the counter is cached it is fine:
    ///
    /// ```
    /// # use rayon::join;
    /// # use webgraph_algo::utils::HyperLogLogCounterArrayBuilder;
    /// # use webgraph_algo::prelude::Counter;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let counters = HyperLogLogCounterArrayBuilder::new()
    ///     .rsd(0.1)
    ///     .num_elements_upper_bound(10)
    ///     .build(2)?;
    /// let mut c1 = counters.get_counter(0);
    /// let mut c2 = counters.get_counter(1);
    /// let c1_shared = counters.get_counter(0);
    /// let c2_shared = counters.get_counter(1);
    /// # counters.get_counter(0).add(0);
    ///
    /// unsafe {
    ///     c1.cache();
    ///     c2.cache();
    /// }
    ///
    /// // This is fine
    /// join(|| unsafe {c1.merge_unsafe(&c2_shared)}, || unsafe {c2.merge_unsafe(&c1_shared)});
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn merge_unsafe(&mut self, other: &Self) -> bool {
        // The temporary vectors if no thread helper is used
        let mut acc_internal;
        let mut mask_internal;

        let num_words = self.counter_array.words_per_counter();
        let num_words_minus_1 = num_words - 1;
        let register_size_minus_1 = self.counter_array.register_size - 1;
        let shift_register_size_minus_1 = W::BITS - register_size_minus_1;
        let last_word_mask = self.counter_array.residual_mask;

        let msb_mask = self.counter_array.msb_mask.as_slice();
        let lsb_mask = self.counter_array.lsb_mask.as_slice();
        let (x, mut changed) = match &mut self.cached_bits {
            Some((bits, c)) => (bits.as_mut_slice(), *c),
            None => {
                let bits_offset = self.offset * self.counter_array.register_size;
                // Counters should be byte-aligned
                debug_assert!(bits_offset % 8 == 0);
                let byte_offset = bits_offset / 8;
                let num_bytes = num_words * W::BYTES;
                // We should copy whole words, not parts
                debug_assert!((num_bytes * 8) % W::BITS == 0);

                let pointer =
                    (other.counter_array.bits.as_slice().as_ptr() as *mut W).byte_add(byte_offset);
                debug_assert!(pointer.is_aligned());

                (std::slice::from_raw_parts_mut(pointer, num_words), false)
            }
        };
        let (acc, mask) = if let Some(helper) = &mut self.thread_helper {
            helper.acc.set_len(0);
            helper.mask.set_len(0);
            (&mut helper.acc, &mut helper.mask)
        } else {
            acc_internal = Vec::with_capacity(num_words);
            mask_internal = Vec::with_capacity(num_words);
            (&mut acc_internal, &mut mask_internal)
        };
        let y = match &other.cached_bits {
            Some((bits, _)) => bits.as_slice(),
            None => {
                let bits_offset = other.offset * self.counter_array.register_size;
                // Counters should be byte-aligned
                debug_assert!(bits_offset % 8 == 0);
                let byte_offset = bits_offset / 8;
                let num_bytes = num_words * W::BYTES;
                // We should copy whole words, not parts
                debug_assert!((num_bytes * 8) % W::BITS == 0);

                let pointer = (other.counter_array.bits.as_slice().as_ptr() as *const W)
                    .byte_add(byte_offset);
                debug_assert!(pointer.is_aligned());

                std::slice::from_raw_parts(pointer, num_words)
            }
        };

        // We split x, y and the masks so we treat the last word appropriately.
        let (x_last, x_slice) = x.split_last_mut().unwrap_unchecked();
        let x_last_masked = *x_last & last_word_mask;
        let (&y_last, y_slice) = y.split_last().unwrap_unchecked();
        let y_last_masked = y_last & last_word_mask;
        let (&msb_last, msb_slice) = msb_mask.split_last().unwrap_unchecked();

        /* We work in two phases. Let H_r (msb_mask) be the mask with the
         * highest bit of each register (of size r) set, and L_r (lsb_mask)
         * be the mask with the lowest bit of each register set.
         * We describe the algorithm on a single word.
         *
         * In the first phase we perform an unsigned strict register-by-register
         * comparison of x and y, using the formula
         *
         * z = ((((y | H_r) - (x & !H_r)) | (y ^ x)) ^ (y | !x)) & H_r
         *
         * Then, we generate a register-by-register mask of all ones or
         * all zeroes, depending on the result of the comparison, using the
         * formula
         *
         * (((z >> r-1 | H_r) - L_r) | H_r) ^ z
         *
         * At that point, it is trivial to select from x and y the right values.
         */

        // We load y | H_r into the accumulator.
        acc.extend(
            y_slice
                .iter()
                .zip(msb_slice)
                .map(|(&y_word, &msb_word)| y_word | msb_word),
        );
        acc.push(y_last_masked | msb_last);

        // We load x & !H_r into mask as temporary storage.
        mask.extend(
            x_slice
                .iter()
                .zip(msb_slice)
                .map(|(x_word, &msb_word)| x_word & !msb_word),
        );
        mask.push(x_last_masked & !msb_last);

        // We subtract x & !H_r, using mask as temporary storage
        Self::subtract(acc, mask);

        // We OR with y ^ x, XOR with (y | !x), and finally AND with H_r.
        {
            let (acc_last, acc_slice) = acc.split_last_mut().unwrap_unchecked();
            acc_slice
                .iter_mut()
                .zip(x_slice.iter())
                .zip(y_slice.iter())
                .zip(msb_slice.iter())
                .for_each(|(((acc_word, x_word), &y_word), &msb_word)| {
                    *acc_word = ((*acc_word | (y_word ^ x_word)) ^ (y_word | !x_word)) & msb_word
                });
            *acc_last = ((*acc_last | (y_last_masked ^ x_last_masked))
                ^ (y_last_masked | !x_last_masked))
                & msb_last;
        }

        // We shift by register_size - 1 places and put the result into mask.
        {
            let (mask_last, mask_slice) = mask.split_last_mut().unwrap_unchecked();
            mask_slice
                .iter_mut()
                .zip(acc[0..num_words_minus_1].iter())
                .zip(acc[1..].iter())
                .zip(msb_slice.iter())
                .rev()
                .for_each(|(((mask_word, &acc_word), &next_acc_word), &msb_word)| {
                    // W is always unsigned so the shift is always with a 0
                    *mask_word = (acc_word >> register_size_minus_1)
                        | (next_acc_word << shift_register_size_minus_1)
                        | msb_word
                });
            *mask_last = (acc[num_words_minus_1] >> register_size_minus_1) | msb_last;
        }

        // We subtract L_r from mask.
        Self::subtract(mask, lsb_mask);

        // We OR with H_r and XOR with the accumulator.
        let (mask_last, mask_slice) = mask.split_last_mut().unwrap_unchecked();
        let (&acc_last, acc_slice) = acc.split_last().unwrap_unchecked();
        mask_slice
            .iter_mut()
            .zip(msb_slice.iter())
            .zip(acc_slice.iter())
            .for_each(|((mask_word, &msb_word), &acc_word)| {
                *mask_word = (*mask_word | msb_word) ^ acc_word
            });
        *mask_last = (*mask_last | msb_last) ^ acc_last;

        // Finally, we use mask to select the right bits from x and y and store the result.
        if changed {
            x_slice
                .iter_mut()
                .zip(y_slice.iter())
                .zip(mask_slice.iter())
                .for_each(|((x_word, &y_word), mask_word)| {
                    *x_word = *x_word ^ ((*x_word ^ y_word) & mask_word);
                });
            *x_last = (*x_last & !last_word_mask)
                | (x_last_masked ^ ((x_last_masked ^ y_last_masked) & *mask_last));
        } else {
            x_slice
                .iter_mut()
                .zip(y_slice.iter())
                .zip(mask_slice.iter())
                .for_each(|((x_word, &y_word), mask_word)| {
                    let new_x_word = *x_word ^ ((*x_word ^ y_word) & mask_word);
                    if new_x_word != *x_word {
                        changed = true;
                        *x_word = new_x_word;
                    }
                });
            let new_x_last = (*x_last & !last_word_mask)
                | (x_last_masked ^ ((x_last_masked ^ y_last_masked) & *mask_last));
            if new_x_last != *x_last {
                changed = true;
                *x_last = new_x_last;
            }

            if changed {
                if let Some((_, cache_changed)) = self.cached_bits.as_mut() {
                    *cache_changed = changed;
                }
            }
        }

        changed
    }

    /// Commits changes to this counter to the backend [`HyperLogLogCounterArray`].
    ///
    /// Calling this method on a counter whose registers aren't cached with [`Self::cache`]
    /// or whose local cache isn't changed will result in a panic.
    ///
    /// # Arguments
    /// * `keep_cached`: whether to keep the counter cached or to return to a non-cached one.
    ///
    /// # Safety
    ///
    /// Calling this method while reading from the same memory zone in the backend
    /// [`HyperLogLogCounterArray`] (ie. with [`Self::cache`] on the same counter from
    /// another instance) is [undefined behavior].
    /// ```no_run
    /// # use rayon::join;
    /// # use webgraph_algo::utils::HyperLogLogCounterArrayBuilder;
    /// # use webgraph_algo::prelude::Counter;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let counters = HyperLogLogCounterArrayBuilder::new()
    ///     .rsd(0.1)
    ///     .num_elements_upper_bound(10)
    ///     .build(2)?;
    /// let mut c1 = counters.get_counter(0);
    /// let mut c1_copy = counters.get_counter(0);
    ///
    /// unsafe { c1.cache() };
    /// c1.add(0);
    ///
    /// // This is undefined behavior
    /// join(|| unsafe {c1.commit_changes(false)}, || unsafe {c1_copy.cache()});
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn commit_changes(&mut self, keep_cached: bool) {
        assert!(self.cached_bits.is_some());
        assert!(self.is_changed());

        let cached = self.cached_bits.as_ref().unwrap().0.as_slice();

        let bits_to_write = self.counter_array.num_registers * self.counter_array.register_size;
        debug_assert!((W::BITS * cached.len()) - bits_to_write < W::BITS);
        debug_assert!(bits_to_write % 8 == 0);
        debug_assert_eq!(cached.len(), self.counter_array.words_per_counter());
        let bytes_to_write = bits_to_write / 8;

        let bits_offset = self.offset * self.counter_array.register_size;
        debug_assert!(bits_offset % 8 == 0);
        let byte_offset = bits_offset / 8;

        let pointer =
            (self.counter_array.bits.as_slice().as_ptr() as *mut u8).byte_add(byte_offset);

        std::ptr::copy_nonoverlapping(cached.as_ptr() as *const u8, pointer, bytes_to_write);

        if keep_cached {
            if let Some((_, changed)) = self.cached_bits.as_mut() {
                *changed = false;
            }
        } else {
            self.cached_bits = None;
        }
    }

    /// Commits changes to this counter to the backend [`HyperLogLogCounterArray`].
    ///
    /// This is a shorthand for `self.commit_changes(true)`.
    ///
    /// Calling this method on a counter whose registers aren't cached with [`Self::cache`]
    /// or whose local cache isn't changed will result in a panic.
    ///
    /// # Safety
    ///
    /// Calling this method while reading from the same memory zone in the backend
    /// [`HyperLogLogCounterArray`] (ie. with [`Self::cache`] on the same counter from
    /// another instance) is [undefined behavior].
    #[inline(always)]
    pub unsafe fn sync_to_backend(&mut self) {
        self.commit_changes(true);
    }

    /// Cache the counter's registers.
    ///
    /// Once this method is called every change applied to this counter isn't reflected
    /// in the backend [`HyperLogLogCounterArray`] until [`Self::commit_changes`] is
    /// called.
    ///
    /// # Safety
    ///
    /// Calling this method while writing to the same memory zone in the backend
    /// [`HyperLogLogCounterArray`] (ie. with [`Self::commit_changes`] on the same counter from
    /// another instance) is [undefined behavior].
    /// ```no_run
    /// # use rayon::join;
    /// # use webgraph_algo::utils::HyperLogLogCounterArrayBuilder;
    /// # use webgraph_algo::prelude::Counter;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let counters = HyperLogLogCounterArrayBuilder::new()
    ///     .rsd(0.1)
    ///     .num_elements_upper_bound(10)
    ///     .build(2)?;
    /// let mut c1 = counters.get_counter(0);
    /// let mut c1_copy = counters.get_counter(0);
    ///
    /// unsafe { c1.cache() };
    /// c1.add(0);
    ///
    /// // This is undefined behavior
    /// join(|| unsafe {c1.commit_changes(false)}, || unsafe {c1_copy.cache()});
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn cache(&mut self) {
        let bits_offset = self.offset * self.counter_array.register_size;
        // Counters should be byte-aligned
        debug_assert!(bits_offset % 8 == 0);
        let byte_offset = bits_offset / 8;
        let num_words = self.counter_array.words_per_counter();
        let num_bytes = num_words * W::BYTES;
        // We should copy whole words, not parts
        debug_assert!((num_bytes * 8) % W::BITS == 0);

        let pointer =
            (self.counter_array.bits.as_slice().as_ptr() as *const u8).byte_add(byte_offset);

        let mut v = Vec::with_capacity(num_words);
        std::ptr::copy_nonoverlapping(pointer, v.as_mut_ptr() as *mut u8, num_bytes);
        v.set_len(num_words);

        self.cached_bits = Some((
            BitFieldVec::from_raw_parts(
                v,
                self.counter_array.register_size,
                self.counter_array.num_registers,
            ),
            false,
        ));
    }

    /// Sets the content of the counter to the content of the passed counter.
    ///
    /// # Arguments
    /// * `counter`: the counter from which to copy the contents.
    ///
    /// # Safety
    ///
    /// Calling this method while reading from the same memory zone in the backend
    /// [`HyperLogLogCounterArray`] (ie. with [`Self::cache`] on the same counter from
    /// another instance) is [undefined behavior].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn set_to(&mut self, counter: &Self) {
        debug_assert_eq!(
            self.counter_array.register_size,
            counter.counter_array.register_size
        );
        debug_assert_eq!(
            self.counter_array.num_registers,
            counter.counter_array.num_registers
        );
        debug_assert_eq!(
            self.counter_array.words_per_counter(),
            counter.counter_array.words_per_counter()
        );
        debug_assert_eq!(
            self.counter_array.residual_mask,
            counter.counter_array.residual_mask
        );

        let bits_to_copy = self.counter_array.num_registers * self.counter_array.register_size;
        debug_assert!(bits_to_copy % 8 == 0);
        let bytes_to_copy = bits_to_copy / 8;

        let bits_offset = counter.offset * self.counter_array.register_size;
        // Counters should be byte-aligned
        debug_assert!(bits_offset % 8 == 0);
        let byte_offset = bits_offset / 8;

        let counter_pointer = if let Some((cached_bits, _)) = &counter.cached_bits {
            cached_bits.as_slice().as_ptr() as *const u8
        } else {
            (counter.counter_array.bits.as_slice().as_ptr() as *const u8).byte_add(byte_offset)
        };

        match &mut self.cached_bits {
            Some((bits, changed)) => {
                let cache_pointer = bits.as_mut_slice().as_mut_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(counter_pointer, cache_pointer, bytes_to_copy);

                let backend_pointer =
                    (self.counter_array.bits.as_slice().as_ptr() as *mut u8).byte_add(byte_offset);
                let backend_slice = std::slice::from_raw_parts(backend_pointer, bytes_to_copy);
                let cache_slice = std::slice::from_raw_parts(
                    bits.as_slice().as_ptr() as *const u8,
                    bytes_to_copy,
                );

                *changed = backend_slice == cache_slice;
            }
            None => {
                let bits_offset = self.offset * self.counter_array.register_size;
                // Counters should be byte-aligned
                debug_assert!(bits_offset % 8 == 0);
                let byte_offset = bits_offset / 8;

                let backend_pointer =
                    (self.counter_array.bits.as_slice().as_ptr() as *mut u8).byte_add(byte_offset);

                std::ptr::copy_nonoverlapping(counter_pointer, backend_pointer, bytes_to_copy);
            }
        }
    }

    /// Sets the couter to use the specified thread helper.
    #[inline(always)]
    pub fn use_thread_helper(&mut self, helper: &'a mut ThreadHelper<W>) {
        self.thread_helper = Some(helper);
    }

    /// Stops the counter from using the thread helper.
    #[inline(always)]
    pub fn remove_thread_helper(&mut self) {
        self.thread_helper = None;
    }
}

impl<'a, T, W: Word + IntoAtomic, H: BuildHasher> HyperLogLogCounter<'a, T, W, H>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Sets a register of the counter to the specified new value.
    ///
    /// If the counter is cached the new value isn't propagated to the backend
    /// [`HyperLogLogCounterArray`] until [`Self::commit_changes`] is called on
    /// this counter.
    ///
    /// # Arguments
    /// * `index`: the index of the register to edit.
    /// * `new_value`: the new value to store in the register.
    #[inline(always)]
    fn set_register(&mut self, index: usize, new_value: W) {
        match &mut self.cached_bits {
            Some((bits, changed)) => {
                let old_value = bits.get(index);
                if old_value != new_value {
                    *changed = true;
                    bits.set(index, new_value)
                }
            }
            None => self.counter_array.bits.set_atomic(
                self.offset + index,
                new_value,
                Ordering::Relaxed,
            ),
        }
    }

    /// Gets the current value of the specified register.
    ///
    /// If the counter is cached and has been modified, this methods returns
    /// the value present in the local cache, not the one present in the
    /// backend.
    ///
    /// # Arguments
    /// * `index`: the index of the register to read.
    #[inline(always)]
    fn get_register(&self, index: usize) -> W {
        match &self.cached_bits {
            Some((bits, _)) => bits.get(index),
            None => self
                .counter_array
                .bits
                .get_atomic(self.offset + index, Ordering::Relaxed),
        }
    }
}

impl<
        'a,
        T: Hash,
        W: Word + TryFrom<HashResult> + UpcastableInto<HashResult> + IntoAtomic,
        H: BuildHasher,
    > Counter<T> for HyperLogLogCounter<'a, T, W, H>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    #[inline]
    fn add(&mut self, element: T) {
        let x = self.counter_array.hasher_builder.hash_one(element);
        let j = x & self.counter_array.num_registers_minus_1;
        let r = (x >> self.counter_array.log_2_num_registers | self.counter_array.sentinel_mask)
            .trailing_zeros() as HashResult;
        let register = j as usize;

        debug_assert!(r < (1 << self.counter_array.register_size) - 1);
        debug_assert!(register < self.counter_array.num_registers);

        let current_value = self.get_register(register);
        let candidate_value = r + 1;
        let new_value = std::cmp::max(
            current_value,
            candidate_value.try_into().unwrap_or_else(|_| {
                panic!(
                    "Should be able to convert {} from hash result type {} to word type {}.",
                    candidate_value,
                    std::any::type_name::<HashResult>(),
                    std::any::type_name::<W>()
                )
            }),
        );
        if current_value != new_value {
            self.set_register(register, new_value);
        }
    }

    #[inline]
    fn count(&self) -> u64 {
        self.estimate_count().round() as u64
    }

    #[inline]
    fn clear(&mut self) {
        for i in 0..self.counter_array.num_registers {
            self.set_register(i, W::ZERO);
        }
    }

    #[inline]
    fn merge(&mut self, other: &Self) {
        assert_eq!(
            self.counter_array.num_registers,
            other.counter_array.num_registers
        );
        assert_eq!(
            self.counter_array.register_size,
            other.counter_array.register_size
        );
        for i in 0..self.counter_array.num_registers {
            let current_value = self.get_register(i);
            let other_value = other.get_register(i);

            if other_value > current_value {
                self.set_register(i, other_value);
            }
        }
    }
}

impl<
        'a,
        T: Hash,
        W: Word + TryFrom<HashResult> + UpcastableInto<HashResult> + IntoAtomic,
        H: BuildHasher,
    > ApproximatedCounter<T> for HyperLogLogCounter<'a, T, W, H>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    #[inline]
    fn estimate_count(&self) -> f64 {
        let mut harmonic_mean = 0.0;
        let mut zeroes = 0;

        for i in 0..self.counter_array.num_registers {
            let value = self.get_register(i).upcast();
            if value == 0 {
                zeroes += 1;
            }
            harmonic_mean += 1.0 / (1 << value) as f64;
        }

        let mut estimate = self.counter_array.alpha_m_m / harmonic_mean;
        if zeroes != 0 && estimate < 2.5 * self.counter_array.num_registers as f64 {
            estimate = self.counter_array.num_registers as f64
                * (self.counter_array.num_registers as f64 / zeroes as f64).ln();
        }
        estimate
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_counter_creation() -> Result<()> {
        let counters = HyperLogLogCounterArrayBuilder::new()
            .log_2_num_registers(6)
            .build::<usize>(10)?;

        let counter_4 = counters.get_counter(4);

        assert!(counter_4.cached_bits.is_none());
        assert_eq!(counter_4.offset, 2.pow(6) * 4);

        Ok(())
    }

    #[test]
    fn test_counter_edit_inplace() -> Result<()> {
        let counters = HyperLogLogCounterArrayBuilder::new()
            .log_2_num_registers(6)
            .word_type::<u64>()
            .build(10)?;
        let mut counter_4 = counters.get_counter(4);

        counter_4.add(42);

        assert_eq!(counters.residual_mask.count_ones(), 64);
        assert!(counter_4.cached_bits.is_none());
        for i in 0..10 {
            let pointer = counters.bits.as_slice().as_ptr() as *const u64;
            let slice = unsafe {
                let ptr = pointer.add(i * counters.words_per_counter());
                std::slice::from_raw_parts(ptr, counters.words_per_counter())
            };
            if i != 4 {
                assert_eq!(slice, vec![0; counters.words_per_counter()]);
            } else {
                let mut ones = 0;
                for word in slice {
                    ones += word.count_ones();
                }
                assert!(ones > 0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_counter_edit_cached() -> Result<()> {
        let counters = HyperLogLogCounterArrayBuilder::new()
            .log_2_num_registers(6)
            .word_type::<u64>()
            .build(10)?;
        let mut counter_4 = counters.get_counter(4);

        assert!(counter_4.cached_bits.is_none());

        unsafe {
            counter_4.cache();
        }

        assert!(counter_4.cached_bits.is_some());
        assert!(!counter_4.cached_bits.as_ref().unwrap().1);
        assert_eq!(
            counter_4.cached_bits.as_ref().unwrap().0.as_slice(),
            vec![0; counters.words_per_counter()]
        );

        counter_4.add(42);

        assert!(counter_4.cached_bits.as_ref().unwrap().1);
        for i in 0..10 {
            let pointer = counters.bits.as_slice().as_ptr() as *const u64;
            let slice = unsafe {
                let ptr = pointer.add(i * counters.words_per_counter());
                std::slice::from_raw_parts(ptr, counters.words_per_counter())
            };
            assert_eq!(slice, vec![0; counters.words_per_counter()]);
        }

        let mut ones = 0;
        for word in counter_4.cached_bits.as_ref().unwrap().0.as_slice() {
            ones += word.count_ones();
        }
        assert!(ones > 0);

        Ok(())
    }

    #[test]
    fn test_counter_commit_changes() -> Result<()> {
        let counters = HyperLogLogCounterArrayBuilder::new()
            .log_2_num_registers(6)
            .word_type::<u64>()
            .build(10)?;
        let mut counter_4 = counters.get_counter(4);
        unsafe {
            counter_4.cache();
        }
        counter_4.add(42);

        unsafe {
            counter_4.commit_changes(false);
        }

        assert!(counter_4.cached_bits.is_none());
        for i in 0..10 {
            let pointer = counters.bits.as_slice().as_ptr() as *const u64;
            let slice = unsafe {
                let ptr = pointer.add(i * counters.words_per_counter());
                std::slice::from_raw_parts(ptr, counters.words_per_counter())
            };
            if i != 4 {
                assert_eq!(slice, vec![0; counters.words_per_counter()]);
            } else {
                let mut ones = 0;
                for word in slice {
                    ones += word.count_ones();
                }
                assert!(ones > 0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_counter_sync_changes() -> Result<()> {
        let counters = HyperLogLogCounterArrayBuilder::new()
            .log_2_num_registers(6)
            .word_type::<u64>()
            .build(10)?;
        let mut counter_4 = counters.get_counter(4);
        unsafe {
            counter_4.cache();
        }
        counter_4.add(42);

        unsafe {
            counter_4.commit_changes(true);
        }

        assert!(counter_4.cached_bits.is_some());
        assert!(!counter_4.cached_bits.as_ref().unwrap().1);
        for i in 0..10 {
            let pointer = counters.bits.as_slice().as_ptr() as *const u64;
            let slice = unsafe {
                let ptr = pointer.add(i * counters.words_per_counter());
                std::slice::from_raw_parts(ptr, counters.words_per_counter())
            };
            if i != 4 {
                assert_eq!(slice, vec![0; counters.words_per_counter()]);
            } else {
                assert_eq!(slice, counter_4.cached_bits.as_ref().unwrap().0.as_slice())
            }
        }

        Ok(())
    }

    #[test]
    fn test_counter_sync() -> Result<()> {
        let counters = HyperLogLogCounterArrayBuilder::new()
            .log_2_num_registers(6)
            .word_type::<u64>()
            .build(10)?;
        let mut counter_4 = counters.get_counter(4);
        unsafe {
            counter_4.cache();
        }
        counter_4.add(42);

        unsafe {
            counter_4.sync_to_backend();
        }

        assert!(counter_4.cached_bits.is_some());
        assert!(!counter_4.cached_bits.as_ref().unwrap().1);
        for i in 0..10 {
            let pointer = counters.bits.as_slice().as_ptr() as *const u64;
            let slice = unsafe {
                let ptr = pointer.add(i * counters.words_per_counter());
                std::slice::from_raw_parts(ptr, counters.words_per_counter())
            };
            if i != 4 {
                assert_eq!(slice, vec![0; counters.words_per_counter()]);
            } else {
                assert_eq!(slice, counter_4.cached_bits.as_ref().unwrap().0.as_slice())
            }
        }

        Ok(())
    }
}
