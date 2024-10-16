use crate::utils::closure_vec;
use anyhow::{ensure, Context, Result};
use mmap_rs::{MmapMut, MmapOptions};
use std::{
    fs::File,
    mem::size_of,
    ops::{Deref, DerefMut},
    path::PathBuf,
};
use tempfile::{tempfile, tempfile_in};

#[doc(hidden)]
pub use mmap_rs::MmapFlags;

/// Options for [`MmapSlice`].
/// This determines where data is stored.
#[derive(Debug, Clone)]
pub enum TempMmapOptions {
    /// Data is stored in a [`Vec`] in RAM.
    None,
    /// Data is stored in a tempfile created with [`tempfile::tempfile`] and is memory mapped
    /// using the provided [`MmapFlags`].
    TempDir(MmapFlags),
    /// Data is stored in a tempfile created with [`tempfile::tempfile_in`] using the provided
    /// path and is memory mapped using the provided [`MmapFlags`].
    CustomDir(PathBuf, MmapFlags),
}

/// A utility struct to reduce RAM consumption by allowing storing data in persistent memory and
/// memory mapping it.
///
/// This still allows to choose to keep everything in ram by using [`TempMmapOptions::None`]
///
/// # Examples
///
/// You can create a [`MmapSlice`] from a value.
/// Note that all constructors return a [`Result`].
///
/// ```
/// use webgraph_algo::utils::*;
/// # use anyhow::Result;
///
/// #
/// # fn main() -> Result<()> {
/// // Create a slice of 100 elements initialized to 42 (type is usually inferred) stored in RAM
/// let mmap_slice: MmapSlice<usize> = MmapSlice::from_value(42, 100, TempMmapOptions::None)?;
/// # assert_eq!(mmap_slice.as_slice(), vec![42; 100].as_slice());
///
/// // Create a slice of 100 elements initialized to the type's default (type can be inferred
/// // but usually isn't) stored in RAM
/// let mmap_slice: MmapSlice<usize> = MmapSlice::new(100, TempMmapOptions::None)?;
/// # assert_eq!(mmap_slice.as_slice(), vec![0; 100].as_slice());
///
/// // Create a slice from a vec stored in a temporary file created in the default temporary
/// // directory with the SHARED and RANDOM_ACCESS flags (types are explicit only for clarity)
/// let v: Vec<usize> = (0..100).collect();
/// # let c = v.clone();
///
/// let mut flags = MmapFlags::empty();
/// flags.set(MmapFlags::SHARED, true);
/// flags.set(MmapFlags::RANDOM_ACCESS, true);
///
/// let mmap_slice: MmapSlice<usize> = MmapSlice::from_vec(v, TempMmapOptions::TempDir(flags))?;
/// # assert_eq!(mmap_slice.as_slice(), c.as_slice());
/// # Ok(())
/// # }
///
/// ```
pub struct MmapSlice<T> {
    /// The memory map if used
    mmap: Option<(File, MmapMut, usize)>,
    /// The in memory vector. Empty if not used or if using an empty slice.
    in_memory_vec: Vec<T>,
}

impl<T: Default> MmapSlice<T> {
    /// Creates a new slice of length `len` with the provided [`TempMmapOptions`] and with all
    /// the elements initialized to the type's default value.
    ///
    /// # Examples
    ///
    /// ```
    /// use webgraph_algo::utils::*;
    ///
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let slice: MmapSlice<usize> = MmapSlice::new(100, TempMmapOptions::None)?;
    /// # assert_eq!(slice.as_slice(), vec![0usize; 100].as_slice());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Note how the type is not often inferred with this constructor.
    ///
    /// The type may also be specified with the turbofish syntax.
    ///
    /// ```
    /// use webgraph_algo::utils::*;
    ///
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let slice = MmapSlice::<usize>::new(100, TempMmapOptions::None)?;
    /// # assert_eq!(slice.as_slice(), vec![0usize; 100].as_slice());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(len: usize, options: TempMmapOptions) -> Result<Self> {
        Self::from_closure(T::default, len, options)
    }
}

impl<T: Clone> MmapSlice<T> {
    /// Creates a new slice of length `len` with the provided [`TempMmapOptions`] and with all
    /// the elements initialized to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use webgraph_algo::utils::*;
    ///
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let slice = MmapSlice::from_value(0, 100, TempMmapOptions::None)?;
    /// # assert_eq!(slice.as_slice(), vec![0; 100].as_slice());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_value(value: T, len: usize, options: TempMmapOptions) -> Result<Self> {
        match options {
            TempMmapOptions::None => Ok(Self {
                mmap: None,
                in_memory_vec: vec![value; len],
            }),
            TempMmapOptions::TempDir(flags) => {
                let mut mmap_slice = Self::from_tempfile_and_len(
                    len,
                    tempfile().with_context(|| "Cannot create tempfile in temporary directory")?,
                    flags,
                )
                .with_context(|| {
                    format!("Cannot create mmap of len {} in temporary directory", len)
                })?;
                mmap_slice.fill(value);
                Ok(mmap_slice)
            }
            TempMmapOptions::CustomDir(dir, flags) => {
                let mut mmap_slice = Self::from_tempfile_and_len(
                    len,
                    tempfile_in(dir.as_path()).with_context(|| {
                        format!("Cannot create tempfile in directory {}", dir.display())
                    })?,
                    flags,
                )
                .with_context(|| {
                    format!(
                        "Cannot create mmap of len {} in directory {}",
                        len,
                        dir.display()
                    )
                })?;
                mmap_slice.fill(value);
                Ok(mmap_slice)
            }
        }
    }
}

impl<T> MmapSlice<T> {
    /// The number of bytes required to store a single element of the slice.
    const BLOCK_SIZE: usize = size_of::<T>();

    fn mmap(file: File, flags: MmapFlags) -> Result<Self> {
        let file_len: usize = file
            .metadata()
            .with_context(|| "Cannot retrieve file metadata")?
            .len()
            .try_into()
            .with_context(|| "Cannot convert to usize")?;
        let mmap_len =
            file_len + (Self::BLOCK_SIZE - (file_len % Self::BLOCK_SIZE)) % Self::BLOCK_SIZE;

        if mmap_len == 0 {
            return Ok(MmapSlice {
                mmap: None,
                in_memory_vec: Vec::new(),
            });
        }

        ensure!(
            mmap_len == file_len,
            "file len is not of the correct length for element of size {}",
            Self::BLOCK_SIZE
        );

        let mmap = unsafe {
            MmapOptions::new(mmap_len)
                .with_context(|| format!("Cannot initialize mmap of size {}", mmap_len))?
                .with_file(&file, 0)
                .with_flags(flags)
                .map_mut()
                .with_context(|| "Cannot mutably mmap")?
        };

        assert!(
            (mmap.as_ptr() as *const T).is_aligned(),
            "mmap pointer is not aligned for {} ({} bytes)",
            std::any::type_name::<T>(),
            std::mem::size_of::<T>()
        );

        Ok(Self {
            mmap: Some((file, mmap, mmap_len / Self::BLOCK_SIZE)),
            in_memory_vec: Vec::new(),
        })
    }

    /// Creates a new slice from a [`Vec`] with the provided [`TempMmapOptions`].
    ///
    /// # Examples
    ///
    /// ```
    /// use webgraph_algo::utils::*;
    ///
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let v = vec![0; 100];
    /// let slice = MmapSlice::from_vec(v, TempMmapOptions::None)?;
    /// # assert_eq!(slice.as_slice(), vec![0; 100].as_slice());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Note how this consumes the [`Vec`].
    /// As such the following is illegal and should not compile.
    ///
    /// ```compile_fail
    /// use webgraph_algo::utils::*;
    ///
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let v = vec![0; 100];
    /// let slice = MmapSlice::from_vec(v, TempMmapOptions::None)?;
    /// assert_eq!(slice.as_slice(), v.as_slice());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_vec(v: Vec<T>, options: TempMmapOptions) -> Result<Self> {
        match options {
            TempMmapOptions::None => Ok(Self {
                mmap: None,
                in_memory_vec: v,
            }),
            TempMmapOptions::TempDir(flags) => Ok(Self::from_tempfile_and_vec(
                v,
                tempfile().with_context(|| "Cannot create tempfile in temporary directory")?,
                flags,
            )
            .with_context(|| "Cannot create mmap in temporary directory")?),
            TempMmapOptions::CustomDir(dir, flags) => Ok(Self::from_tempfile_and_vec(
                v,
                tempfile_in(dir.as_path()).with_context(|| {
                    format!("Cannot create tempfile in directory {}", dir.display())
                })?,
                flags,
            )
            .with_context(|| format!("Cannot create mmap in directory {}", dir.display()))?),
        }
    }

    /// Creates a new slice of length `len` with the provided [`TempMmapOptions`] and with all
    /// the elements initialized to the value returned by `closure`.
    ///
    /// # Examples
    ///
    /// ```
    /// use webgraph_algo::utils::*;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let slice = MmapSlice::from_closure(|| AtomicUsize::new(0), 100, TempMmapOptions::None)?;
    /// # slice.as_slice().iter().for_each(|n| assert_eq!(n.load(Ordering::Relaxed), 0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_closure(
        closure: impl FnMut() -> T,
        len: usize,
        options: TempMmapOptions,
    ) -> Result<Self> {
        match options {
            TempMmapOptions::None => Ok(Self {
                mmap: None,
                in_memory_vec: closure_vec(closure, len),
            }),
            TempMmapOptions::TempDir(flags) => {
                let mut mmap_slice = Self::from_tempfile_and_len(
                    len,
                    tempfile().with_context(|| "Cannot create tempfile in temporary directory")?,
                    flags,
                )
                .with_context(|| {
                    format!("Cannot create mmap of len {} in temporary directory", len)
                })?;
                mmap_slice.fill_with(closure);
                Ok(mmap_slice)
            }
            TempMmapOptions::CustomDir(dir, flags) => {
                let mut mmap_slice = Self::from_tempfile_and_len(
                    len,
                    tempfile_in(dir.as_path()).with_context(|| {
                        format!("Cannot create tempfile in directory {}", dir.display())
                    })?,
                    flags,
                )
                .with_context(|| {
                    format!(
                        "Cannot create mmap of len {} in directory {}",
                        len,
                        dir.display()
                    )
                })?;
                mmap_slice.fill_with(closure);
                Ok(mmap_slice)
            }
        }
    }

    fn from_tempfile_and_len(len: usize, file: File, flags: MmapFlags) -> Result<Self> {
        let expected_len = len * Self::BLOCK_SIZE;
        file.set_len(
            expected_len
                .try_into()
                .with_context(|| "Cannot convert file len")?,
        )
        .with_context(|| format!("Cannot set file len to {} bytes", expected_len))?;

        let mmap = Self::mmap(file, flags).with_context(|| "Cannot create mmap from tempfile")?;

        Ok(mmap)
    }

    fn from_tempfile_and_vec(v: Vec<T>, file: File, flags: MmapFlags) -> Result<Self> {
        let expected_len = v.len() * Self::BLOCK_SIZE;
        file.set_len(
            expected_len
                .try_into()
                .with_context(|| "Cannot convert file len")?,
        )
        .with_context(|| format!("Cannot set file len to {} bytes", expected_len))?;

        let mut mmap =
            Self::mmap(file, flags).with_context(|| "Cannot create mmap from tempfile")?;

        let v = std::mem::ManuallyDrop::new(v);
        let src = v.as_ptr();
        let dst = mmap.as_mut_ptr();

        unsafe {
            // Safety: regions are non-overlapping and src is dropped by this method so
            // only dst can be read
            std::ptr::copy_nonoverlapping(src, dst, v.len());
        }

        Ok(mmap)
    }

    /// Extracts a slice containing the entire data.
    ///
    /// Equivalent to `&s[..]`
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }

    /// Extracts a mutable slice containing the entire data.
    ///
    /// Equivalent to `&mut s[..]`
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }
}

impl<T> AsRef<[T]> for MmapSlice<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        if let Some((_, mmap, len)) = self.mmap.as_ref() {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const T, *len) }
        } else {
            self.in_memory_vec.as_slice()
        }
    }
}

impl<T> AsMut<[T]> for MmapSlice<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        if let Some((_, mmap, len)) = self.mmap.as_mut() {
            unsafe { std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut T, *len) }
        } else {
            self.in_memory_vec.as_mut_slice()
        }
    }
}

impl<T> Deref for MmapSlice<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> DerefMut for MmapSlice<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}
