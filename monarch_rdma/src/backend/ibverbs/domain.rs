/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ibverbs RDMA domain.
//!
//! An [`IbvDomain`] manages the device context and protection domain (PD)
//! required for RDMA operations. It provides the foundation for creating
//! queue pairs and establishing connections between RDMA devices.

use std::ffi::c_void;
use std::io::Error;
use std::os::fd::AsRawFd;
use std::os::fd::FromRawFd;
use std::os::fd::OwnedFd;
use std::result::Result;
use std::sync::Arc;

use anyhow::Context;

use super::memory_region::IbvMemoryRegionKeepalive;
use super::memory_region::IbvMemoryRegionView;
use super::memory_region::MrSegment;
use super::memory_region::SegmentedMrs;
use super::primitives::IbvConfig;
use super::primitives::IbvContext;
use super::primitives::IbvCq;
use super::primitives::IbvDeviceInfo;
use super::primitives::IbvMr;
use super::primitives::IbvPd;
use super::queue_pair::IbvQueuePair;
use crate::device_selection::MemoryLocation;
use crate::local_memory::KeepaliveLocalMemory;

/// Manages RDMA resources including context and protection domain.
///
/// # Fields
///
/// * `device_info`: Metadata for the device this PD is allocated on.
/// * `domain_impl`: The backend [`IbvDomainImpl`] strategy for this PD. It may
///   own FFI resources allocated against the PD, so it is declared before `pd`
///   and thus dropped first — releasing those resources while the PD is still
///   alive.
/// * `pd`: The protection domain (and, through [`IbvPd`], the device context).
///   Held in an `Arc` because the resources built against it — memory regions
///   and queue pairs — each keep their own clone (via [`Self::pd`]) to hold the
///   PD open for as long as they need it, independent of this domain.
///
/// `I` is the backend [`IbvDomainImpl`] strategy parameterizing per-PD
/// behavior.
///
/// `IbvDomain` is not `Clone`: it owns the backend strategy's FFI resources.
/// Hand out [`Self::pd`] clones to share the protection domain.
pub struct IbvDomain<I: IbvDomainImpl> {
    pub device_info: IbvDeviceInfo,
    domain_impl: I,
    pd: Arc<IbvPd>,
}

impl<I: IbvDomainImpl> std::fmt::Debug for IbvDomain<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IbvDomain")
            .field("context", &format!("{:p}", self.pd.context().as_ptr()))
            .field("pd", &format!("{:p}", self.pd.as_ptr()))
            .field("device_info", &self.device_info)
            .field("domain_impl", &self.domain_impl)
            .finish()
    }
}

impl<I: IbvDomainImpl> IbvDomain<I> {
    /// Creates an `IbvDomain` over an already-opened device `context`.
    ///
    /// Builds the backend [`IbvDomainImpl`] strategy `I` from `config`, then
    /// allocates the protection domain against `context`. The PD is allocated
    /// only *after* the strategy is built, so a panicking
    /// [`IbvDomainImpl::new`] never leaks a PD.
    ///
    /// Note:
    /// Host memory registration may use implicit ODP when configured, which
    /// exposes a large virtual address range to the NIC instead of pinning only
    /// the allocation.
    ///
    /// We currently assume a trusted, single-environment and are not enforcing finer-grained memory isolation
    /// at this layer. We plan to investigate mitigations - such as memory windows or tighter registration
    /// boundaries in future follow-ups.
    ///
    /// # Safety
    ///
    /// `context` must wrap a valid, live `ibv_context`: the `Arc<IbvContext>`
    /// keeps it open for this call and for the returned domain's lifetime (the
    /// PD allocated here is freed against that context on `Drop`).
    ///
    /// # Panics
    ///
    /// Panics if `context` is null.
    ///
    /// # Errors
    ///
    /// Returns an error if protection-domain allocation fails.
    pub unsafe fn new(
        context: Arc<IbvContext>,
        device_info: IbvDeviceInfo,
        config: &IbvConfig,
    ) -> Result<Self, anyhow::Error> {
        assert!(
            !context.as_ptr().is_null(),
            "IbvDomain::new requires a non-null ibv_context"
        );
        // Build the strategy first; the PD below is allocated only if this
        // returns, so a panicking `IbvDomainImpl::new` never leaks a PD.
        // SAFETY: per this function's contract `context` wraps a valid, live
        // `ibv_context`, which is what `I::new` requires to query the device.
        let domain_impl = unsafe { I::new(&context, &device_info, config) };
        // SAFETY: per this function's contract `context` wraps a valid, live
        // `ibv_context`, which `IbvPd::create` allocates the PD against. The PD
        // takes ownership of `context`; the domain reaches it via `pd.context()`.
        let pd = Arc::new(unsafe { IbvPd::create(context) }?);
        Ok(Self {
            device_info,
            domain_impl,
            pd,
        })
    }

    /// Test-only constructor assembling a domain from parts, so unit tests can
    /// fabricate a domain (typically with a null `pd` whose `Drop` is a no-op).
    ///
    /// # Safety
    ///
    /// `pd` must satisfy [`IbvPd`]'s validity contract.
    #[cfg(test)]
    pub(super) unsafe fn for_test(
        pd: Arc<IbvPd>,
        device_info: IbvDeviceInfo,
        domain_impl: I,
    ) -> Self {
        Self {
            device_info,
            domain_impl,
            pd,
        }
    }

    /// The protection domain pointer. Valid for the lifetime of `&self`.
    /// Prefer this over touching the field directly (which is private).
    pub fn as_ptr(&self) -> *mut rdmaxcel_sys::ibv_pd {
        self.pd.as_ptr()
    }

    /// The protection domain, shareable as a keepalive. Holders that only need
    /// the PD (and, through it, the context) kept alive clone this rather than
    /// the whole domain.
    pub(super) fn pd(&self) -> &Arc<IbvPd> {
        &self.pd
    }

    /// The device context this domain's PD was allocated against, used by the
    /// data-path verbs.
    pub fn context(&self) -> &Arc<IbvContext> {
        self.pd.context()
    }

    /// The backend [`IbvDomainImpl`] strategy for this domain.
    pub(super) fn domain_impl(&self) -> &I {
        &self.domain_impl
    }

    /// Metadata for the device this domain's PD is allocated on.
    pub fn device_info(&self) -> &IbvDeviceInfo {
        &self.device_info
    }

    /// Access flags used when registering memory regions and creating queue pairs
    /// on this domain, from the backend [`IbvDomainImpl`] strategy.
    pub fn access_flags(&self) -> i32 {
        self.domain_impl().access_flags()
    }

    /// Register `mem` against this domain's PD, dispatching to the backend
    /// [`IbvDomainImpl`] strategy.
    pub fn register_mr(&self, mem: &KeepaliveLocalMemory) -> anyhow::Result<IbvMemoryRegionView> {
        // SAFETY: a fully-constructed `IbvDomain` holds a null-or-live PD per its
        // construction contract, and `KeepaliveLocalMemory` keeps `mem`'s backing
        // memory alive for the registration.
        unsafe { I::register_mr(self, mem) }
    }

    /// Create a queue pair against this domain, reporting its completions on
    /// `cq`, dispatching to the backend [`IbvDomainImpl`] strategy.
    pub fn create_queue_pair(
        &self,
        config: &IbvConfig,
        cq: Arc<IbvCq>,
    ) -> anyhow::Result<I::QueuePair> {
        // SAFETY: a fully-constructed `IbvDomain` holds a null-or-live PD, and
        // through it a null-or-live context, per its construction contract --
        // which is what `I::create_queue_pair` requires.
        unsafe { I::create_queue_pair(self, config, cq) }
    }
}

/// Per-backend strategy for a protection domain: how memory regions are
/// registered and how queue pairs are built against the PD.
///
/// One strategy is constructed per domain via [`Self::new`], which
/// inspects the device behind the context to decide backend-specific behavior
/// up front, and is then stored in the [`IbvDomain`] it drives. The per-op
/// methods are associated functions taking `&IbvDomain<Self>` and reach the
/// strategy itself through [`IbvDomain::domain_impl`].
pub trait IbvDomainImpl: std::fmt::Debug + Send + Sync + 'static + Sized {
    /// The concrete queue-pair type built against this domain's PD.
    type QueuePair: IbvQueuePair;

    /// Build the strategy for the device behind `context` (whose queried
    /// metadata is `device_info`), using `config` for any setup it performs.
    ///
    /// # Safety
    ///
    /// If `context` is non-null it must wrap a valid, live `ibv_context`;
    /// implementations may query the device behind it.
    unsafe fn new(context: &IbvContext, device_info: &IbvDeviceInfo, config: &IbvConfig) -> Self;

    /// Access flags used when registering memory regions and creating queue
    /// pairs on this domain.
    fn access_flags(&self) -> i32;

    /// Register `mem` against `domain`'s PD and return a view of the
    /// resulting memory region.
    ///
    /// The default implementation covers host memory (`ibv_reg_mr`) and
    /// the device-memory dmabuf path (`ibv_reg_dmabuf_mr`); backends
    /// override to add hardware-specific registration, falling back to
    /// this default for the cases they do not special-case.
    ///
    /// # Safety
    ///
    /// `domain`'s PD (`domain.as_ptr()`) must be null or a live protection
    /// domain; `mem`'s backing memory must stay valid for the returned MR's
    /// lifetime.
    unsafe fn register_mr(
        domain: &IbvDomain<Self>,
        mem: &KeepaliveLocalMemory,
    ) -> anyhow::Result<IbvMemoryRegionView> {
        // SAFETY: `domain.as_ptr()` is null or a live PD (per this method's
        // contract; `register_host_or_dmabuf_mr` errors on null), and the caller
        // keeps `mem`'s backing memory valid for the MR's lifetime.
        unsafe { register_host_or_dmabuf_mr(domain, mem) }
    }

    /// Create a queue pair against `domain`, reporting every completion on
    /// `cq`. The default builds [`Self::QueuePair`] directly; backends override
    /// to construct their own queue-pair type.
    ///
    /// One queue serves both the send and receive sides, because nothing posts
    /// receive work requests on these queue pairs -- a separate receive queue
    /// would only ever stay empty.
    ///
    /// # Safety
    ///
    /// `domain` must hold a null or live PD (`domain.as_ptr()`) and, with it, a
    /// null or live device context, and `cq` must hold a live `ibv_cq` created
    /// on that context: the queue pair is built against that PD and reports its
    /// completions on that queue. A null PD yields `Err` rather than reaching
    /// the driver.
    unsafe fn create_queue_pair(
        domain: &IbvDomain<Self>,
        config: &IbvConfig,
        cq: Arc<IbvCq>,
    ) -> anyhow::Result<Self::QueuePair> {
        // Reject a null PD (e.g. a test domain) up front, so a domain that
        // cannot back a queue pair says so rather than reaching the driver.
        if domain.as_ptr().is_null() {
            anyhow::bail!("cannot create a queue pair on a null protection domain");
        }

        // SAFETY: `domain` holds a live PD (null was rejected above) and `cq` a
        // live queue on its context, per this method's contract, which is what
        // `IbvQueuePair::new` requires.
        unsafe { Self::QueuePair::new(domain, config.clone(), Arc::clone(&cq), cq) }
    }
}

/// Register host memory as a standard MR via `ibv_reg_mr`.
///
/// # Safety
///
/// If `pd` is non-null it must be a live protection domain whose context
/// outlives this call. For an ordinary registration, `[addr, addr + size)` must
/// name a host mapping that stays valid for the lifetime of the returned MR.
/// For implicit ODP, pass address zero, `usize::MAX`, and
/// `IBV_ACCESS_ON_DEMAND`.
pub(super) unsafe fn register_host_mr(
    pd: &Arc<IbvPd>,
    addr: usize,
    size: usize,
    access_flags: i32,
) -> anyhow::Result<IbvMr> {
    if pd.as_ptr().is_null() {
        anyhow::bail!("register_host_mr called with a null protection domain");
    }
    // SAFETY: `pd.as_ptr()` is non-null (checked above) and, per this function's
    // contract, a live protection domain; `[addr, addr + size)` is a
    // caller-guaranteed valid mapping. `ibv_reg_mr` returns null on failure,
    // which we check before wrapping the pointer.
    let mr =
        unsafe { rdmaxcel_sys::ibv_reg_mr(pd.as_ptr(), addr as *mut c_void, size, access_flags) };
    if mr.is_null() {
        anyhow::bail!("failed to register host MR");
    }
    // SAFETY: `mr` is non-null (checked above) and freshly registered against
    // `pd`.
    Ok(unsafe { IbvMr::from_raw(mr, pd.clone()) })
}

/// Register exactly `[addr, addr + size)` of device memory as a dmabuf MR via
/// `ibv_reg_dmabuf_mr`, mapped at iova 0.
///
/// `cuMemGetHandleForAddressRange` requires both `addr` and `size` to be
/// host-page aligned, so this errors if either is not.
///
/// # Safety
///
/// If `pd` is non-null it must be a live protection domain whose context
/// outlives this call. `[addr, addr + size)` must name device memory that
/// stays valid for the lifetime of the returned MR.
pub(super) unsafe fn register_dmabuf_range(
    pd: &Arc<IbvPd>,
    addr: usize,
    size: usize,
    access_flags: i32,
) -> anyhow::Result<IbvMr> {
    if pd.as_ptr().is_null() {
        anyhow::bail!("register_dmabuf_range called with a null protection domain");
    }
    // SAFETY: `sysconf` reads a process-global parameter and takes no pointers.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    anyhow::ensure!(
        page_size > 0,
        "sysconf(_SC_PAGESIZE) failed: {}",
        Error::last_os_error()
    );
    let host_page_size = page_size as usize;
    anyhow::ensure!(
        addr.is_multiple_of(host_page_size) && size.is_multiple_of(host_page_size),
        "dmabuf range is not host-page aligned (addr: 0x{addr:x}, size: {size}, page size: {host_page_size})"
    );

    let mut fd: i32 = -1;
    // SAFETY: `rdmaxcel_cuMemGetHandleForAddressRange` writes the dmabuf fd for
    // `[addr, addr + size)` into `fd` and touches no Rust memory beyond the
    // `&mut fd` out-param; it reports failure via its return code, checked next.
    let cu_err = unsafe {
        rdmaxcel_sys::rdmaxcel_cuMemGetHandleForAddressRange(
            &mut fd,
            addr as rdmaxcel_sys::CUdeviceptr,
            size,
            rdmaxcel_sys::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0,
        )
    };
    if cu_err != rdmaxcel_sys::CUDA_SUCCESS || fd < 0 {
        anyhow::bail!(
            "failed to get dmabuf handle for CUDA memory (addr: 0x{:x}, size: {}, cu_err: {}, fd: {})",
            addr,
            size,
            cu_err,
            fd
        );
    }
    // SAFETY: `fd >= 0` (checked above) is a fresh dmabuf descriptor we
    // exclusively own; wrapping it in `OwnedFd` closes it on drop and keeps it
    // open across the registration below.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: `pd.as_ptr()` is a non-null protection domain (checked above)
    // belonging to a live context; `fd` is a valid dmabuf descriptor kept open
    // by the `OwnedFd` across this call; `size` matches the range queried above.
    // `ibv_reg_dmabuf_mr` returns null on failure, which we check.
    let mr = unsafe {
        rdmaxcel_sys::ibv_reg_dmabuf_mr(pd.as_ptr(), 0, size, 0, fd.as_raw_fd(), access_flags)
    };
    if mr.is_null() {
        anyhow::bail!("failed to register dmabuf MR");
    }
    // SAFETY: `mr` is non-null (checked above) and freshly registered against
    // `pd`.
    Ok(unsafe { IbvMr::from_raw(mr, pd.clone()) })
}

/// One physical VMM mapping registered as its own dmabuf MR. `offset` is the
/// mapping's start relative to the base of the range passed to
/// [`register_dmabuf_mr_per_mapping`]; `size` is its byte length.
pub(super) struct MappingMr {
    pub(super) mr: IbvMr,
    pub(super) offset: usize,
    pub(super) size: usize,
}

/// Register `[base, base + total)` as one dmabuf MR **per physical VMM mapping**.
///
/// `hipMemGetHandleForAddressRange` exports a dmabuf covering only the first
/// physical mapping of a multi-mapping range (by design: the kernel forbids one
/// dmabuf spanning multiple backing buffers). This walks the range one mapping at
/// a time: export at the current boundary, measure the (truncated) dmabuf with
/// `lseek` to learn the mapping's size, register exactly that mapping, advance.
/// Single-mapping memory (plain `hipMalloc`) yields one MR.
///
/// `base` must be host-page aligned (callers widen to the allocation base, which
/// is granularity-aligned); mapping sizes are granularity-aligned, so every
/// subsequent boundary stays aligned.
///
/// # Safety
///
/// `pd` must be null or a live PD whose context outlives the returned MRs;
/// `[base, base + total)` must name live CUDA device memory valid for their
/// lifetime.
pub(super) unsafe fn register_dmabuf_mr_per_mapping(
    pd: &Arc<IbvPd>,
    base: usize,
    total: usize,
    access_flags: i32,
) -> anyhow::Result<Vec<MappingMr>> {
    if pd.as_ptr().is_null() {
        anyhow::bail!("register_dmabuf_mr_per_mapping called with a null protection domain");
    }
    anyhow::ensure!(
        total > 0,
        "register_dmabuf_mr_per_mapping called with a zero-size range (base: 0x{base:x})"
    );
    let end = base
        .checked_add(total)
        .with_context(|| format!("range [0x{base:x}, +{total}) overflows usize"))?;

    let mut mrs: Vec<MappingMr> = Vec::new();
    let mut cur = base;
    while cur < end {
        let remaining = end - cur;

        // Export a dmabuf at `cur`. On multi-mapping memory HIP truncates this to
        // the single physical mapping that starts at `cur`.
        let mut fd: i32 = -1;
        // SAFETY: writes only `&mut fd`; `cur` is passed by value as an opaque
        // device address; failure is reported via the return code, checked next.
        let cu_err = unsafe {
            rdmaxcel_sys::rdmaxcel_cuMemGetHandleForAddressRange(
                &mut fd,
                cur as rdmaxcel_sys::CUdeviceptr,
                remaining,
                rdmaxcel_sys::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                0,
            )
        };
        if cu_err != rdmaxcel_sys::CUDA_SUCCESS || fd < 0 {
            anyhow::bail!(
                "failed to export dmabuf for mapping at 0x{cur:x} (remaining: {remaining}, cu_err: {cu_err}, fd: {fd})"
            );
        }
        // SAFETY: `fd >= 0` is a dmabuf descriptor we own; `OwnedFd` closes it on
        // drop and keeps it open across the registration below.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };

        // Measure what the kernel actually exported: this one mapping's size.
        // SAFETY: `fd` is a live descriptor; `lseek` takes no pointers.
        let mapped = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
        if mapped <= 0 {
            anyhow::bail!(
                "could not measure dmabuf size for mapping at 0x{cur:x} (lseek returned {mapped})"
            );
        }
        // A mapping never extends past the range we asked for.
        let mapped = (mapped as usize).min(remaining);

        // Register exactly this mapping.
        // SAFETY: `pd` is a live PD (checked above); `fd` is a live dmabuf kept
        // open across the call; `mapped` matches the exported dmabuf.
        let mr = unsafe {
            rdmaxcel_sys::ibv_reg_dmabuf_mr(pd.as_ptr(), 0, mapped, 0, fd.as_raw_fd(), access_flags)
        };
        if mr.is_null() {
            anyhow::bail!("failed to register dmabuf MR for mapping at 0x{cur:x} (size: {mapped})");
        }
        // SAFETY: `mr` is non-null and freshly registered against `pd`.
        mrs.push(MappingMr {
            mr: unsafe { IbvMr::from_raw(mr, pd.clone()) },
            offset: cur - base,
            size: mapped,
        });
        cur += mapped;
    }
    Ok(mrs)
}

/// Base address and size of the CUDA allocation containing `addr`, as reported
/// by `cuMemGetAddressRange`.
///
/// `cuMemGetAddressRange` resolves `addr` in the current CUDA context, so this
/// returns an error unless `addr` is a device pointer whose owning context is
/// current (see [`crate::local_memory::set_ctx_for_ptr`]).
fn cuda_alloc_range(addr: usize) -> anyhow::Result<(usize, usize)> {
    let mut base: rdmaxcel_sys::CUdeviceptr = unsafe { std::mem::zeroed() };
    let mut size: usize = 0;
    // SAFETY: the out-params point to local variables and `addr` is passed by
    // value as an opaque device address (never dereferenced); the call writes
    // only through the out-params and reports a bad address or context via its
    // return code, checked next.
    let cu_err = unsafe {
        rdmaxcel_sys::rdmaxcel_cuMemGetAddressRange(
            &mut base,
            &mut size,
            addr as rdmaxcel_sys::CUdeviceptr,
        )
    };
    if cu_err != rdmaxcel_sys::CUDA_SUCCESS {
        anyhow::bail!(
            "failed to get address range for CUDA memory (addr: 0x{:x}, cu_err: {})",
            addr,
            cu_err
        );
    }
    Ok((base as usize, size))
}

/// The widened span `[base, end)` a dmabuf registration of `[addr, addr + size)`
/// must cover: from the base of the allocation holding `addr` to the end of the
/// allocation holding the range's last byte. A range may run past the allocation
/// its first byte falls in, so `cuMemGetAddressRange` is queried at both ends.
/// Allocation bounds are granularity- (hence host-page-) aligned.
///
/// Must be called with `addr`'s CUDA context current (it queries
/// `cuMemGetAddressRange`; see [`crate::local_memory::set_ctx_for_ptr`]).
fn cuda_alloc_span(addr: usize, size: usize) -> anyhow::Result<(usize, usize)> {
    anyhow::ensure!(
        size > 0,
        "cuda_alloc_span called with a zero-size range (addr: 0x{addr:x})"
    );
    let (base, base_size) = cuda_alloc_range(addr)?;
    let last = addr.checked_add(size - 1).with_context(|| {
        format!("final byte of range [{addr}, {addr} + {size}) would overflow max usize")
    })?;
    // When the range lies within the first allocation, that allocation is the
    // end; otherwise a second lookup resolves the allocation the last byte is in.
    let end = if last - base < base_size {
        base + base_size
    } else {
        let (last_base, last_size) = cuda_alloc_range(last)?;
        last_base + last_size
    };
    Ok((base, end))
}

/// Register the CUDA device memory backing `[addr, addr + size)` as a single
/// dmabuf MR, mapped at iova 0. The returned `usize` is the offset of `addr`
/// within that MR.
///
/// The MR covers, in full, every allocation the requested range touches: it
/// spans from the base of the allocation containing `addr` to the end of the
/// allocation containing the range's last byte. A range may extend past the
/// allocation holding its first byte, so `cuMemGetAddressRange` — which reports
/// the bounds of the allocation containing a given pointer — is queried at both
/// ends, and the MR covers everything between them.
///
/// Allocation bounds are granularity-aligned, hence host-page aligned, so the
/// base and size handed to [`register_dmabuf_range`] pass its alignment check.
///
/// # Safety
///
/// If `pd` is non-null it must be a live protection domain whose context
/// outlives this call. `addr` must belong to CUDA device memory, and
/// `[addr, addr + size)` must stay valid for the lifetime of the returned MR.
pub(super) unsafe fn register_dmabuf_mr(
    pd: &Arc<IbvPd>,
    addr: usize,
    size: usize,
    access_flags: i32,
) -> anyhow::Result<(IbvMr, usize)> {
    anyhow::ensure!(
        size > 0,
        "register_dmabuf_mr called with a zero-size range (addr: 0x{addr:x})"
    );

    // `cuMemGetAddressRange` resolves the pointer in the *current* CUDA context,
    // so make the pointer's own device context current first. Without this, in a
    // multi-GPU process it fails with `CUDA_ERROR_NOT_FOUND` whenever the active
    // context belongs to a different device than `addr`.
    // SAFETY: this path is only taken for device memory (the
    // `MemoryLocation::Gpu` arm in `register_host_or_dmabuf_mr`), so `addr` is a
    // valid CUDA device pointer as `set_ctx_for_ptr` requires. The guard
    // restores the prior context on drop.
    let _ctx_guard = unsafe { crate::local_memory::set_ctx_for_ptr(addr)? };

    let (base, end) = cuda_alloc_span(addr, size)?;

    // SAFETY: forwards this function's contract; `register_dmabuf_range` checks
    // that `base` and `end - base` are host-page aligned.
    let mr = unsafe { register_dmabuf_range(pd, base, end - base, access_flags)? };
    Ok((mr, addr - base))
}

/// Register the CUDA device memory backing `[addr, addr + size)` as one dmabuf
/// MR **per physical mapping**, returning a view of the requested sub-range.
///
/// This is the fallback [`register_host_or_dmabuf_mr`] takes when the single-MR
/// [`register_dmabuf_mr`] cannot register multi-mapping memory (ROCm
/// `expandable_segments` / vLLM cumem), whose whole-range dmabuf export
/// truncates to the first mapping. It widens to the enclosing allocation (one
/// dmabuf export must start at a mapping boundary, which the sub-range's start
/// need not be), registers each physical mapping, and narrows the result back to
/// `[addr, addr + size)`.
///
/// Returns `Ok(None)` when the allocation is single-mapping (one MR): there is
/// then no multi-mapping to stitch, and the caller should surface the original
/// single-MR error. Otherwise `Ok(Some(view))` — segmented across the mappings
/// the sub-range spans, or collapsed to the flat form when it lies within one.
///
/// # Safety
///
/// As [`register_dmabuf_mr`]: `pd` null or live, `[addr, addr + size)` live CUDA
/// device memory valid for the returned view's MRs' lifetime.
unsafe fn register_dmabuf_segmented_mr(
    pd: &Arc<IbvPd>,
    addr: usize,
    size: usize,
    access_flags: i32,
    device_name: &str,
) -> anyhow::Result<Option<IbvMemoryRegionView>> {
    anyhow::ensure!(
        size > 0,
        "register_dmabuf_segmented_mr called with a zero-size range (addr: 0x{addr:x})"
    );
    // SAFETY: device memory only (the `Gpu` arm of `register_host_or_dmabuf_mr`),
    // so `addr` is a valid CUDA device pointer; the guard restores the prior
    // context on drop.
    let _ctx_guard = unsafe { crate::local_memory::set_ctx_for_ptr(addr)? };
    let (base, end) = cuda_alloc_span(addr, size)?;

    // SAFETY: forwards this function's contract; the per-mapping registrar checks
    // PD validity and that `base` is host-page aligned (allocation bases are).
    let mappings =
        unsafe { register_dmabuf_mr_per_mapping(pd, base, end - base, access_flags)? };
    // One mapping is not the multi-mapping case this path exists for; signal the
    // caller to keep its original single-MR error.
    if mappings.len() <= 1 {
        return Ok(None);
    }

    let mut segments = Vec::with_capacity(mappings.len());
    let mut mrs = Vec::with_capacity(mappings.len());
    for mapping in mappings {
        // SAFETY: `mapping.mr` wraps a non-null, freshly-registered MR at iova 0.
        let (rdma_addr, lkey, rkey) = unsafe {
            let p = mapping.mr.as_ptr();
            ((*p).addr as usize, (*p).lkey, (*p).rkey)
        };
        segments.push(MrSegment {
            offset: mapping.offset,
            size: mapping.size,
            rdma_addr,
            lkey,
            rkey,
        });
        mrs.push(mapping.mr);
    }
    // One guard owns every per-mapping MR (each deregistered on its `Drop`) and
    // anchors their PD past deregistration.
    let guard: Arc<dyn IbvMemoryRegionKeepalive> = Arc::new(SegmentedMrs(mrs));
    // The view over the whole allocation, narrowed to the requested sub-range:
    // `try_slice` keeps only the mappings `[addr, addr + size)` spans and
    // collapses to the flat form when that is a single mapping.
    let whole = IbvMemoryRegionView::new_segmented(
        base,
        end - base,
        device_name.to_string(),
        segments,
        guard,
    );
    Ok(Some(whole.try_slice(addr - base, size)?))
}

/// Build a flat (single-MR) view of `[addr, addr + size)` from a freshly
/// registered `mr` whose iova-0 address space places the requested range at
/// `mr_offset`.
fn flat_region_view(
    mr: IbvMr,
    mr_offset: usize,
    addr: usize,
    size: usize,
    device_name: String,
) -> IbvMemoryRegionView {
    // SAFETY: `mr` wraps a non-null, freshly-registered `ibv_mr`, so reading its
    // `addr`/`lkey`/`rkey` is sound.
    let (mr_addr, lkey, rkey) = unsafe {
        let p = mr.as_ptr();
        ((*p).addr as usize, (*p).lkey, (*p).rkey)
    };
    // The view addresses the requested sub-range, which sits at `mr_offset`
    // within the MR's zero-based address space.
    let rdma_addr = mr_addr + mr_offset;
    // The `IbvMr` guard owns the MR (deregistered on its `Drop`) and anchors the
    // PD past that deregistration; it coerces to `Arc<dyn IbvMemoryRegionKeepalive>`.
    let guard = Arc::new(mr);
    IbvMemoryRegionView::new(addr, rdma_addr, size, lkey, rkey, device_name, guard)
}

/// Default MR registration: host memory via [`register_host_mr`], device
/// memory via [`register_dmabuf_mr`], falling back to per-mapping registration
/// ([`register_dmabuf_segmented_mr`]) for multi-mapping device memory the
/// single-MR path cannot handle. Shared by the [`IbvDomainImpl`] default
/// `register_mr` and by backends as the fallback for memory they do not
/// special-case.
///
/// # Safety
///
/// `domain.as_ptr()` must be null or a live protection domain whose context
/// outlives this call (a null PD yields an error). `mem`'s
/// `[addr, addr + size)` must stay valid for the lifetime of the returned
/// view's MR — the MR keepalive maintains the `ibv_mr` but does not keep the
/// backing memory mapped.
pub(super) unsafe fn register_host_or_dmabuf_mr<I: IbvDomainImpl>(
    domain: &IbvDomain<I>,
    mem: &KeepaliveLocalMemory,
) -> anyhow::Result<IbvMemoryRegionView> {
    let addr = mem.addr();
    let size = mem.size();
    let access_flags = domain.access_flags();
    let device_name = domain.device_info().name().to_string();
    // SAFETY: per this function's contract `domain.pd()` is null or a live PD
    // (the helpers error on null), and `[addr, addr + size)` stays valid for the
    // returned view's MR lifetime.
    match mem.location() {
        // Host memory registers as itself: the MR is the requested range, so the
        // requested address sits at offset 0 within it.
        MemoryLocation::Cpu(_) => {
            let mr = unsafe { register_host_mr(domain.pd(), addr, size, access_flags)? };
            Ok(flat_region_view(mr, 0, addr, size, device_name))
        }
        // Device memory registers as a single dmabuf MR over the whole enclosing
        // allocation (the requested range may start partway in). That is the only
        // path on CUDA and on single-mapping ROCm memory.
        MemoryLocation::Gpu(_) => {
            match unsafe { register_dmabuf_mr(domain.pd(), addr, size, access_flags) } {
                Ok((mr, mr_offset)) => {
                    Ok(flat_region_view(mr, mr_offset, addr, size, device_name))
                }
                // The whole-range export truncates on multi-mapping ROCm memory
                // (`expandable_segments` / vLLM cumem), so that registration
                // fails. Register each physical mapping instead and stitch them in
                // software. Only genuinely multi-mapping memory takes this path;
                // for anything else the original error is the real one.
                Err(flat_err) => {
                    match unsafe {
                        register_dmabuf_segmented_mr(
                            domain.pd(),
                            addr,
                            size,
                            access_flags,
                            &device_name,
                        )
                    } {
                        Ok(Some(view)) => Ok(view),
                        Ok(None) | Err(_) => Err(flat_err),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda_test_utils::CudaAllocator;
    use crate::backend::cuda_test_utils::cuda_device_count;
    use crate::backend::ibverbs::device::IbvDevice;
    use crate::backend::ibverbs::device::IbvDeviceImpl;
    use crate::backend::ibverbs::device_selection::get_cuda_device_to_ibv_devices;
    use crate::backend::ibverbs::mlx_device::MlxDevice;
    use crate::backend::ibverbs::mlx_domain::MlxDomain;

    fn host_page_size() -> usize {
        // SAFETY: `sysconf` reads a process-global parameter and takes no pointers.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    /// Open an [`IbvDomain`] on the NIC mapped to CUDA `device`. These tests
    /// require a GPU with a mapped RDMA NIC, so a missing device, missing NIC, or
    /// open/creation failure panics. The returned domain owns its context, so it
    /// (and its PD) stays valid after the local [`IbvDevice`] drops.
    fn open_domain_for_cuda_device(device: i32) -> IbvDomain<MlxDomain> {
        // Ranking NICs against a CUDA ordinal asks the driver for the GPU's PCI
        // address, and `device_selection` deliberately never loads or initializes
        // the driver itself. Do it here: nothing else in this module touches CUDA
        // before this point, so the ranking below would otherwise fail with
        // `CUDA_ERROR_NOT_INITIALIZED`.
        assert!(
            cuda_device_count() > device,
            "CUDA device {device} is required by this test",
        );
        let ordinal_to_nics = get_cuda_device_to_ibv_devices::<MlxDevice>()
            .expect("the CUDA driver is initialized above");
        let nic = ordinal_to_nics
            .get(device as usize)
            .and_then(|nics| nics.first())
            .expect("CUDA device should map to RDMA NIC")
            .name()
            .clone();
        let mut config = IbvConfig::default();
        MlxDevice::apply_config_defaults(&mut config);
        let dev =
            IbvDevice::<MlxDevice>::try_open(&nic, config.clone()).expect("mapped NIC should open");
        // SAFETY: `dev.context()` wraps the live `ibv_context` opened above; the
        // returned `Arc<IbvContext>` keeps it open for the new domain's lifetime.
        unsafe { IbvDomain::new(dev.context(), dev.device_info().clone(), &config) }
            .expect("domain creation should succeed")
    }

    /// Open an [`IbvDomain`] on the first NIC that can register GPU memory on
    /// CUDA device 0 via dmabuf, ignoring NIC/GPU affinity ranking.
    ///
    /// Unlike [`open_domain_for_cuda_device`], this does NOT require an ACTIVE
    /// fabric port: `ibv_reg_dmabuf_mr` binds GPU memory to the HCA over PCIe
    /// peer-to-peer, which is independent of link state, so these registration
    /// tests run even on a host whose fabric is down (every port `PORT_DOWN`),
    /// where affinity selection -- which drops any NIC with no ACTIVE port --
    /// would exclude every NIC. The probe (a committed GPU allocation that this
    /// NIC must be able to register) also skips NICs with no p2p path to the GPU.
    /// Panics if no NIC can register GPU memory, which means this host cannot run
    /// GPU-Direct dmabuf tests at all.
    ///
    /// TODO(ionic): this fixture is mlx5-only (`MlxDevice`/`MlxDomain`), so it
    /// does NOT exercise the ionic provider and is NOT valid for ionic
    /// validation. The primitives under test take a raw `pd` and are
    /// backend-agnostic; verifying per-mapping registration on a real ionic NIC
    /// means generifying this over the backend (or adding an ionic variant).
    /// mlx5-green proves the enumeration/data-model logic, not ionic provider
    /// acceptance or the ionic transfer path (owner-side WRITE / Option A).
    fn open_gpu_capable_domain() -> IbvDomain<MlxDomain> {
        assert!(
            cuda_device_count() > 0,
            "a CUDA device is required by this test",
        );
        let probe = CudaAllocator::get().allocate(0, 2 * 1024 * 1024, 2 * 1024 * 1024);
        let mut config = IbvConfig::default();
        MlxDevice::apply_config_defaults(&mut config);
        for nic in IbvDevice::<MlxDevice>::list() {
            let name = nic.name().clone();
            let Ok(dev) = IbvDevice::<MlxDevice>::try_open(&name, config.clone()) else {
                continue;
            };
            // SAFETY: `dev.context()` wraps the live `ibv_context` just opened;
            // the `Arc<IbvContext>` keeps it open for this domain's lifetime.
            let domain =
                match unsafe { IbvDomain::new(dev.context(), dev.device_info().clone(), &config) } {
                    Ok(domain) => domain,
                    Err(_) => continue,
                };
            // SAFETY: `domain.pd()` is a live PD; `probe` is a live committed GPU
            // allocation kept mapped for the duration of this call. The MR drops
            // at the end of this `if let`, before the next iteration.
            if let Ok((_mr, _offset)) = unsafe {
                register_dmabuf_mr(domain.pd(), probe.ptr(), probe.size(), domain.access_flags())
            } {
                return domain;
            }
        }
        panic!("no NIC can register GPU memory via dmabuf on this host");
    }

    /// A 4 MiB allocation, fully committed (reserved == committed) so it is
    /// entirely mapped and `cuMemGetAddressRange` reports the whole extent.
    fn committed_allocation() -> crate::backend::cuda_test_utils::CudaAllocation {
        CudaAllocator::get().allocate(0, 4 * 1024 * 1024, 4 * 1024 * 1024)
    }

    /// `addr` (iova) and `length` of an `ibv_mr`.
    ///
    /// # Safety
    ///
    /// `mr` must be a live MR (e.g. owned by a not-yet-dropped [`IbvMr`]).
    unsafe fn mr_extent(mr: *mut rdmaxcel_sys::ibv_mr) -> (usize, usize) {
        // SAFETY: per this function's contract `mr` is a live MR.
        unsafe { ((*mr).addr as usize, (*mr).length) }
    }

    // `register_dmabuf_mr` registers the whole enclosing allocation and reports
    // the requested address's offset within it — even when that address is not
    // host-page aligned, which is the case the bug fix targets.
    #[test]
    fn register_dmabuf_mr_covers_whole_allocation() {
        let domain = open_domain_for_cuda_device(0);
        let pd = domain.pd();
        let access = domain.access_flags();
        let alloc = committed_allocation();
        let alloc_size = alloc.size();

        // Requesting the whole allocation: offset 0, MR spans it at iova 0. `mr`
        // deregisters on drop, after the assertions below.
        // SAFETY: `pd` is a live PD; `alloc.ptr()` is a live CUDA allocation kept
        // mapped by `alloc` for the MR's lifetime.
        let (mr, offset) =
            unsafe { register_dmabuf_mr(pd, alloc.ptr(), alloc_size, access) }.unwrap();
        // SAFETY: `mr` owns a live MR and has not been dropped.
        let (iova, length) = unsafe { mr_extent(mr.as_ptr()) };
        assert_eq!(offset, 0, "base address sits at offset 0");
        assert_eq!(iova, 0, "dmabuf MR is mapped at iova 0");
        assert_eq!(length, alloc_size, "MR covers the whole allocation");

        // At an unaligned interior address, with a range that stays within the
        // allocation: the MR still spans the whole allocation and the offset
        // locates the requested address.
        let unaligned: usize = 257;
        assert!(!unaligned.is_multiple_of(host_page_size()));
        // SAFETY: as above; `alloc.ptr() + unaligned` is inside the allocation.
        let (mr, offset) =
            unsafe { register_dmabuf_mr(pd, alloc.ptr() + unaligned, host_page_size(), access) }
                .unwrap();
        // SAFETY: `mr` owns a live MR and has not been dropped.
        let (iova, length) = unsafe { mr_extent(mr.as_ptr()) };
        assert_eq!(offset, unaligned, "offset locates the requested address");
        assert_eq!(iova, 0, "dmabuf MR is mapped at iova 0");
        assert_eq!(length, alloc_size, "MR covers the whole allocation");
    }

    // A range crossing the mapping boundaries of an expandable segment yields a
    // single MR covering every mapping it spans, in full — and no further. This
    // is the case whole-allocation registration cannot handle: `addr` and the
    // range's last byte live in different mappings.
    #[test]
    fn register_dmabuf_mr_spans_multiple_mappings() {
        let domain = open_domain_for_cuda_device(0);
        let pd = domain.pd();
        let access = domain.access_flags();

        // Reserve generously, commit one chunk, then expand three times so the
        // segment is backed by four separate mappings at offsets 0, g, 2g, 3g.
        let alloc = CudaAllocator::get().allocate(0, 128 * 1024 * 1024, 1);
        let g = alloc.granularity();
        for _ in 0..3 {
            alloc
                .expand(g)
                .expect("expand stays within the reserved range");
        }
        assert_eq!(alloc.size(), 4 * g, "four mappings of one granularity each");

        // Start at an unaligned interior address in the first mapping and run
        // into the third: the range straddles the 0->1 and 1->2 boundaries but
        // stops short of the fourth mapping.
        let start_offset: usize = 257;
        assert!(!start_offset.is_multiple_of(host_page_size()));
        let size = 2 * g + 100;
        let last_offset = start_offset + size - 1;
        assert!(
            (2 * g..3 * g).contains(&last_offset),
            "range must end inside the third mapping"
        );

        // SAFETY: `pd` is a live PD; `alloc` keeps all four mappings resident for
        // the MR's lifetime (it outlives `mr`, which drops first).
        let (mr, offset) =
            unsafe { register_dmabuf_mr(pd, alloc.ptr() + start_offset, size, access) }.unwrap();
        // SAFETY: `mr` owns a live MR and has not been dropped.
        let (iova, length) = unsafe { mr_extent(mr.as_ptr()) };
        assert_eq!(offset, start_offset, "offset locates the requested address");
        assert_eq!(iova, 0, "dmabuf MR is mapped at iova 0");
        assert_eq!(
            length,
            3 * g,
            "MR covers the three spanned mappings in full, not the fourth"
        );
    }

    // `register_dmabuf_mr_per_mapping` registers a multi-mapping allocation as one
    // dmabuf MR per physical mapping, which is the ROCm-correct path:
    // `hipMemGetHandleForAddressRange` truncates a multi-mapping export to the
    // first mapping, so `register_dmabuf_mr` (above) cannot cover the whole range.
    #[test]
    fn register_dmabuf_mr_per_mapping_covers_every_mapping() {
        let domain = open_gpu_capable_domain();
        let pd = domain.pd();
        let access = domain.access_flags();

        // Four separate mappings of one granularity each.
        let alloc = CudaAllocator::get().allocate(0, 128 * 1024 * 1024, 1);
        let g = alloc.granularity();
        for _ in 0..3 {
            alloc
                .expand(g)
                .expect("expand stays within the reserved range");
        }
        assert_eq!(alloc.size(), 4 * g, "four mappings of one granularity each");

        // SAFETY: `pd` is a live PD; `alloc` keeps all four mappings resident for
        // the MRs' lifetime (it outlives `mrs`, which drop first).
        let mrs =
            unsafe { register_dmabuf_mr_per_mapping(pd, alloc.ptr(), 4 * g, access) }.unwrap();

        assert_eq!(mrs.len(), 4, "one MR per physical mapping");
        for (i, m) in mrs.iter().enumerate() {
            assert_eq!(m.offset, i * g, "mapping {i} starts at offset {i}*g");
            assert_eq!(m.size, g, "mapping {i} is one granularity");
            // SAFETY: `m.mr` owns a live MR and has not been dropped.
            let (iova, length) = unsafe { mr_extent(m.mr.as_ptr()) };
            assert_eq!(iova, 0, "each dmabuf MR is mapped at iova 0");
            assert_eq!(length, g, "each MR covers exactly its mapping");
        }
    }

    /// A reserved region grown to `n` separate mappings of one granularity each,
    /// returning the allocation and the granularity `g`.
    fn n_mapping_allocation(n: usize) -> (crate::backend::cuda_test_utils::CudaAllocation, usize) {
        let alloc = CudaAllocator::get().allocate(0, 128 * 1024 * 1024, 1);
        let g = alloc.granularity();
        for _ in 1..n {
            alloc
                .expand(g)
                .expect("expand stays within the reserved range");
        }
        assert_eq!(alloc.size(), n * g, "{n} mappings of one granularity each");
        (alloc, g)
    }

    // `register_dmabuf_segmented_mr` registers a request that spans several
    // physical mappings as one MR per mapping and returns a segmented view tiling
    // the request, each segment's MR at iova 0.
    #[test]
    fn segmented_mr_splits_a_request_spanning_several_mappings() {
        let domain = open_gpu_capable_domain();
        let device = domain.device_info().name().to_string();
        let (alloc, g) = n_mapping_allocation(4);

        // SAFETY: live PD; `alloc` keeps every mapping resident for the view's MRs
        // (it outlives `view`, which drops first).
        let view = unsafe {
            register_dmabuf_segmented_mr(
                domain.pd(),
                alloc.ptr(),
                4 * g,
                domain.access_flags(),
                &device,
            )
        }
        .unwrap()
        .expect("a request spanning four mappings yields a segmented view");

        assert!(view.is_segmented(), "four mappings -> segmented");
        assert_eq!(view.size, 4 * g);
        assert_eq!(view.virtual_addr, alloc.ptr());
        let segs = view.segments().expect("segmented");
        assert_eq!(segs.len(), 4, "one segment per mapping");
        for (i, s) in segs.iter().enumerate() {
            assert_eq!(s.offset, i * g, "segment {i} at offset {i}*g");
            assert_eq!(s.size, g, "segment {i} is one granularity");
            assert_eq!(s.rdma_addr, 0, "each per-mapping MR is at iova 0");
        }
    }

    // A request confined to a single mapping is not the multi-mapping case, so the
    // segmented path declines (`None`) and the caller keeps its single-MR result.
    #[test]
    fn segmented_mr_returns_none_for_a_single_mapping_request() {
        let domain = open_gpu_capable_domain();
        let device = domain.device_info().name().to_string();
        let (alloc, g) = n_mapping_allocation(4);

        // One granularity starting at the base: a single mapping.
        // SAFETY: as above.
        let result = unsafe {
            register_dmabuf_segmented_mr(domain.pd(), alloc.ptr(), g, domain.access_flags(), &device)
        }
        .unwrap();
        assert!(
            result.is_none(),
            "a single-mapping request declines the segmented path"
        );
    }

    // A request spanning an interior subset of the mappings widens only to the
    // mappings it touches: the view is segmented over exactly those, rebased so it
    // begins at the request's start.
    #[test]
    fn segmented_mr_covers_only_the_mappings_a_request_spans() {
        let domain = open_gpu_capable_domain();
        let device = domain.device_info().name().to_string();
        let (alloc, g) = n_mapping_allocation(4);

        // The middle two mappings: [ptr + g, ptr + 3g).
        // SAFETY: as above.
        let view = unsafe {
            register_dmabuf_segmented_mr(
                domain.pd(),
                alloc.ptr() + g,
                2 * g,
                domain.access_flags(),
                &device,
            )
        }
        .unwrap()
        .expect("a request spanning two mappings yields a segmented view");

        assert!(view.is_segmented(), "two mappings -> segmented");
        assert_eq!(view.virtual_addr, alloc.ptr() + g);
        assert_eq!(view.size, 2 * g);
        let segs = view.segments().expect("segmented");
        assert_eq!(segs.len(), 2, "only the two spanned mappings");
        assert_eq!((segs[0].offset, segs[0].size), (0, g));
        assert_eq!((segs[1].offset, segs[1].size), (g, g));
    }

    // `register_dmabuf_range` registers exactly the requested range — a strict
    // page-aligned sub-range of the allocation here, not the whole thing.
    #[test]
    fn register_dmabuf_range_registers_exact_range() {
        let domain = open_domain_for_cuda_device(0);
        let pd = domain.pd();
        let access = domain.access_flags();
        let alloc = committed_allocation();

        let offset = 2 * host_page_size();
        let size = 1024 * 1024;
        assert!(offset + size < alloc.size(), "sub-range must fit, strictly");

        // SAFETY: `pd` is a live PD; `[alloc.ptr() + offset, ... + size)` is a
        // host-page-aligned range within the fully mapped allocation.
        let mr = unsafe { register_dmabuf_range(pd, alloc.ptr() + offset, size, access) }.unwrap();
        // SAFETY: `mr` owns a live MR and has not been dropped.
        let (iova, length) = unsafe { mr_extent(mr.as_ptr()) };
        assert_eq!(iova, 0, "dmabuf MR is mapped at iova 0");
        assert_eq!(length, size, "MR covers exactly the requested sub-range");
    }

    // `register_dmabuf_range` rejects an unaligned address, gracefully, before
    // touching the driver.
    #[test]
    fn register_dmabuf_range_rejects_unaligned_addr() {
        let domain = open_domain_for_cuda_device(0);
        let pd = domain.pd();
        let access = domain.access_flags();
        let alloc = committed_allocation();

        // One byte past the (aligned) base is not host-page aligned.
        // SAFETY: `pd` is a live PD; the call errors on the alignment check
        // before touching the driver.
        let err = unsafe { register_dmabuf_range(pd, alloc.ptr() + 1, host_page_size(), access) }
            .unwrap_err();
        assert!(
            err.to_string().contains("host-page aligned"),
            "unexpected error: {err}"
        );
    }

    // `register_dmabuf_range` rejects an unaligned size, gracefully, before
    // touching the driver.
    #[test]
    fn register_dmabuf_range_rejects_unaligned_size() {
        let domain = open_domain_for_cuda_device(0);
        let pd = domain.pd();
        let access = domain.access_flags();
        let alloc = committed_allocation();

        // A size that is not a multiple of the host page size.
        // SAFETY: as above.
        let err = unsafe { register_dmabuf_range(pd, alloc.ptr(), host_page_size() + 1, access) }
            .unwrap_err();
        assert!(
            err.to_string().contains("host-page aligned"),
            "unexpected error: {err}"
        );
    }

    // A null PD is rejected before any driver call.
    #[test]
    fn register_dmabuf_range_rejects_null_pd() {
        let page = host_page_size();
        let pd = Arc::new(IbvPd::null());
        // SAFETY: a null PD is the documented error path; no memory is touched.
        let err = unsafe { register_dmabuf_range(&pd, page, page, 0) }.unwrap_err();
        assert!(
            err.to_string().contains("null protection domain"),
            "unexpected error: {err}"
        );
    }

    // The full `register_host_or_dmabuf_mr` path over device memory covering the
    // whole allocation: the view reports the allocation's address and size, and
    // an `rdma_addr` of 0 (the request sits at the MR's base).
    #[test]
    fn register_host_or_dmabuf_mr_covers_whole_allocation() {
        let domain = open_domain_for_cuda_device(0);
        let alloc = committed_allocation();
        let mem = alloc.keepalive_slice(0, alloc.size());

        // SAFETY: `domain`'s PD is live; `mem` keeps the allocation mapped for
        // the view's MR lifetime.
        let view = unsafe { register_host_or_dmabuf_mr(&domain, &mem) }.unwrap();
        // Tie the view's lifetime to the allocation's lifetime so that the safety contract
        // above holds.
        mem.install_mr::<MlxDevice>(view.clone())
            .expect("the region has no registration on this device yet");
        assert_eq!(view.virtual_addr, alloc.ptr());
        assert_eq!(view.size, alloc.size());
        assert_eq!(view.rdma_addr, 0, "whole allocation starts at MR offset 0");
    }

    // The full path over a page-aligned sub-range: the underlying MR still spans
    // the whole allocation, so `rdma_addr` is the sub-range's offset within it.
    #[test]
    fn register_host_or_dmabuf_mr_covers_subrange() {
        let domain = open_domain_for_cuda_device(0);
        let alloc = committed_allocation();
        let offset = 2 * host_page_size();
        let size = 1024 * 1024;
        assert!(
            size < alloc.size(),
            "sub-range ({}) must be strictly smaller than allocation ({})",
            size,
            alloc.size()
        );
        let mem = alloc.keepalive_slice(offset, size);

        // SAFETY: as above.
        let view = unsafe { register_host_or_dmabuf_mr(&domain, &mem) }.unwrap();
        // Tie the view's lifetime to the allocation's lifetime so that the safety contract
        // above holds.
        mem.install_mr::<MlxDevice>(view.clone())
            .expect("the region has no registration on this device yet");
        assert_eq!(view.virtual_addr, alloc.ptr() + offset);
        assert_eq!(view.size, size);
        assert_eq!(view.rdma_addr, offset, "rdma_addr is the sub-range offset");
    }

    // The full path tolerates an unaligned sub-range address: registering the
    // whole (aligned) allocation means the unaligned request still succeeds, and
    // `rdma_addr` locates it.
    #[test]
    fn register_host_or_dmabuf_mr_handles_unaligned_addr() {
        let domain = open_domain_for_cuda_device(0);
        let alloc = committed_allocation();
        let offset: usize = 257;
        assert!(!offset.is_multiple_of(host_page_size()));
        let size = 1024 * 1024;
        let mem = alloc.keepalive_slice(offset, size);

        // SAFETY: as above.
        let view = unsafe { register_host_or_dmabuf_mr(&domain, &mem) }.unwrap();
        // Tie the view's lifetime to the allocation's lifetime so that the safety contract
        // above holds.
        mem.install_mr::<MlxDevice>(view.clone())
            .expect("the region has no registration on this device yet");
        assert_eq!(view.virtual_addr, alloc.ptr() + offset);
        assert_eq!(view.size, size);
        assert_eq!(view.rdma_addr, offset);
    }

    // The full path tolerates an unaligned size for the same reason.
    #[test]
    fn register_host_or_dmabuf_mr_handles_unaligned_size() {
        let domain = open_domain_for_cuda_device(0);
        let alloc = committed_allocation();
        let size: usize = 1000;
        assert!(!size.is_multiple_of(host_page_size()));
        let mem = alloc.keepalive_slice(0, size);

        // SAFETY: as above.
        let view = unsafe { register_host_or_dmabuf_mr(&domain, &mem) }.unwrap();
        // Tie the view's lifetime to the allocation's lifetime so that the safety contract
        // above holds.
        mem.install_mr::<MlxDevice>(view.clone())
            .expect("the region has no registration on this device yet");
        assert_eq!(view.virtual_addr, alloc.ptr());
        assert_eq!(view.size, size);
        assert_eq!(view.rdma_addr, 0);
    }
}
