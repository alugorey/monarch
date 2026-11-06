/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Registered memory regions returned by [`IbvDomainImpl::register_mr`].
//!
//! [`IbvMemoryRegionView`] is the cheap, cloneable handle this process addresses
//! its own registered memory through: the keys and addresses for a slice of
//! registered memory, plus an `Arc<dyn IbvMemoryRegionKeepalive>` that keeps the
//! backing registration's resources alive until the last clone of the view
//! drops.
//!
//! [`IbvRemoteMemoryRegionView`] is what a peer gets instead: the same region
//! reduced to what the wire can carry and the far side can use.
//!
//! [`IbvDomainImpl::register_mr`]: super::domain::IbvDomainImpl::register_mr

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use super::primitives::IbvMr;

/// Guards the resources behind a registered MR, releasing them when the last
/// [`IbvMemoryRegionView`] over it drops. Each implementor frees whatever it
/// owns in its own `Drop`; the trait carries no methods and exists only to
/// type-erase the guards so a view can hold any of them behind an
/// `Arc<dyn IbvMemoryRegionKeepalive>`.
pub(super) trait IbvMemoryRegionKeepalive: std::fmt::Debug + Send + Sync {}

/// A standalone [`IbvMr`] guards its own registration: its `Drop` runs
/// `ibv_dereg_mr` against the PD it owns.
impl IbvMemoryRegionKeepalive for IbvMr {}

/// Guards every per-mapping MR of a segmented registration at once: dropping it
/// runs each [`IbvMr`]'s `ibv_dereg_mr`. Used as the keepalive behind a
/// multi-mapping [`IbvMemoryRegionView`], so the whole set stays registered
/// until the last view clone drops.
#[derive(Debug)]
// The MRs are held only so their `Drop` deregisters them; the field is never
// read, which dead-code analysis cannot see through `Drop`.
pub(super) struct SegmentedMrs(#[allow(dead_code)] pub(super) Vec<IbvMr>);
impl IbvMemoryRegionKeepalive for SegmentedMrs {}

/// One physical mapping within a segmented registration.
///
/// ROCm multi-mapping HIP-VMM memory (`expandable_segments` / vLLM cumem) cannot
/// be exported as a single dmabuf — the kernel truncates a multi-mapping export
/// to the first mapping (a by-design security boundary) — so each physical
/// mapping is registered as its own single-buffer dmabuf MR and the mappings are
/// stitched back together in software. A segment carries that MR's keys and the
/// RDMA address a peer uses, plus where the mapping sits in the logical region.
#[derive(Debug, Clone)]
pub(super) struct MrSegment {
    /// Byte offset of this mapping from the region base.
    pub offset: usize,
    /// Length of this mapping in bytes (one physical mapping).
    pub size: usize,
    /// RDMA address of this mapping's MR; a peer addresses this segment here.
    pub rdma_addr: usize,
    pub lkey: u32,
    pub rkey: u32,
}

/// A cloneable handle to a slice of registered memory: the keys and addresses
/// a peer needs, plus an `Arc<dyn IbvMemoryRegionKeepalive>` keepalive.
///
/// Cheap to clone; every clone shares the same guard, so the backing
/// registration stays alive (and registered) until the last clone drops.
#[derive(Debug, Clone)]
pub struct IbvMemoryRegionView {
    /// Virtual address in the local process address space.
    pub virtual_addr: usize,
    /// RDMA address, possibly offset from the region's base MR address.
    pub rdma_addr: usize,
    pub size: usize,
    pub lkey: u32,
    pub rkey: u32,
    /// Name of the RDMA device the view's protection domain is on.
    pub device_name: String,
    /// When the region spans multiple physical mappings (ROCm multi-mapping
    /// HIP-VMM memory registered one MR per mapping), the per-mapping segments
    /// in ascending `offset` order, covering `[0, size)` with no gaps or
    /// overlaps. `None` is the common single-MR fast path, where the flat
    /// `rdma_addr`/`lkey`/`rkey` address the whole region on their own.
    ///
    /// For a segmented view the flat `rdma_addr`/`lkey`/`rkey` mirror
    /// `segments[0]`, so a read confined to the first mapping still works; a
    /// transfer that may cross a mapping boundary must consult `segments`.
    pub(super) segments: Option<Arc<Vec<MrSegment>>>,
    /// Keeps the backing registration alive for every clone of this view; the
    /// last drop releases its resources. Never read directly.
    pub(super) _guard: Arc<dyn IbvMemoryRegionKeepalive>,
}

impl IbvMemoryRegionView {
    pub(super) fn new(
        virtual_addr: usize,
        rdma_addr: usize,
        size: usize,
        lkey: u32,
        rkey: u32,
        device_name: String,
        guard: Arc<dyn IbvMemoryRegionKeepalive>,
    ) -> Self {
        Self {
            virtual_addr,
            rdma_addr,
            size,
            lkey,
            rkey,
            device_name,
            segments: None,
            _guard: guard,
        }
    }

    /// Builds a view over a region registered as `segments` separate per-mapping
    /// MRs (ROCm multi-mapping HIP-VMM memory). `segments` must be in ascending
    /// `offset` order and tile `[0, total_size)` with no gaps or overlaps;
    /// `guard` must keep every one of those MRs registered (see [`SegmentedMrs`]).
    ///
    /// A single-segment list collapses to the flat single-MR form, so the
    /// segmented machinery only ever carries a genuinely multi-mapping region.
    pub(super) fn new_segmented(
        virtual_addr: usize,
        total_size: usize,
        device_name: String,
        segments: Vec<MrSegment>,
        guard: Arc<dyn IbvMemoryRegionKeepalive>,
    ) -> Self {
        debug_assert!(
            !segments.is_empty()
                && segments.first().map(|s| s.offset) == Some(0)
                && segments
                    .windows(2)
                    .all(|w| w[0].offset + w[0].size == w[1].offset)
                && segments.last().map(|s| s.offset + s.size) == Some(total_size),
            "segments must tile [0, {total_size}) in order with no gaps",
        );
        // The flat fields mirror the first segment, so a read confined to it
        // (the single-segment case included) works without consulting the table.
        let first = &segments[0];
        let (rdma_addr, lkey, rkey) = (first.rdma_addr, first.lkey, first.rkey);
        let segments = (segments.len() > 1).then(|| Arc::new(segments));
        Self {
            virtual_addr,
            rdma_addr,
            size: total_size,
            lkey,
            rkey,
            device_name,
            segments,
            _guard: guard,
        }
    }

    /// Whether this region is registered as more than one physical mapping, so a
    /// transfer crossing a mapping boundary must be split along [`Self::segments`].
    // Used by tests and by the step-5 transfer path; not yet read elsewhere.
    #[allow(dead_code)]
    pub(super) fn is_segmented(&self) -> bool {
        self.segments.is_some()
    }

    /// The per-mapping segments, or `None` for the single-MR fast path.
    pub(super) fn segments(&self) -> Option<&[MrSegment]> {
        self.segments.as_deref().map(Vec::as_slice)
    }

    /// Returns a view of `size` bytes starting at `offset`.
    ///
    /// The slice shares the same registration, keys, and device, and keeps the
    /// registration alive. For a segmented region the slice carries only the
    /// mappings it overlaps, each trimmed and rebased to the slice origin; a
    /// slice that lands within a single mapping collapses to the flat form.
    pub fn try_slice(&self, offset: usize, size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            offset.checked_add(size).is_some_and(|end| end <= self.size),
            "slice [{offset}, {offset}+{size}) is out of bounds for a {}-byte registration of \
             [{:#x}, {:#x}) on {}",
            self.size,
            self.rdma_addr,
            self.rdma_addr + self.size,
            self.device_name,
        );
        // Single-MR fast path, and the degenerate zero-byte slice (which needs no
        // segment table): shift both addresses and keep the keys. For a segmented
        // view the flat `rdma_addr` mirrors `segments[0]`, so a zero-byte view's
        // address is nominal — it is never used to move bytes.
        if self.segments.is_none() || size == 0 {
            return Ok(Self {
                virtual_addr: self.virtual_addr + offset,
                rdma_addr: self.rdma_addr + offset,
                size,
                segments: None,
                ..self.clone()
            });
        }
        // Segmented: keep the mappings overlapping [offset, offset+size), each
        // trimmed to the slice and rebased so the slice begins at offset 0. With
        // size > 0 over a gapless tiling, at least one mapping overlaps.
        let (start, end) = (offset, offset + size);
        let sliced: Vec<MrSegment> = self
            .segments()
            .expect("segmented in this arm")
            .iter()
            .filter_map(|seg| {
                let (lo, hi) = (seg.offset.max(start), (seg.offset + seg.size).min(end));
                (lo < hi).then(|| MrSegment {
                    offset: lo - start,
                    size: hi - lo,
                    rdma_addr: seg.rdma_addr + (lo - seg.offset),
                    lkey: seg.lkey,
                    rkey: seg.rkey,
                })
            })
            .collect();
        Ok(Self::new_segmented(
            self.virtual_addr + offset,
            size,
            self.device_name.clone(),
            sliced,
            Arc::clone(&self._guard),
        ))
    }

    /// Fabricate a zero-sized view naming `device_name` and carrying `key` as
    /// both its `lkey` and its `rkey`, over a null MR keepalive whose `Drop`
    /// is a no-op. For tests that care only about which device a registration
    /// belongs to, and which of several registrations they are holding.
    #[cfg(test)]
    pub(crate) fn for_test(device_name: &str, key: u32) -> Self {
        Self::new(
            0,
            0,
            0,
            key,
            key,
            device_name.to_string(),
            Arc::new(IbvMr::null()),
        )
    }
}

/// One physical mapping of a peer's segmented registration, as the wire carries
/// it: the wire form of an [`MrSegment`], without the local-only `lkey`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteMrSegment {
    /// Byte offset of this mapping from the region base.
    pub offset: usize,
    /// Length of this mapping in bytes.
    pub size: usize,
    /// RDMA address of this mapping's MR; the peer targets this segment here.
    pub addr: usize,
    pub rkey: u32,
}

/// What a peer needs in order to address one of our registered memory regions
/// over RDMA: the region's `rkey`, its RDMA address, its size, and the device
/// serving it.
///
/// This is the wire form of an [`IbvMemoryRegionView`]. It carries no `lkey` and
/// no keepalive: an `lkey` only means anything to the protection domain that
/// issued it, and the registration is kept alive by the views the owning side
/// holds.
///
/// A multi-mapping region (ROCm `expandable_segments` / vLLM cumem, registered
/// one MR per physical mapping) carries its per-mapping `segments`; a transfer
/// crossing a mapping boundary must be split along them. `None` is the common
/// single-MR case, where `rkey`/`addr` address the whole region on their own and
/// the field is omitted from the wire entirely.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub struct IbvRemoteMemoryRegionView {
    pub rkey: u32,
    /// RDMA address (may differ from virtual address).
    pub addr: usize,
    pub size: usize,
    /// Name of the RDMA device this region is registered on (e.g., "mlx5_0").
    pub device_name: String,
    /// Per-mapping segments for a multi-mapping region, in ascending `offset`
    /// order, tiling `[0, size)`; `None` for the single-MR fast path. For a
    /// segmented region `rkey`/`addr` mirror `segments[0]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<RemoteMrSegment>>,
}

impl IbvRemoteMemoryRegionView {
    /// Builds a remote view over `segments` separate per-mapping MRs. `segments`
    /// must be in ascending `offset` order and tile `[0, total_size)` with no
    /// gaps or overlaps. A single-segment list collapses to the flat form, so a
    /// segmented remote view is always genuinely multi-mapping.
    fn from_segments(total_size: usize, device_name: String, segments: Vec<RemoteMrSegment>) -> Self {
        debug_assert!(
            !segments.is_empty()
                && segments.first().map(|s| s.offset) == Some(0)
                && segments
                    .windows(2)
                    .all(|w| w[0].offset + w[0].size == w[1].offset)
                && segments.last().map(|s| s.offset + s.size) == Some(total_size),
            "segments must tile [0, {total_size}) in order with no gaps",
        );
        let first = &segments[0];
        let (rkey, addr) = (first.rkey, first.addr);
        let segments = (segments.len() > 1).then_some(segments);
        Self {
            rkey,
            addr,
            size: total_size,
            device_name,
            segments,
        }
    }

    /// Returns a view of `size` bytes starting at `offset` in the peer's region.
    /// The slice retains the remote key(s) and device. For a segmented region it
    /// keeps only the mappings it overlaps, each trimmed and rebased to the slice
    /// origin; a slice landing within a single mapping collapses to the flat form.
    pub fn try_slice(&self, offset: usize, size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            offset.checked_add(size).is_some_and(|end| end <= self.size),
            "slice [{offset}, {offset}+{size}) is out of bounds for a {}-byte peer registration \
             of [{:#x}, {:#x}) on {}",
            self.size,
            self.addr,
            self.addr + self.size,
            self.device_name,
        );
        // Single-MR fast path, and the degenerate zero-byte slice: shift the
        // address, keep the key. For a segmented view the flat `addr` mirrors
        // `segments[0]`, so a zero-byte view's address is nominal.
        if self.segments.is_none() || size == 0 {
            return Ok(Self {
                addr: self.addr + offset,
                size,
                segments: None,
                ..self.clone()
            });
        }
        let (start, end) = (offset, offset + size);
        let sliced: Vec<RemoteMrSegment> = self
            .segments
            .as_ref()
            .expect("segmented in this arm")
            .iter()
            .filter_map(|seg| {
                let (lo, hi) = (seg.offset.max(start), (seg.offset + seg.size).min(end));
                (lo < hi).then(|| RemoteMrSegment {
                    offset: lo - start,
                    size: hi - lo,
                    addr: seg.addr + (lo - seg.offset),
                    rkey: seg.rkey,
                })
            })
            .collect();
        Ok(Self::from_segments(size, self.device_name.clone(), sliced))
    }
}

impl From<&IbvMemoryRegionView> for IbvRemoteMemoryRegionView {
    /// The wire transport details are fully derived from the registered MR view:
    /// the remote key(s), the RDMA address(es), the size, and the device name.
    /// A segmented local view carries its per-mapping MRs over as segments.
    fn from(view: &IbvMemoryRegionView) -> Self {
        match view.segments() {
            None => Self {
                rkey: view.rkey,
                addr: view.rdma_addr,
                size: view.size,
                device_name: view.device_name.clone(),
                segments: None,
            },
            Some(segments) => Self::from_segments(
                view.size,
                view.device_name.clone(),
                segments
                    .iter()
                    .map(|seg| RemoteMrSegment {
                        offset: seg.offset,
                        size: seg.size,
                        addr: seg.rdma_addr,
                        rkey: seg.rkey,
                    })
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(rdma_addr: usize, size: usize) -> IbvMemoryRegionView {
        IbvMemoryRegionView::new(
            0x1000,
            rdma_addr,
            size,
            0x1234,
            0x5678,
            "mlx5_0".to_string(),
            Arc::new(IbvMr::null()),
        )
    }

    #[test]
    fn a_slice_moves_both_addresses_and_keeps_the_keys() {
        let sliced = view(0x4000, 4096)
            .try_slice(1024, 512)
            .expect("a slice inside the region");
        assert_eq!(sliced.rdma_addr, 0x4400, "the RDMA address advances");
        assert_eq!(sliced.virtual_addr, 0x1400, "so does the virtual address");
        assert_eq!(sliced.size, 512);
        assert_eq!((sliced.lkey, sliced.rkey), (0x1234, 0x5678));
        assert_eq!(sliced.device_name, "mlx5_0");
    }

    #[test]
    fn a_remote_slice_moves_the_address_and_keeps_the_rkey() {
        let remote = IbvRemoteMemoryRegionView::from(&view(0x4000, 4096));
        let sliced = remote.try_slice(2048, 2048).expect("a slice at the end");
        assert_eq!((sliced.addr, sliced.size), (0x4800, 2048));
        assert_eq!(sliced.rkey, 0x5678);
        assert_eq!(sliced.device_name, "mlx5_0");
    }

    #[test]
    fn slices_are_bounded_by_the_registration() {
        let local = view(0x4000, 4096);
        let remote = IbvRemoteMemoryRegionView::from(&local);
        assert!(local.try_slice(0, 4096).is_ok());
        assert!(local.try_slice(4096, 0).is_ok());
        assert!(local.try_slice(0, 4097).is_err());
        assert!(local.try_slice(4096, 1).is_err());
        assert!(local.try_slice(1, usize::MAX).is_err());
        assert!(remote.try_slice(2048, 2049).is_err());
    }

    /// A three-mapping region: mappings of 0x1000 each at RDMA addresses
    /// 0xA000 / 0xB000 / 0xC000 (deliberately non-contiguous, as per-mapping MRs
    /// are) with distinct keys, covering virtual `[0x1000, 0x4000)`.
    fn segmented_view() -> IbvMemoryRegionView {
        let segments = vec![
            MrSegment { offset: 0x0000, size: 0x1000, rdma_addr: 0xA000, lkey: 1, rkey: 11 },
            MrSegment { offset: 0x1000, size: 0x1000, rdma_addr: 0xB000, lkey: 2, rkey: 12 },
            MrSegment { offset: 0x2000, size: 0x1000, rdma_addr: 0xC000, lkey: 3, rkey: 13 },
        ];
        IbvMemoryRegionView::new_segmented(
            0x1000,
            0x3000,
            "mlx5_0".to_string(),
            segments,
            Arc::new(IbvMr::null()),
        )
    }

    #[test]
    fn a_segmented_view_tiles_its_mappings_and_mirrors_the_first() {
        let v = segmented_view();
        assert!(v.is_segmented());
        assert_eq!(v.size, 0x3000);
        // Flat fields mirror segment 0 so a read confined to it still works.
        assert_eq!((v.rdma_addr, v.lkey, v.rkey), (0xA000, 1, 11));
        let segs = v.segments().expect("segmented");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[2].offset, 0x2000);
        assert_eq!(segs[2].rdma_addr, 0xC000);
    }

    #[test]
    fn a_single_segment_collapses_to_the_flat_form() {
        let v = IbvMemoryRegionView::new_segmented(
            0x1000,
            0x1000,
            "mlx5_0".to_string(),
            vec![MrSegment { offset: 0, size: 0x1000, rdma_addr: 0xA000, lkey: 7, rkey: 9 }],
            Arc::new(IbvMr::null()),
        );
        assert!(!v.is_segmented(), "one segment uses the single-MR fast path");
        assert_eq!((v.rdma_addr, v.lkey, v.rkey, v.size), (0xA000, 7, 9, 0x1000));
    }

    #[test]
    fn a_slice_within_one_mapping_collapses_and_rebases_onto_that_mapping() {
        // [0x1800, 0x1A00) lives entirely in segment 1 (RDMA base 0xB000),
        // 0x800 into it.
        let sliced = segmented_view()
            .try_slice(0x1800, 0x200)
            .expect("inside segment 1");
        assert!(!sliced.is_segmented(), "a single-mapping slice is flat");
        assert_eq!(sliced.virtual_addr, 0x2800);
        assert_eq!(sliced.rdma_addr, 0xB800, "rebased onto segment 1's MR");
        assert_eq!((sliced.lkey, sliced.rkey, sliced.size), (2, 12, 0x200));
    }

    #[test]
    fn a_slice_crossing_mappings_keeps_each_overlapped_mapping_trimmed() {
        // [0x800, 0x2800): last half of seg 0, all of seg 1, first half of seg 2.
        let sliced = segmented_view()
            .try_slice(0x800, 0x2000)
            .expect("spans three mappings");
        let segs = sliced.segments().expect("still segmented");
        assert_eq!(segs.len(), 3);
        // Rebased so the slice starts at offset 0; addresses trimmed into each MR.
        assert_eq!(
            (segs[0].offset, segs[0].size, segs[0].rdma_addr, segs[0].rkey),
            (0x0, 0x800, 0xA800, 11),
            "seg 0: its second half, rebased to slice offset 0",
        );
        assert_eq!(
            (segs[1].offset, segs[1].size, segs[1].rdma_addr, segs[1].rkey),
            (0x800, 0x1000, 0xB000, 12),
            "seg 1: in full",
        );
        assert_eq!(
            (segs[2].offset, segs[2].size, segs[2].rdma_addr, segs[2].rkey),
            (0x1800, 0x800, 0xC000, 13),
            "seg 2: its first half",
        );
        // Flat fields mirror the first retained mapping.
        assert_eq!((sliced.rdma_addr, sliced.rkey, sliced.size), (0xA800, 11, 0x2000));
    }

    #[test]
    fn a_slice_aligned_to_mapping_boundaries_drops_untouched_mappings() {
        // [0x1000, 0x2000) is exactly segment 1.
        let sliced = segmented_view().try_slice(0x1000, 0x1000).expect("segment 1");
        assert!(!sliced.is_segmented(), "exactly one mapping -> flat");
        assert_eq!((sliced.rdma_addr, sliced.rkey), (0xB000, 12));

        // [0x1000, 0x3000) is segments 1 and 2.
        let sliced = segmented_view().try_slice(0x1000, 0x2000).expect("segments 1,2");
        let segs = sliced.segments().expect("two mappings");
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].rdma_addr, segs[0].rkey), (0xB000, 12));
        assert_eq!((segs[1].offset, segs[1].rdma_addr, segs[1].rkey), (0x1000, 0xC000, 13));
    }

    #[test]
    fn segmented_slices_are_bounded_and_zero_is_degenerate() {
        let v = segmented_view();
        assert!(v.try_slice(0, 0x3000).is_ok(), "the whole region");
        assert!(v.try_slice(0x3000, 1).is_err(), "past the end");
        assert!(v.try_slice(0x2800, 0x1000).is_err(), "runs off the end");
        let zero = v.try_slice(0x3000, 0).expect("empty slice at the end");
        assert!(!zero.is_segmented(), "a zero-byte slice carries no segment table");
        assert_eq!(zero.size, 0);
    }

    #[test]
    fn remote_from_a_segmented_local_view_carries_every_mapping() {
        let remote = IbvRemoteMemoryRegionView::from(&segmented_view());
        assert_eq!(remote.size, 0x3000);
        // Flat fields mirror segment 0 (no lkey/virtual_addr on the wire form).
        assert_eq!((remote.rkey, remote.addr), (11, 0xA000));
        let segs = remote.segments.as_ref().expect("segmented on the wire");
        assert_eq!(segs.len(), 3);
        assert_eq!((segs[0].offset, segs[0].size, segs[0].addr, segs[0].rkey), (0x0, 0x1000, 0xA000, 11));
        assert_eq!((segs[1].offset, segs[1].size, segs[1].addr, segs[1].rkey), (0x1000, 0x1000, 0xB000, 12));
        assert_eq!((segs[2].offset, segs[2].size, segs[2].addr, segs[2].rkey), (0x2000, 0x1000, 0xC000, 13));
    }

    #[test]
    fn remote_from_a_single_mr_view_has_no_segment_table() {
        // The common single-MR view produces no segments, so the wire form is
        // byte-for-byte what it was before segmentation existed.
        let remote = IbvRemoteMemoryRegionView::from(&view(0x4000, 4096));
        assert!(remote.segments.is_none());
        assert_eq!((remote.rkey, remote.addr, remote.size), (0x5678, 0x4000, 4096));
    }

    #[test]
    fn a_remote_slice_crossing_mappings_keeps_each_trimmed() {
        let remote = IbvRemoteMemoryRegionView::from(&segmented_view());
        // [0x800, 0x2800): last half of seg 0, all of seg 1, first half of seg 2.
        let sliced = remote.try_slice(0x800, 0x2000).expect("spans three mappings");
        let segs = sliced.segments.as_ref().expect("still segmented");
        assert_eq!(segs.len(), 3);
        assert_eq!((segs[0].offset, segs[0].size, segs[0].addr, segs[0].rkey), (0x0, 0x800, 0xA800, 11));
        assert_eq!((segs[1].offset, segs[1].size, segs[1].addr, segs[1].rkey), (0x800, 0x1000, 0xB000, 12));
        assert_eq!((segs[2].offset, segs[2].size, segs[2].addr, segs[2].rkey), (0x1800, 0x800, 0xC000, 13));
        assert_eq!((sliced.rkey, sliced.addr, sliced.size), (11, 0xA800, 0x2000));
    }

    #[test]
    fn a_remote_slice_within_one_mapping_collapses_to_flat() {
        let remote = IbvRemoteMemoryRegionView::from(&segmented_view());
        // [0x1800, 0x1A00) is entirely in segment 1 (RDMA base 0xB000), 0x800 in.
        let sliced = remote.try_slice(0x1800, 0x200).expect("inside segment 1");
        assert!(sliced.segments.is_none(), "a single-mapping slice is flat");
        assert_eq!((sliced.rkey, sliced.addr, sliced.size), (12, 0xB800, 0x200));
    }

    #[test]
    fn remote_segmented_slices_are_bounded() {
        let remote = IbvRemoteMemoryRegionView::from(&segmented_view());
        assert!(remote.try_slice(0, 0x3000).is_ok());
        assert!(remote.try_slice(0x3000, 1).is_err());
        assert!(remote.try_slice(0x2800, 0x1000).is_err());
    }
}
