/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Broadcom domain strategy for [`IbvDomainImpl`].

use std::sync::Arc;

use super::device::IbvContext;
use super::domain::IbvDomain;
use super::domain::IbvDomainImpl;
use super::primitives::IbvConfig;
use super::primitives::IbvDeviceInfo;
use super::primitives::IbvQpType;
use super::queue_pair::IbvQueuePair;

/// Broadcom (bnxt_re) [`IbvDomainImpl`]. Standard RoCE RC: uses the default
/// host/dmabuf MR registration; Broadcom has no device-specific memory-key
/// binding to add (unlike mlx5dv indirect mkeys).
#[derive(Debug)]
pub struct BroadcomDomain;

impl IbvDomainImpl for BroadcomDomain {
    unsafe fn new(
        _context: &IbvContext,
        _device_info: &IbvDeviceInfo,
        _config: &IbvConfig,
    ) -> Self {
        BroadcomDomain
    }

    fn mr_access_flags(&self) -> i32 {
        // Standard RoCE access set
        (rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_READ
            | rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC)
            .0 as i32
    }

    fn create_queue_pair(
        domain: Arc<IbvDomain<Self>>,
        config: &IbvConfig,
    ) -> anyhow::Result<IbvQueuePair> {
        // Broadcom uses standard RC queue pairs (not mlx5dv extended QPs or
        // EFA SRD). Pin the standard QP type so the shared `resolve_qp_type`
        // Auto path (which keys off global mlx5dv/EFA detection) cannot
        // mis-select another backend's QP type on a mixed-vendor host.
        let mut config = config.clone();
        config.qp_type = IbvQpType::Standard;
        IbvQueuePair::new(domain, config)
    }
}
