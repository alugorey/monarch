/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Broadcom backend for [`IbvDevice`].

use std::sync::Arc;

use typeuri::Named;

use super::device::IbvContext;
use super::device::IbvDeviceImpl;
use super::broadcom_domain::BroadcomDomain;
use super::primitives::IbvConfig;
use crate::register_ibv_device_impl;

/// PCI vendor ID for Broadcom.
const BROADCOM_VENDOR_ID: u32 = 0x14e4;

/// Broadcom NetXtreme (bnxt_re) backend.
#[derive(Debug, Named)]
pub struct BroadcomDevice;

impl IbvDeviceImpl for BroadcomDevice {
    type IbvDomainImpl = BroadcomDomain;

    fn backend_name() -> &'static str {
        "broadcom"
    }

    fn is_instance(ctx: Arc<IbvContext>) -> bool {
        // Match by PCI vendor id, mirroring `MlxDevice`. `is_instance` runs
        // only on devices ibverbs already enumerated as RDMA-capable, so
        // non-RoCE Broadcom NICs (e.g. the management BCM5720) are never seen
        // here; every Broadcom RDMA device is bnxt_re.
        let mut attr = rdmaxcel_sys::ibv_device_attr::default();
        // SAFETY: `ctx.as_ptr()` is a non-null context owned by the
        // `Arc<IbvContext>` for the duration of this call; `&mut attr` is a
        // writable, properly aligned `ibv_device_attr`.
        let queried = unsafe { rdmaxcel_sys::ibv_query_device(ctx.as_ptr(), &mut attr) } == 0;
        queried && attr.vendor_id == BROADCOM_VENDOR_ID
    }

    fn apply_config_defaults(_config: &mut IbvConfig) {
        // BROADCOM-SPECIFIC FLAG: RoCEv2 GID selection (`config.gid_index`)
        // and SGE/atomic caps may need adapter-specific values here, the way
        // `EfaDevice` sets its own limits. The standard QP type is pinned in
        // `BroadcomDomain::create_queue_pair`, so nothing is required here for
        // basic operation. Left at the generic defaults until validated on
        // hardware.
    }
}

register_ibv_device_impl!(BroadcomDevice);
