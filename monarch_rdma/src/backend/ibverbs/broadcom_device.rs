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
use super::primitives::IbvConfig;
use super::primitives::IbvQpType;
use crate::register_ibv_device_impl;

/// PCI vendor ID for Broadcom.
const BROADCOM_VENDOR_ID: u32 = 0x14e4;

/// Broadcom NetXtreme (bnxt_re) backend.
#[derive(Debug, Named)]
pub struct BroadcomDevice;

impl IbvDeviceImpl for BroadcomDevice {
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

    fn apply_config_defaults(config: &mut IbvConfig) {
        // Broadcom RoCE uses standard RC queue pairs (not mlx5dv extended QPs
        // or EFA SRD). Force the standard QP type so that, on a host that also
        // has mlx5 hardware present, `resolve_qp_type`'s Auto path (which keys
        // off the global `mlx5dv_supported()`) does not mis-select mlx5dv for
        // a Broadcom buffer.
        config.qp_type = IbvQpType::Standard;
        // BROADCOM-SPECIFIC FLAG: RoCEv2 GID selection (`config.gid_index`)
        // and SGE/atomic caps may need adapter-specific values here, the way
        // `EfaDevice` sets its own limits. Left at the generic defaults until
        // validated on hardware.
    }
}

register_ibv_device_impl!(BroadcomDevice);
