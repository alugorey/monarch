/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! AMD Pensando (ionic) backend for [`IbvDevice`].

use std::sync::Arc;

use typeuri::Named;

use super::device::IbvDeviceImpl;
use super::ionic_domain::IonicDomain;
use super::primitives::IbvConfig;
use super::primitives::IbvContext;
use super::primitives::IbvQpType;
use crate::register_ibv_device_impl;

/// PCI vendor ID for AMD Pensando Systems.
pub(super) const PENSANDO_VENDOR_ID: u32 = 0x1dd8;

/// Scatter/gather entries per work request that ionic devices accept
/// (`ibv_query_device` reports `max_sge = 8`, below the generic default of
/// 30). [`IonicDomain`] re-checks against the queried device limit when it
/// builds a queue pair.
pub(super) const IONIC_MAX_SGE: u32 = 8;

/// AMD Pensando AINIC (`ionic` provider) backend. RoCE v2, standard RC queue
/// pairs, host and dmabuf memory registration.
#[derive(Debug, Named)]
pub struct IonicDevice;

impl IonicDevice {
    /// Whether a device reporting PCI vendor `vendor_id` is an ionic device.
    fn is_ionic_vendor(vendor_id: u32) -> bool {
        vendor_id == PENSANDO_VENDOR_ID
    }
}

impl IbvDeviceImpl for IonicDevice {
    type Domain = IonicDomain;

    fn backend_name() -> &'static str {
        "ionic"
    }

    fn is_instance(ctx: Arc<IbvContext>) -> bool {
        let mut attr = rdmaxcel_sys::ibv_device_attr::default();
        // SAFETY: `ctx.as_ptr()` is a non-null context owned by
        // the `Arc<IbvContext>` for the duration of this call;
        // `&mut attr` is a writable, properly aligned
        // `ibv_device_attr`.
        let queried = unsafe { rdmaxcel_sys::ibv_query_device(ctx.as_ptr(), &mut attr) } == 0;
        queried && Self::is_ionic_vendor(attr.vendor_id)
    }

    /// Seeds ionic limits over the generic defaults. Checked against
    /// `ibv_devinfo -v` on AINIC 25.08 (part 4099):
    ///
    /// - `qp_type`: ionic has no mlx5dv or EFA verbs. `Auto` resolves from a
    ///   process-wide probe of whichever ibverbs device enumerates first, so
    ///   on a host that also has an mlx5 NIC it can pick mlx5dv for ionic.
    ///   Only the legacy queue-pair path reads it; [`IonicDomain`] always
    ///   builds a standard RC queue pair.
    /// - `max_send_sge`/`max_recv_sge`: default 30 exceeds the device's
    ///   `max_sge` of 8, and `ibv_create_qp` rejects the excess.
    ///
    /// The other defaults already fit and are left alone: `max_send_wr` /
    /// `max_recv_wr` 512 (device `max_qp_wr` 65535, `max_cqe` 65435),
    /// `path_mtu` 4096 (device `max_mtu`/`active_mtu` 4096),
    /// `max_rd_atomic`/`max_dest_rd_atomic` 16 (device `max_qp_init_rd_atom` /
    /// `max_qp_rd_atom` 16). The source GID is chosen at queue-pair creation as
    /// the first global RoCE v2 GID (index 1 on these ports), not from config.
    fn apply_config_defaults(config: &mut IbvConfig) {
        config.qp_type = IbvQpType::Standard;
        config.max_send_sge = config.max_send_sge.min(IONIC_MAX_SGE);
        config.max_recv_sge = config.max_recv_sge.min(IONIC_MAX_SGE);
    }
}

register_ibv_device_impl!(IonicDevice);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ibverbs::device::IbvDevice;

    #[test]
    fn is_ionic_vendor_matches_only_pensando() {
        assert!(IonicDevice::is_ionic_vendor(0x1dd8));
        // Mellanox, Broadcom, and an unset vendor id are not ionic.
        assert!(!IonicDevice::is_ionic_vendor(0x02c9));
        assert!(!IonicDevice::is_ionic_vendor(0x14e4));
        assert!(!IonicDevice::is_ionic_vendor(0));
    }

    #[test]
    fn apply_config_defaults_fits_ionic_limits() {
        let mut config = IbvConfig::default();
        IonicDevice::apply_config_defaults(&mut config);
        assert_eq!(config.qp_type, IbvQpType::Standard);
        assert_eq!(config.max_send_sge, IONIC_MAX_SGE);
        assert_eq!(config.max_recv_sge, IONIC_MAX_SGE);

        // Defaults that already fit the device are left untouched.
        let default = IbvConfig::default();
        assert_eq!(config.max_send_wr, default.max_send_wr);
        assert_eq!(config.max_recv_wr, default.max_recv_wr);
        assert_eq!(config.path_mtu, default.path_mtu);
        assert_eq!(config.max_rd_atomic, default.max_rd_atomic);
        assert_eq!(config.max_dest_rd_atomic, default.max_dest_rd_atomic);
    }

    #[test]
    fn apply_config_defaults_keeps_smaller_sge() {
        let mut config = IbvConfig {
            max_send_sge: 1,
            max_recv_sge: 2,
            ..Default::default()
        };
        IonicDevice::apply_config_defaults(&mut config);
        assert_eq!(config.max_send_sge, 1);
        assert_eq!(config.max_recv_sge, 2);
    }

    /// On a host with ionic NICs, every device the registry assigns to
    /// `IonicDevice` reports the Pensando vendor id and is named `ionic_*`.
    /// Skips when no ionic device is present.
    #[test]
    fn ionic_devices_are_claimed_by_ionic_backend() {
        let devices = IbvDevice::<IonicDevice>::list();
        if devices.is_empty() {
            eprintln!("no ionic devices on this host; skipping");
            return;
        }
        for info in &devices {
            assert_eq!(info.vendor_id(), PENSANDO_VENDOR_ID, "{}", info.name());
            assert!(info.name().starts_with("ionic_"), "{}", info.name());
            assert!(info.max_sge() >= IONIC_MAX_SGE as i32, "{}", info.name());
        }
    }
}
