/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! AMD Pensando (ionic) domain strategy for [`IbvDomainImpl`].

use std::sync::Arc;

use super::domain::IbvDomain;
use super::domain::IbvDomainImpl;
use super::primitives::IbvConfig;
use super::primitives::IbvContext;
use super::primitives::IbvCq;
use super::primitives::IbvDeviceInfo;
use super::queue_pair::IbvQueuePair;
use super::queue_pair::RCQueuePair;

/// ionic [`IbvDomainImpl`]. Standard RoCE v2 RC queue pairs over plain
/// ibverbs, and the default host (`ibv_reg_mr`) / device-memory
/// (`ibv_reg_dmabuf_mr`) MR registration; ionic has no device-specific
/// memory-key binding to add (unlike mlx5dv indirect mkeys).
#[derive(Debug)]
pub struct IonicDomain;

/// Lowers `config`'s scatter/gather caps to the device's `max_sge`. A
/// non-positive `max_sge` (an unqueried device) leaves `config` unchanged.
fn fit_sge_to_device(config: &mut IbvConfig, max_sge: i32) {
    if let Ok(max_sge) = u32::try_from(max_sge)
        && max_sge > 0
    {
        config.max_send_sge = config.max_send_sge.min(max_sge);
        config.max_recv_sge = config.max_recv_sge.min(max_sge);
    }
}

impl IbvDomainImpl for IonicDomain {
    type QueuePair = RCQueuePair;

    unsafe fn new(
        _context: &IbvContext,
        _device_info: &IbvDeviceInfo,
        _config: &IbvConfig,
    ) -> Self {
        IonicDomain
    }

    fn access_flags(&self) -> i32 {
        // The device reports `ATOMIC_GLOB`, so grant remote atomics like mlx5.
        (rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_READ
            | rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC)
            .0 as i32
    }

    /// Builds an [`RCQueuePair`] after fitting the scatter/gather caps to the
    /// device. `IonicDevice::apply_config_defaults` already does this, but the
    /// manager seeds those defaults only when it spawns without an explicit
    /// config, and one explicit [`IbvConfig`] is shared by every ibverbs
    /// backend. Its generic default of 30 SGEs would fail `ibv_create_qp`
    /// here (ionic `max_sge` is 8).
    unsafe fn create_queue_pair(
        domain: &IbvDomain<Self>,
        config: &IbvConfig,
        cq: Arc<IbvCq>,
    ) -> anyhow::Result<RCQueuePair> {
        if domain.as_ptr().is_null() {
            anyhow::bail!("cannot create a queue pair on a null protection domain");
        }
        let mut config = config.clone();
        fit_sge_to_device(&mut config, domain.device_info().max_sge());
        // SAFETY: `domain` holds a live PD (null was rejected above) and `cq` a
        // live queue on its context, per this method's contract, which is what
        // `IbvQueuePair::new` requires.
        unsafe { RCQueuePair::new(domain, config, Arc::clone(&cq), cq) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ibverbs::device::IbvDevice;
    use crate::backend::ibverbs::device::IbvDeviceImpl;
    use crate::backend::ibverbs::ionic_device::IonicDevice;
    use crate::backend::ibverbs::primitives::IbvPd;

    #[test]
    fn fit_sge_to_device_lowers_only_oversized_caps() {
        let mut config = IbvConfig {
            max_send_sge: 30,
            max_recv_sge: 4,
            ..Default::default()
        };
        fit_sge_to_device(&mut config, 8);
        assert_eq!(config.max_send_sge, 8);
        assert_eq!(config.max_recv_sge, 4);
    }

    #[test]
    fn fit_sge_to_device_ignores_unqueried_limit() {
        let mut config = IbvConfig::default();
        fit_sge_to_device(&mut config, 0);
        assert_eq!(config.max_send_sge, IbvConfig::default().max_send_sge);
        fit_sge_to_device(&mut config, -1);
        assert_eq!(config.max_recv_sge, IbvConfig::default().max_recv_sge);
    }

    #[test]
    fn access_flags_include_remote_atomic() {
        let flags = IonicDomain.access_flags();
        let atomic = rdmaxcel_sys::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC.0 as i32;
        assert_eq!(flags & atomic, atomic);
    }

    // A domain with no protection domain cannot build a queue pair, and says so
    // rather than reaching the driver with a null handle.
    #[test]
    fn create_queue_pair_rejects_null_pd() {
        // SAFETY: `IbvPd::null()` holds a null PD (and, through it, a null
        // context) whose `Drop`s are no-ops.
        let domain = unsafe {
            IbvDomain::for_test(
                Arc::new(IbvPd::null()),
                IbvDeviceInfo::for_test_named("ionic_0"),
                IonicDomain,
            )
        };
        let err = domain
            .create_queue_pair(&IbvConfig::default(), Arc::new(IbvCq::null()))
            .expect_err("a null protection domain cannot back a queue pair");
        assert!(
            err.to_string().contains("null protection domain"),
            "unexpected error: {err}"
        );
    }

    /// First ionic device on this host, or `None` (the hardware tests below
    /// then skip).
    fn first_ionic_device() -> Option<IbvDeviceInfo> {
        let device = IbvDevice::<IonicDevice>::list().into_iter().next();
        if device.is_none() {
            eprintln!("no ionic devices on this host; skipping");
        }
        device
    }

    /// On ionic hardware: the generic default of 30 SGEs is more than the
    /// device accepts, so a plain RC queue pair built from it fails. This is
    /// what `fit_sge_to_device` guards against.
    #[test]
    fn ionic_rejects_generic_default_sge() {
        let Some(info) = first_ionic_device() else {
            return;
        };
        let mut device = IbvDevice::<IonicDevice>::try_open(info.name(), IbvConfig::default())
            .expect("ionic device should open");
        let context = device.context();
        let domain = device
            .get_or_create_domain("test")
            .expect("ionic domain should be created");
        // SAFETY: `context` is the live context `device` holds open.
        let cq = Arc::new(unsafe { IbvCq::create(context, 1024) }.expect("create cq"));
        let config = IbvConfig::default();
        assert!(config.max_send_sge as i32 > domain.device_info().max_sge());
        // SAFETY: `domain` holds a live PD and `cq` a live queue on its context.
        let result = unsafe { RCQueuePair::new(domain, config, Arc::clone(&cq), cq) };
        assert!(
            result.is_err(),
            "ionic accepted max_send_sge above its max_sge; the SGE clamp may be unneeded"
        );
    }

    /// On ionic hardware: a queue pair builds from both the ionic-seeded
    /// defaults and the unmodified generic defaults (whose SGE caps the domain
    /// fits to the device).
    #[test]
    fn ionic_creates_queue_pair_from_default_configs() {
        let Some(info) = first_ionic_device() else {
            return;
        };
        let mut seeded = IbvConfig::default();
        IonicDevice::apply_config_defaults(&mut seeded);
        for config in [seeded, IbvConfig::default()] {
            let mut device = IbvDevice::<IonicDevice>::try_open(info.name(), config.clone())
                .expect("ionic device should open");
            let (_qp, _lease) = device
                .create_queue_pair("test", &config)
                .unwrap_or_else(|e| panic!("create QP on {} from {config}: {e:#}", info.name()));
        }
    }
}
