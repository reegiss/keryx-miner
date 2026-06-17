use crate::{Error, NonceGenEnum};
use cust::context::CurrentContext;
use cust::device::DeviceAttribute;
use cust::function::Function;
use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::*;
use keryx_miner::xoshiro256starstar::Xoshiro256StarStar;
use keryx_miner::Worker;
use log::{error, info};
use rand::{Fill, RngCore};
use std::ffi::CString;
use std::sync::{Arc, Weak};

static BPS: f32 = 1.;

static PTX_100: &str = include_str!("../resources/keryx-cuda-sm100.ptx");
static PTX_89: &str = include_str!("../resources/keryx-cuda-sm89.ptx");
static PTX_86: &str = include_str!("../resources/keryx-cuda-sm86.ptx");
static PTX_80: &str = include_str!("../resources/keryx-cuda-sm80.ptx");
static PTX_75: &str = include_str!("../resources/keryx-cuda-sm75.ptx");
static PTX_61: &str = include_str!("../resources/keryx-cuda-sm61.ptx");
// sm_30 (Kepler) and sm_20 (Fermi) dropped: CUDA 12+ no longer compiles for
// these architectures, and they predate practical GPU mining anyway.

pub struct Kernel<'kernel> {
    func: Arc<Function<'kernel>>,
    block_size: u32,
    grid_size: u32,
}

impl<'kernel> Kernel<'kernel> {
    pub fn new(module: Weak<Module>, name: &'kernel str) -> Result<Kernel<'kernel>, Error> {
        let func = Arc::new(unsafe {
            module.as_ptr().as_ref().unwrap().get_function(name).map_err(|e| {
                error!("Error loading function: {}", e);
                e
            })?
        });
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

        let device = CurrentContext::get_device()?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let grid_size = sm_count * func.max_active_blocks_per_multiprocessor(block_size.into(), 0)?;

        Ok(Self { func, block_size, grid_size })
    }

    pub fn get_workload(&self) -> u32 {
        self.block_size * self.grid_size
    }

    pub fn set_workload(&mut self, workload: u32) {
        self.grid_size = (workload + self.block_size - 1) / self.block_size
    }
}

pub struct CudaGPUWorker<'gpu> {
    // NOTE: The order is important! _context must be dropped last.
    heavy_hash_kernel: Kernel<'gpu>,

    // Ping stream: where the NEXT kernel launch goes.
    stream_ping: Stream,
    start_event_ping: Event,
    stop_event_ping: Event,
    rand_state_ping: DeviceBuffer<u64>,
    final_nonce_ping: DeviceBuffer<u64>,
    stream_ping_used: bool,

    // Pong stream: the previously launched stream, awaited in sync().
    stream_pong: Stream,
    start_event_pong: Event,
    stop_event_pong: Event,
    rand_state_pong: DeviceBuffer<u64>,
    final_nonce_pong: DeviceBuffer<u64>,
    stream_pong_used: bool,

    /// True once the first copy_output_to() has run; enables sync() to wait on pong.
    warmed_up: bool,

    _module: Arc<Module>,
    device_id: u32,
    pub workload: usize,
    _context: Context,
    random: NonceGenEnum,
}

impl<'gpu> Worker for CudaGPUWorker<'gpu> {
    fn id(&self) -> String {
        let device = CurrentContext::get_device().unwrap();
        format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) {
        let u8matrix: [[u8; 64]; 64] = matrix.map(|row| row.map(|v| v as u8));
        let mut hash_header_gpu = self._module.get_global::<[u8; 72]>(&CString::new("hash_header").unwrap()).unwrap();
        hash_header_gpu.copy_from(hash_header).map_err(|e| e.to_string()).unwrap();

        let mut matrix_gpu = self._module.get_global::<[[u8; 64]; 64]>(&CString::new("matrix").unwrap()).unwrap();
        matrix_gpu.copy_from(&u8matrix).map_err(|e| e.to_string()).unwrap();

        let mut target_gpu = self._module.get_global::<[u64; 4]>(&CString::new("target").unwrap()).unwrap();
        target_gpu.copy_from(target).map_err(|e| e.to_string()).unwrap();
    }

    #[inline(always)]
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) {
        let func = &self.heavy_hash_kernel.func;
        let stream = &self.stream_ping;
        let random: u8 = match self.random {
            NonceGenEnum::Lean => {
                self.rand_state_ping.copy_from(&[rand::thread_rng().next_u64()]).unwrap();
                0
            }
            NonceGenEnum::Xoshiro => 1,
        };

        self.start_event_ping.record(stream).unwrap();
        unsafe {
            launch!(
                func<<<
                    self.heavy_hash_kernel.grid_size, self.heavy_hash_kernel.block_size,
                    0, stream
                >>>(
                    nonce_mask, nonce_fixed,
                    self.workload,
                    random,
                    self.rand_state_ping.as_device_ptr(),
                    self.final_nonce_ping.as_device_ptr()
                )
            )
            .unwrap(); // We see errors in sync
        }
        self.stop_event_ping.record(stream).unwrap();
        self.stream_ping_used = true;
    }

    #[inline(always)]
    fn sync(&self) -> Result<(), Error> {
        if !self.warmed_up {
            // Pong hasn't been used yet (first iteration). Nothing to wait for.
            return Ok(());
        }
        self.stop_event_pong.synchronize()?;
        if self.stop_event_pong.elapsed_time_f32(&self.start_event_pong)? > 1000. / BPS {
            return Err("Cuda takes longer then block rate. Please reduce your workload.".into());
        }
        Ok(())
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    #[inline(always)]
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        // On the very first call warmed_up is false: pong was never launched so its
        // buffer is still zeroed — copy returns 0 which means "no nonce found". Safe.
        self.final_nonce_pong.copy_to(nonces)?;
        // Rotate: the stream just launched (ping) becomes the new pong for the next sync().
        self.swap_buffers();
        self.warmed_up = true;
        Ok(())
    }

    fn drain(&mut self) -> Result<(), Error> {
        // Flush whichever streams have been recorded.
        if self.stream_pong_used {
            self.stop_event_pong.synchronize()?;
        }
        if self.stream_ping_used {
            self.stop_event_ping.synchronize()?;
        }
        self.warmed_up = false;
        self.stream_ping_used = false;
        self.stream_pong_used = false;
        Ok(())
    }
}

impl<'gpu> CudaGPUWorker<'gpu> {
    /// Rotate ping → pong. After the swap, the stream that was just launched becomes
    /// pong (will be waited on next sync()), and the idle stream becomes the new ping.
    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.stream_ping, &mut self.stream_pong);
        std::mem::swap(&mut self.start_event_ping, &mut self.start_event_pong);
        std::mem::swap(&mut self.stop_event_ping, &mut self.stop_event_pong);
        std::mem::swap(&mut self.rand_state_ping, &mut self.rand_state_pong);
        std::mem::swap(&mut self.final_nonce_ping, &mut self.final_nonce_pong);
        std::mem::swap(&mut self.stream_ping_used, &mut self.stream_pong_used);
    }

    pub fn new(
        device_id: u32,
        workload: f32,
        is_absolute: bool,
        blocking_sync: bool,
        random: NonceGenEnum,
    ) -> Result<Self, Error> {
        info!("Starting a CUDA worker");
        let sync_flag = match blocking_sync {
            true => ContextFlags::SCHED_BLOCKING_SYNC,
            false => ContextFlags::SCHED_AUTO,
        };
        let device = Device::get_device(device_id).unwrap();
        let _context = Context::new(device)?;
        _context.set_flags(sync_flag)?;

        let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
        let _module: Arc<Module>;
        info!("Device #{} compute version is {}.{}", device_id, major, minor);

        let load_ptx = |ptx, label: &str| {
            Module::from_ptx(ptx, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Failed to load {} PTX (driver too old?): {}", label, e);
                e
            })
        };

        // For sm_89 (Ada/RTX 40) and sm_100 (Blackwell/RTX 50), the PTX was compiled with
        // CUDA 13.2 (PTX ISA 9.2) which requires driver >= 570. If the driver is older, we
        // fall back to sm_86 (CUDA 12.0 / PTX 8.0, driver >= 520) which runs on all these
        // architectures via NVIDIA's backward-compatible PTX JIT.
        if major >= 10 {
            // sm_100+ (RTX 50 / Blackwell and future)
            _module = Arc::new(match load_ptx(PTX_100, "sm_100") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_100 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_86 PTX (update driver to 570+ for full Blackwell optimisation)", device_id);
                    load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                }
            });
        } else if major == 9 || (major == 8 && minor >= 9) {
            // sm_89 (RTX 40 / Ada Lovelace)
            _module = Arc::new(match load_ptx(PTX_89, "sm_89") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_89 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_86 PTX (update driver to 570+ for full Ada Lovelace optimisation)", device_id);
                    load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                }
            });
        } else if major == 8 && minor >= 6 {
            // sm_86 (RTX 30 / Ampere)
            _module = Arc::new(load_ptx(PTX_86, "sm_86")?);
        } else if major == 8 {
            // sm_80 (A100 / CMP 170HX, data-center Ampere). Reaching here means minor < 6
            // (sm_86+ and sm_89+ are caught above). The sm_86 PTX would NOT load on sm_80
            // because a PTX .target is a *minimum* compute capability, so we ship a native
            // sm_80 PTX. If the driver is too old for its PTX ISA, fall back to sm_75, which
            // runs on sm_80 and up via the backward-compatible PTX JIT.
            _module = Arc::new(match load_ptx(PTX_80, "sm_80") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_80 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_75 PTX (update driver for native sm_80)", device_id);
                    load_ptx(PTX_75, "sm_75 (fallback)").map_err(|_| e)?
                }
            });
        } else if major > 7 || (major == 7 && minor >= 5) {
            // sm_75 (RTX 20 / Turing)
            _module = Arc::new(Module::from_ptx(PTX_75, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Error loading PTX. Make sure you have the updated driver for you devices");
                e
            })?);
        } else if major > 6 || (major == 6 && minor >= 1) {
            // sm_61 (GTX 10 / Pascal)
            _module = Arc::new(Module::from_ptx(PTX_61, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Error loading PTX. Make sure you have the updated driver for you devices");
                e
            })?);
        } else {
            return Err(format!(
                "CUDA compute {}.{} not supported. Keryx requires sm_61 (GTX 10xx) or newer.",
                major, minor
            )
            .into());
        }

        let stream_ping = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let stream_pong = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let mut heavy_hash_kernel = Kernel::new(Arc::downgrade(&_module), "heavy_hash")?;

        let mut chosen_workload = 0u32;
        if is_absolute {
            chosen_workload = 1;
        } else {
            let cur_workload = heavy_hash_kernel.get_workload();
            if chosen_workload == 0 || chosen_workload < cur_workload {
                chosen_workload = cur_workload;
            }
        }
        chosen_workload = (chosen_workload as f32 * workload) as u32;
        info!("GPU #{} Chosen workload: {}", device_id, chosen_workload);
        heavy_hash_kernel.set_workload(chosen_workload);

        // Two independent xoshiro rand-state buffers, each covering `chosen_workload` threads.
        // iter_jump_state() advances by long_jump() (2^192 steps) per element, guaranteeing
        // no overlap between the two sets.
        let (rand_state_ping, rand_state_pong) = match random {
            NonceGenEnum::Xoshiro => {
                info!("Using xoshiro for nonce-generation (double-buffered)");
                let mut seed = [1u64; 4];
                seed.try_fill(&mut rand::thread_rng())?;

                let all_states: Vec<u64> = Xoshiro256StarStar::new(&seed)
                    .iter_jump_state()
                    .take(2 * chosen_workload as usize)
                    .flatten()
                    .collect();

                let mid = 4 * chosen_workload as usize;
                let mut buf_ping = DeviceBuffer::<u64>::zeroed(mid)?;
                let mut buf_pong = DeviceBuffer::<u64>::zeroed(mid)?;
                buf_ping.copy_from(&all_states[..mid])?;
                buf_pong.copy_from(&all_states[mid..])?;
                info!("GPU #{} double-buffer xoshiro states initialised", device_id);
                (buf_ping, buf_pong)
            }
            NonceGenEnum::Lean => {
                info!("Using lean nonce-generation (double-buffered)");
                let s0 = rand::thread_rng().next_u64();
                let s1 = rand::thread_rng().next_u64();
                let mut buf_ping = DeviceBuffer::<u64>::zeroed(1)?;
                let mut buf_pong = DeviceBuffer::<u64>::zeroed(1)?;
                buf_ping.copy_from(&[s0])?;
                buf_pong.copy_from(&[s1])?;
                (buf_ping, buf_pong)
            }
        };

        let final_nonce_ping = vec![0u64; 1].as_slice().as_dbuf()?;
        let final_nonce_pong = vec![0u64; 1].as_slice().as_dbuf()?;

        Ok(Self {
            device_id,
            _context,
            _module,
            workload: chosen_workload as usize,
            heavy_hash_kernel,
            random,

            stream_ping,
            start_event_ping: Event::new(EventFlags::DEFAULT)?,
            stop_event_ping: Event::new(EventFlags::DEFAULT)?,
            rand_state_ping,
            final_nonce_ping,
            stream_ping_used: false,

            stream_pong,
            start_event_pong: Event::new(EventFlags::DEFAULT)?,
            stop_event_pong: Event::new(EventFlags::DEFAULT)?,
            rand_state_pong,
            final_nonce_pong,
            stream_pong_used: false,

            warmed_up: false,
        })
    }
}
