/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 *
 * kbase_kmod.c -- a pan_kmod backend for ARM's PROPRIETARY kbase kernel driver.
 *
 * WHY THIS EXISTS. pan_kmod ships two backends, panfrost_kmod (DRM panfrost) and panthor_kmod
 * (DRM panthor). Both talk to a DRM render node. A stock Samsung Exynos device has no /dev/dri
 * at all: its GPU is reached through ARM's kbase at /dev/mali0, which is world-openable
 * (crw-rw-rw-, SELinux gpu_device) and needs no root. So PanVK loads on such a device and then
 * enumerates zero physical devices -- correctly, because there is no DRM node to find.
 *
 * WHERE THE UAPI CAME FROM. Not from headers -- ARM's are not in this tree. Every ioctl number,
 * struct layout and semantic below was recovered by recording the VENDOR driver's own traffic
 * (an LD_PRELOAD spy around a real Vulkan workload) and diffing each payload before and after
 * the call, so input fields, output fields and padding separate themselves without anyone
 * deciding what a struct looks like. Three names in the public tables were WRONG for this DDK
 * and the traffic overruled them; they are marked below.
 *
 * WHY THE MODEL FITS. kbase assigns the GPU VA at allocation time and offers no userspace VA
 * management -- and neither does DRM panfrost, whose backend refuses anything but
 * PAN_KMOD_VM_FLAG_AUTO_VA, one VM per device and whole-BO mappings. The two kernel drivers
 * agree on the shape that matters, so this backend mirrors panfrost_kmod's structure closely.
 */

#include <errno.h>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <pthread.h>
#include <time.h>

#include "drm-uapi/drm.h"
#include "drm-uapi/panthor_drm.h"
#include "util/macros.h"
#include "util/os_mman.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/u_sync_provider.h"

#include "pan_kmod_backend.h"
#include "kbase_kmod.h"
#include "util/bitscan.h"
#include "util/list.h"

extern const struct pan_kmod_ops kbase_kmod_ops;

/* ---- kbase UAPI, recovered by measurement ------------------------------------------------- */

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check {
   uint16_t major;
   uint16_t minor;
};

struct kbase_ioctl_set_flags {
   uint32_t create_flags;
};

struct kbase_ioctl_get_gpuprops {
   uint64_t buffer;
   uint32_t size;
   uint32_t flags;
};

union kbase_ioctl_mem_alloc {
   struct {
      uint64_t va_pages;
      uint64_t commit_pages;
      uint64_t extension;
      uint64_t flags;
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
   } out;
};

struct kbase_ioctl_mem_free {
   uint64_t gpu_addr;
};

/* ioctl 38 appears in no table we had, and the vendor issues it exactly once with va_pages =
 * 0x100000. It reserves the EXECUTABLE VA zone, and that is not a reading of the number -- it is
 * an A/B: without the call, an allocation with the GPU_EX bit comes back as a SAME_VA cookie and
 * the kernel ORs 0x2000 into the returned flags; with it, the SAME request returns
 * 0x7f00001000, the exact address the vendor got, with the flags unchanged. */
struct kbase_ioctl_mem_exec_init {
   uint64_t va_pages;
};

/* nr 0/1/3/5 are as the public tables have them. nr 7 (MEM_FREE) never appeared in the captured
 * traffic -- the workload never freed anything -- so it was measured directly instead: it
 * SUCCEEDS on an exec-zone allocation and is refused, EINVAL, for every SAME_VA one, mapped or
 * not. That is not a bug in the number; it is the model. */
#define KBASE_IOCTL_VERSION_CHECK _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)
#define KBASE_IOCTL_CSF_VERSION_CHECK                                                             \
   _IOWR(KBASE_IOCTL_TYPE, 52, struct kbase_ioctl_version_check)
#define KBASE_IOCTL_SET_FLAGS     _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)
#define KBASE_IOCTL_GET_GPUPROPS  _IOW(KBASE_IOCTL_TYPE, 3, struct kbase_ioctl_get_gpuprops)
#define KBASE_IOCTL_MEM_ALLOC     _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)
#define KBASE_IOCTL_MEM_FREE      _IOW(KBASE_IOCTL_TYPE, 7, struct kbase_ioctl_mem_free)
#define KBASE_IOCTL_MEM_EXEC_INIT _IOW(KBASE_IOCTL_TYPE, 38, struct kbase_ioctl_mem_exec_init)

struct kbase_ioctl_mem_jit_init {
   uint64_t va_pages;
   uint8_t max_allocations;
   uint8_t trim_level;
   uint8_t group_id;
   uint8_t padding[5];
   uint64_t phys_pages;
};
#define KBASE_IOCTL_MEM_JIT_INIT _IOW(KBASE_IOCTL_TYPE, 14, struct kbase_ioctl_mem_jit_init)
/* 4 GB of VA for the kernel's own allocations on a CSF context (csfheap.c used this). */
#define KBASE_CSF_JIT_VA_PAGES (1ull << 20)

/* nr 22, a dma-buf (UMM) import: phandle points at the fd. The shape is kbase's public one; the
 * vendor could not be spied importing (its ICD loads in the sphal namespace, out of LD_PRELOAD's
 * reach), so what settles it is reading the rendered pixels back through the
 * dma-buf. Like MEM_ALLOC, a SAME_VA import returns a cookie, and the anchor mapping of that
 * cookie is the GPU VA. */
union kbase_ioctl_mem_import {
   struct {
      uint64_t flags;
      uint64_t phandle;
      uint32_t type;
      uint32_t padding;
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
      uint64_t va_pages;
   } out;
   /* The kbase of the Tab S12+ (MT6993, kernel 6.12) takes 40 bytes: its ioctl number is
    * 0xc0288016 (read off the vendor libGLES_mali.so), and the 24-byte one gets ENOTTY. The
    * vendor fills the same in/out fields and zeroes the rest. */
   uint8_t pad40[40];
};
#define KBASE_IOCTL_MEM_IMPORT                                                                    \
   _IOC(_IOC_READ | _IOC_WRITE, KBASE_IOCTL_TYPE, 22, 24)
#define KBASE_IOCTL_MEM_IMPORT_40                                                                 \
   _IOC(_IOC_READ | _IOC_WRITE, KBASE_IOCTL_TYPE, 22, 40)
#define BASE_MEM_IMPORT_TYPE_UMM 2
#define BASE_MEM_PROT_CPU_RD (1ull << 0)
#define BASE_MEM_PROT_CPU_WR (1ull << 1)
#define BASE_MEM_PROT_GPU_RD (1ull << 2)
#define BASE_MEM_PROT_GPU_WR (1ull << 3)
#define BASE_MEM_COHERENT_SYSTEM (1ull << 10)
#define BASE_MEM_SAME_VA         (1ull << 13)
#define BASE_MEM_IMPORT_SHARED   (1ull << 18)
/* What a SAME_VA import reports back instead of SAME_VA: gpu_va is a cookie to mmap. Measured,
 * the returned flags are 0x500f for a request of 0x200f. */
#define BASE_MEM_NEED_MMAP       (1ull << 14)

/* kbase reserves low mmap page offsets for special handles. */
#define BASE_MEM_MAP_TRACKING_HANDLE (3ull << 12)

/* Allocation flags, taken VERBATIM from what the vendor driver was observed to pass. The
 * individual bit meanings are not fully decoded, and inventing a combination would be a guess
 * where a measurement is available: 0x204f is what it used for ordinary read/write buffers and
 * 0x17 for the executable ones. */
#define KBASE_MEM_FLAGS_RW   (0x204full | BASE_MEM_COHERENT_LOCAL)
#define KBASE_MEM_FLAGS_EXEC 0x17ull

/* Bit 11, which the vendor also passes (its other captured shape is 0x380f). Without it the
 * GPU's own units are not coherent with one another on this memory: the tiler did not see what
 * an earlier tiler job left in the shared tiler context, so of several draws in one render pass
 * only the LAST rasterised -- measured, two direct draws, one survived; with this bit both do.
 * The same incoherence is what the CACHE_FLUSH jobs in the draw path were working around.
 * (CACHED_CPU, bit 12, the vendor's other difference, made every draw vanish: nothing here
 * syncs CPU caches.) */
#define BASE_MEM_COHERENT_LOCAL (1ull << 11)

/* Bit 12: the CPU mapping is cacheable. The GPU does not snoop CPU caches here, so the CPU side
 * needs explicit cache maintenance (PAN_KMOD_BO_FLAG_WB_MMAP). */
#define BASE_MEM_CACHED_CPU (1ull << 12)

/* PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT, the tiler heap, the indirect varying buffer and the tess
 * heap: 336 MB between them, all of it committed at vkCreateDevice when this flag was ignored
 * (measured, MemAvailable fell 331 MB; the vendor's device costs 3). GROW_ON_GPF commits
 * KBASE_GROW_INITIAL_PAGES and lets a GPU fault commit the rest KBASE_GROW_STEP_PAGES at a time,
 * which is what the DRM drivers' heap BOs do. */
#define BASE_MEM_GROW_ON_GPF     (1ull << 9)
#define KBASE_GROW_INITIAL_PAGES 256ull
#define KBASE_GROW_STEP_PAGES    512ull

/* What the vendor reserves. Also the only shape the exec zone accepts: adding GPU_WR to
 * KBASE_MEM_FLAGS_EXEC is refused with ENOMEM, and so is dropping GPU_RD -- the zone is for
 * shader code and nothing else. */
#define KBASE_EXEC_ZONE_PAGES 0x100000ull

/* CSF only: the r44p1 UAPI's bits 19 and 21. */
#define BASE_MEM_CSF_EVENT            (1ull << 19)
#define BASE_MEM_UNCACHED_GPU         (1ull << 21)

/* Memory group of an allocation, bits 25:22 of the flags. The vendor driver passes 0x180280f
 * for its read/write buffers and asks for group 6 in CS_TILER_HEAP_INIT; MediaTek's memory
 * group manager (mali_mgm_mt6989) gives that group the system level cache: DRAM reads answered
 * within 128 cycles went from under 1% to about 17% of responses with it (measured in Wild
 * Life), texture fetch stalls -10% and +2.9% fps in Deep Rock Galactic. A kernel whose memory
 * group manager has fewer groups refuses it (the Pixel r44p1 tree declares 4): then group 0 is
 * latched. KBASE_MEM_GROUP=<0-15> picks the group, for comparisons. */
#define BASEP_MEM_GROUP_ID_SHIFT 22
#define BASEP_MEM_GROUP_ID_MASK  (0xfull << BASEP_MEM_GROUP_ID_SHIFT)
#define KBASE_MEM_GROUP_VENDOR   6

static int kbase_mem_group_id = -1;

static uint64_t
kbase_mem_group(void)
{
   int group = p_atomic_read(&kbase_mem_group_id);
   if (group < 0) {
      const int64_t g = debug_get_num_option("KBASE_MEM_GROUP", KBASE_MEM_GROUP_VENDOR);
      p_atomic_cmpxchg(&kbase_mem_group_id, -1,
                       g >= 0 && g <= 15 ? (int)g : KBASE_MEM_GROUP_VENDOR);
      group = p_atomic_read(&kbase_mem_group_id);
   }
   return group;
}

/* An allocation in our group failed and the same one in group 0 worked: keep group 0. */
static void
kbase_mem_group_refused(void)
{
   if (p_atomic_xchg(&kbase_mem_group_id, 0) > 0)
      mesa_logw("kbase: the kernel refuses our memory group, using group 0");
}

/* ---- GPU property stream ------------------------------------------------------------------
 *
 * GET_GPUPROPS returns a packed run of (u32 header, value) pairs where header = (id << 2) | w,
 * w selecting a 1/2/4/8-byte value. It returns the BYTE COUNT, not zero, so only a negative
 * return is a failure -- treating non-zero as an error once made a successful 723-byte read
 * report a stale ENOENT.
 *
 * The ids below are anchored to values known independently for the reference part, rather than
 * copied from a table that proved wrong: PRODUCT_ID reads 0x7211 which is exactly the deviceID
 * the vendor's own Vulkan reports (0x72110000); SHADER_PRESENT reads 0x770077, whose popcount is
 * 12, matching a Mali-G76 MP12; and MMU_FEATURES reads 0x2830, i.e. 48 VA bits and 40 PA bits.
 * Three independent confirmations, so the surrounding ids can be trusted by position. */
enum kbase_gpuprop {
   KBASE_GPUPROP_PRODUCT_ID = 1,
   KBASE_GPUPROP_VERSION_STATUS = 2,
   KBASE_GPUPROP_MINOR_REVISION = 3,
   KBASE_GPUPROP_MAJOR_REVISION = 4,
   KBASE_GPUPROP_TEXTURE_FEATURES_0 = 9,
   KBASE_GPUPROP_TEXTURE_FEATURES_1 = 10,
   KBASE_GPUPROP_TEXTURE_FEATURES_2 = 11,
   KBASE_GPUPROP_L2_LOG2_LINE_SIZE = 13,
   KBASE_GPUPROP_L2_LOG2_CACHE_SIZE = 14,
   KBASE_GPUPROP_L2_NUM_L2_SLICES = 15,
   KBASE_GPUPROP_TILER_BIN_SIZE_BYTES = 16,
   KBASE_GPUPROP_TILER_MAX_ACTIVE_LEVELS = 17,
   KBASE_GPUPROP_MAX_THREADS = 18,
   KBASE_GPUPROP_MAX_WORKGROUP_SIZE = 19,
   KBASE_GPUPROP_MAX_BARRIER_SIZE = 20,
   KBASE_GPUPROP_MAX_REGISTERS = 21,
   KBASE_GPUPROP_MAX_TASK_QUEUE = 22,
   KBASE_GPUPROP_IMPL_TECH = 24,
   KBASE_GPUPROP_RAW_SHADER_PRESENT = 25,
   KBASE_GPUPROP_RAW_TILER_PRESENT = 26,
   KBASE_GPUPROP_RAW_L2_PRESENT = 27,
   KBASE_GPUPROP_RAW_STACK_PRESENT = 28,
   KBASE_GPUPROP_RAW_L2_FEATURES = 29,
   KBASE_GPUPROP_RAW_CORE_FEATURES = 30,
   KBASE_GPUPROP_RAW_MEM_FEATURES = 31,
   KBASE_GPUPROP_RAW_MMU_FEATURES = 32,
   KBASE_GPUPROP_RAW_AS_PRESENT = 33,
   KBASE_GPUPROP_RAW_JS_PRESENT = 34,
   KBASE_GPUPROP_RAW_JS_FEATURES_0 = 35,
   /* 51, NOT 84. 84 is TLS_ALLOC, which is also present in the stream, so reading it did not
    * fall back to the default -- it fed Mesa 0x300 where the real value is 0x809. Decoded, that
    * is a tiler with a 1-BYTE bin and 3 hierarchy levels instead of 512 bytes and 8, i.e. a
    * tiler with nowhere to put a primitive. Every job still completed without a fault; the
    * polygon list was simply empty. Confirmed against the bv_r32p1 UAPI header, which is the
    * DDK this device runs. */
   KBASE_GPUPROP_RAW_TILER_FEATURES = 51,
   KBASE_GPUPROP_RAW_THREAD_FEATURES = 59,
   KBASE_GPUPROP_TLS_ALLOC = 84,
   /* GPU_FEATURES, from kbase r43 on (absent from r38p1 and older). */
   KBASE_GPUPROP_RAW_GPU_FEATURES = 85,
};


struct kbase_sub;
struct kbase_csf_dev;

/* Submission state of a device, see "asynchronous submission". */
struct kbase_queue {
   pthread_mutex_t lock;
   pthread_cond_t cond;              /* atom numbers freed, submissions completed */
   struct kbase_sub *owner[256];     /* by atom number, NULL when free */
   bool is_fence[256];
   unsigned free_atoms;              /* of the numbers 1..255 */
   /* The newest atom in flight on each PAN_KMOD_KBASE_TRACK_*, 0 if none. */
   uint8_t track[PAN_KMOD_KBASE_NR_TRACKS];
   uint64_t next_seq;
   /* Submissions whose callback has not run yet. A fence-only submission (a present) completes on
    * a soft atom, which can report after a later submission's GPU atoms: completion is tracked per
    * submission, not as a high-water mark. */
   uint64_t open_seq[256];
   unsigned nr_open_seq;
   unsigned in_flight;               /* submissions */
   pthread_t thread;
   bool thread_started, stop, fence_warned;
};

struct kbase_kmod_dev {
   struct pan_kmod_dev base;
   struct kbase_kmod_vm *vm;
   uint32_t next_handle;
   int dma_heap_fd;
   struct kbase_queue queue;
   /* enum kbase_atom_layout, measured when the context is created. */
   int atom_layout;
   /* JS0_FEATURES: bit n = the fragment slot runs job type n (G72: NULL, WRITE_VALUE,
    * CACHE_FLUSH, FRAGMENT). */
   uint32_t js0_features;
   /* CSF: queue groups instead of job atoms (kbase UAPI 1.x). */
   bool csf;
   struct drm_panthor_csif_info csif;
   struct kbase_csf_dev *csfq;
};

static void kbase_queue_init(struct kbase_queue *q);
static int kbase_probe_atom_layout_fd(int fd);
static void kbase_queue_finish(struct kbase_kmod_dev *kdev);
static bool kbase_csf_dev_init(struct kbase_kmod_dev *kdev);
static void kbase_csf_dev_finish(struct kbase_kmod_dev *kdev);

struct kbase_kmod_bo {
   struct pan_kmod_bo base;
   uint64_t gpu_va;
   /* The mapping that FIXES gpu_va for an ordinary (SAME_VA) allocation, and whose lifetime is
    * the allocation's. NULL for an executable one, whose address comes from the kernel. */
   void *anchor;
   /* The anchor's length: the BO's, plus its guard page (kbase_guard_mode). */
   uint64_t anchor_size;
   int dmabuf_fd;
   void *cpu_mapping;
   void *gpu_mapping;
   bool imported;
   bool same_va;
   bool exec;
};

struct kbase_kmod_vm {
   struct pan_kmod_vm base;
};

static uint64_t
kbase_gpuprop_get(const uint8_t *buf, unsigned size, unsigned want, uint64_t dflt)
{
   for (unsigned off = 0; off + 4 <= size;) {
      uint32_t hdr;
      memcpy(&hdr, buf + off, 4);
      off += 4;

      const unsigned id = hdr >> 2, type = hdr & 3;
      const unsigned w = type == 0 ? 1 : type == 1 ? 2 : type == 2 ? 4 : 8;
      if (off + w > size)
         break;

      if (id == want) {
         uint64_t v = 0;
         memcpy(&v, buf + off, w);
         return v;
      }
      off += w;
   }
   return dflt;
}

static bool
kbase_kmod_query_props(int fd, struct pan_kmod_dev_props *props, uint32_t *js0_features)
{
   struct kbase_ioctl_get_gpuprops req = {0};

   /* size=0 asks how big the stream is; the return value IS the byte count. */
   const int need = pan_kmod_ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &req);
   if (need <= 0) {
      mesa_loge("kbase: GET_GPUPROPS size query failed (ret=%d, err=%d) -- not a kbase node?",
                need, errno);
      return false;
   }

   uint8_t *buf = calloc(1, need);
   if (!buf)
      return false;

   req.buffer = (uint64_t)(uintptr_t)buf;
   req.size = need;
   if (pan_kmod_ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &req) < 0) {
      mesa_loge("kbase: GET_GPUPROPS read failed (err=%d)", errno);
      free(buf);
      return false;
   }

#define P(name, dflt) kbase_gpuprop_get(buf, need, KBASE_GPUPROP_##name, (dflt))

   /* pan_arch() reads the product id out of bits 16..31, so gpu_id has to be the whole GPU_ID
    * register, not just the product. Rebuilt from its parts: 0x7211 becomes 0x72110000, whose
    * PAN_ARCH_MAJOR is 7 -- Bifrost v7, which is exactly what PanVK builds a jm/bifrost target
    * for. Getting this wrong is silent: a bare 0x7211 would decode as some Midgard part. */
   props->gpu_id = (P(PRODUCT_ID, 0) << 16) | ((P(MAJOR_REVISION, 0) & 0xf) << 12) |
                   ((P(MINOR_REVISION, 0) & 0xff) << 4) | (P(VERSION_STATUS, 0) & 0xf);
   props->gpu_variant = 0;

   props->shader_present = P(RAW_SHADER_PRESENT, 1);
   props->tiler_features = P(RAW_TILER_FEATURES, 0x809);
   props->mem_features = P(RAW_MEM_FEATURES, 0);
   props->mmu_features = P(RAW_MMU_FEATURES, 0x2830);
   props->l2_features = P(RAW_L2_FEATURES, 0);
   *js0_features = P(RAW_JS_FEATURES_0, 0);

   props->texture_features[0] = P(TEXTURE_FEATURES_0, 0);
   /* DEBUG ONLY: OR extra compressed-format bits into what the GPU reports, to ask the texture
    * unit directly whether it decodes a format its feature register leaves out. */
   const char *texfeat_or = getenv("PANVK_TEXFEAT0_OR");
   if (texfeat_or)
      props->texture_features[0] |= (uint32_t)strtoul(texfeat_or, NULL, 0);
   props->texture_features[1] = P(TEXTURE_FEATURES_1, 0);
   props->texture_features[2] = P(TEXTURE_FEATURES_2, 0);
   props->texture_features[3] = 0;

   props->max_threads_per_core = P(MAX_THREADS, 768);
   props->max_threads_per_wg = P(MAX_WORKGROUP_SIZE, props->max_threads_per_core);
   props->max_tasks_per_core = MAX2(P(MAX_TASK_QUEUE, 0), 1);
   props->num_registers_per_core = P(MAX_REGISTERS, 0x6000);

   /* THREAD_TLS_ALLOC: how many thread-storage slots the hardware indexes per core, which is
    * not the thread count. The Mali-G52 in the Galaxy A31 reports 1024 slots for 768 threads;
    * sized for 768, the stack of a thread in a slot past it ran off the end of the allocation,
    * and any spilling kernel (the batch patch job spills 456 bytes) faulted the whole atom at
    * random, depending on which slots the threads landed in. kbase exposes it as TLS_ALLOC (84);
    * pan_kmod's contract falls back to max_threads_per_core when it is 0. */
   props->max_tls_instance_per_core = P(TLS_ALLOC, 0) ?: props->max_threads_per_core;

   /* CSF parts (v10 and later) as panthor reports them: the variant is the low byte of
    * CORE_FEATURES (the G720 table entry is variant 4, and CORE_FEATURES reads 0x50000004), the
    * thread limits come from THREAD_FEATURES, there is no THREAD_TLS_ALLOC, and AFBC is no longer
    * optional. kbase's own MAX_REGISTERS/MAX_TASK_QUEUE decode that register differently on these
    * parts (1024 and 4 where panthor's decode gives 65536 and 16). */
   if (PAN_ARCH_MAJOR(props->gpu_id) >= 10) {
      const uint32_t thread_features = P(RAW_THREAD_FEATURES, 0);
      props->gpu_variant = P(RAW_CORE_FEATURES, 0) & 0xff;
      if (thread_features) {
         props->num_registers_per_core = thread_features & 0x3fffff;
         props->max_tasks_per_core = MAX2(thread_features >> 24, 1);
      }
      props->max_tls_instance_per_core = props->max_threads_per_core;
      props->afbc_features = 0;

      /* GPU_FEATURES bit 2 (RAY_INTERSECTION) is what tells an Immortalis part from the plain
       * Mali one in pan_get_model. Left at 0, the Tab S11's Immortalis-G925 MC12 called itself
       * a G725. Kernels before r43 do not export the register; there Arm's own configurations
       * decide, since G715, G720 and G725 only come with 10 or more cores as Immortalis. */
      props->gpu_features = P(RAW_GPU_FEATURES, UINT64_MAX);
      if (props->gpu_features == UINT64_MAX)
         props->gpu_features =
            __builtin_popcountll(props->shader_present) >= 10 ? (1ull << 2) : 0;
   }

   /* Matching panfrost_kmod: Mali MMUs take 4K and 2M pages. Coherency is left off
    * because nothing observed says this SoC is IO-coherent, and claiming it would
    * silently drop cache maintenance. */
   props->pgsize_bitmap = PAN_PGSIZE_4K | PAN_PGSIZE_2M;
   props->is_io_coherent = false;

   /* CPU-cached mappings, which the vendor driver also hands out (its 0x380f allocations carry
    * CACHED_CPU). Not coherent with the GPU -- RAW_COHERENCY_MODE reads COHERENCY_NONE on the
    * S10e -- so they back PanVK's HOST_CACHED, non-HOST_COHERENT memory type, whose flushes and
    * invalidates PanVK does from userspace (DC CVAC / DC CIVAC). Without such a type every CPU
    * read of GPU output went through an uncached mapping. */
   props->supported_bo_flags |= PAN_KMOD_BO_FLAG_WB_MMAP;

   /* Without this the mask is zero, every queue priority is refused, and vkCreateDevice
    * fails with VK_ERROR_NOT_PERMITTED -- which reads as a permissions problem rather
    * than a field nobody filled in. MEDIUM only: the JM backend does not hook up the
    * other priorities anyway (panvk filters them itself for arch < 10), and claiming a
    * priority kbase has not been asked for would be inventing a capability. */
   props->allowed_group_priorities_mask = PAN_KMOD_GROUP_ALLOW_PRIORITY_MEDIUM;

#if defined(__aarch64__)
   /* The GPU's SYSTEM_TIMESTAMP (what a WRITE_VALUE job stores) is the SoC's system counter, the
    * one the CPU reads as CNTVCT_EL0, so its frequency is CNTFRQ_EL0 and the current value can
    * be read without the kernel. */
   uint64_t freq;
   __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
   if (freq) {
      props->gpu_can_query_timestamp = true;
      props->timestamp_device_coherent = true;
      props->timestamp_frequency = freq;
      props->timestamp_cycles_to_ns_factor = 1000000000.0 / freq;
   }
#endif

#undef P

   free(buf);
   return true;
}

static struct pan_kmod_dev *
kbase_kmod_dev_create(int fd, uint32_t flags, const struct pan_kmod_allocator *allocator)
{
   struct kbase_kmod_dev *kbase_dev = pan_kmod_alloc(allocator, sizeof(*kbase_dev));
   if (!kbase_dev) {
      mesa_loge("kbase: out of memory");
      return NULL;
   }
   memset(kbase_dev, 0, sizeof(*kbase_dev));
   kbase_dev->dma_heap_fd = -1;

   /* The handshake, in the order the vendor driver performs it. VERSION_CHECK VALIDATES rather
    * than reports: the payload comes back unchanged, so a mismatched pair is refused by the
    * kernel rather than negotiated down. 11.31 is what the r32p1 DDK asks for on the reference
    * device.
    *
    * IT IS PER-OPEN, NOT PER-FD. PanVK builds its logical device by dup()ing the physical
    * device's fd, so this function runs a second time on a context that already exists, and
    * kbase refuses the handshake there with EPERM. Failing on that would mean no logical device
    * could ever be created. So the handshake is attempted and, if refused, we simply do not
    * repeat SET_FLAGS: the real gate is whether the node answers GET_GPUPROPS below, which a
    * working kbase context does however it was set up and a non-kbase node never does. */
   /* A CSF kbase (UAPI 1.x) moved the handshake to number 52; its number 0 is a dummy that
    * refuses, and once refused no handshake succeeds on that file. So 52 goes first. A JM kbase
    * refuses 52 with EPERM before the handshake without changing anything, or, where 52 is its
    * handshake too, answers with its own major (11), which the context then simply runs at. */
   struct kbase_ioctl_version_check vc = {.major = 1, .minor = 21};
   bool fresh_context = pan_kmod_ioctl(fd, KBASE_IOCTL_CSF_VERSION_CHECK, &vc) == 0;
   if (!fresh_context) {
      vc = (struct kbase_ioctl_version_check){.major = 11, .minor = 31};
      fresh_context = pan_kmod_ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &vc) == 0;
   }

   if (fresh_context) {
      struct kbase_ioctl_set_flags sf = {.create_flags = 0};
      if (pan_kmod_ioctl(fd, KBASE_IOCTL_SET_FLAGS, &sf)) {
         mesa_loge("kbase: SET_FLAGS failed (err=%d)", errno);
         goto err_free_dev;
      }

      /* THE ORDER IS SET_FLAGS FIRST, THEN THE TRACKING PAGE, and it is not cosmetic. Mapping
       * it first is refused with EPERM and leaves the context half initialised, after which
       * EVERY MEM_ALLOC fails with EINVAL -- including a single page with the exact flags the
       * vendor uses successfully. That is what the failure looks like, so it invites blaming the
       * allocation flags rather than the setup. This is the vendor's own order, read off the
       * captured trace. */
      if (mmap(NULL, 4096, PROT_NONE, MAP_SHARED, fd, BASE_MEM_MAP_TRACKING_HANDLE) ==
          MAP_FAILED) {
         mesa_loge("kbase: failed to map the tracking page (err=%d)", errno);
         goto err_free_dev;
      }

      /* Reserve the executable VA zone. Without this, an allocation asking for GPU_EX is
       * silently downgraded to SAME_VA and returns a cookie instead of an address -- silently,
       * because nothing fails: the allocation succeeds and only the address is wrong. Not fatal
       * if the kernel refuses it; shader BOs then just live in the ordinary zone. */
      struct kbase_ioctl_mem_exec_init ei = {.va_pages = KBASE_EXEC_ZONE_PAGES};

      /* CSF: the kernel's own allocations for this context (tiler heap contexts and chunks)
       * come from the CUSTOM_VA zone, which a 64-bit context only has once JIT is initialised:
       * kbase carves it out of SAME_VA then. Without it CS_TILER_HEAP_INIT is ENOMEM for any
       * parameters (measured). It must come before the first allocation. */
      if (vc.major == 1) {
         struct kbase_ioctl_mem_jit_init jit = {
            .va_pages = KBASE_CSF_JIT_VA_PAGES,
            .max_allocations = 255,
            .phys_pages = KBASE_CSF_JIT_VA_PAGES,
         };
         if (pan_kmod_ioctl(fd, KBASE_IOCTL_MEM_JIT_INIT, &jit))
            mesa_logw("kbase: MEM_JIT_INIT failed (err=%d); no tiler heap", errno);
      }
      if (pan_kmod_ioctl(fd, KBASE_IOCTL_MEM_EXEC_INIT, &ei))
         mesa_logw("kbase: MEM_EXEC_INIT failed (err=%d); executable BOs will be SAME_VA", errno);
   }

   const struct pan_kmod_driver drv_info = {
      .version = {.major = vc.major, .minor = vc.minor},
   };
   if (pan_kmod_dev_init(&kbase_dev->base, fd, flags, &drv_info, allocator, &kbase_kmod_ops,
                         pan_kmod_kbase_sync_provider())) {
      mesa_loge("kbase: out of memory");
      goto err_free_dev;
   }

   if (!kbase_kmod_query_props(fd, &kbase_dev->base.props, &kbase_dev->js0_features)) {
      pan_kmod_dev_cleanup(&kbase_dev->base);
      goto err_free_dev;
   }

   kbase_dev->next_handle = 1;
   kbase_queue_init(&kbase_dev->queue);

   const char *dma_heap = getenv("PANVK_KBASE_DMA_HEAP");
   if (!dma_heap || !dma_heap[0])
      dma_heap = "/dev/dma_heap/system";
   kbase_dev->dma_heap_fd = open(dma_heap, O_RDONLY | O_CLOEXEC);
   if (kbase_dev->dma_heap_fd < 0)
      mesa_logd("kbase: dma-heap unavailable at %s: %s", dma_heap, strerror(errno));

   /* A dup()ed fd of an existing context answers neither handshake, so CSF is decided by the
    * GPU: v10 and later are CSF parts. */
   kbase_dev->csf = PAN_ARCH_MAJOR(kbase_dev->base.props.gpu_id) >= 10;
   if (kbase_dev->csf) {
      if (!kbase_csf_dev_init(kbase_dev)) {
         kbase_queue_finish(kbase_dev);
         pan_kmod_dev_cleanup(&kbase_dev->base);
         goto err_free_dev;
      }
   } else {
      kbase_dev->atom_layout = kbase_probe_atom_layout_fd(fd);
   }
   return &kbase_dev->base;

err_free_dev:
   if (kbase_dev->dma_heap_fd >= 0)
      close(kbase_dev->dma_heap_fd);
   pan_kmod_free(allocator, kbase_dev);
   return NULL;
}

static void
kbase_kmod_dev_destroy(struct pan_kmod_dev *dev)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   kbase_csf_dev_finish(kbase_dev);
   kbase_queue_finish(kbase_dev);
   if (kbase_dev->dma_heap_fd >= 0)
      close(kbase_dev->dma_heap_fd);
   pan_kmod_dev_cleanup(dev);
   pan_kmod_free(dev->allocator, kbase_dev);
}

static struct pan_kmod_va_range
kbase_kmod_dev_query_user_va_range(const struct pan_kmod_dev *dev)
{
   /* kbase hands out the VA itself, so this only has to describe the space those addresses live
    * in. MMU_FEATURES carries the VA width in its low byte (0x30 = 48 bits on the reference
    * part). The first 4 GB are left alone: kbase uses low addresses for its own special
    * mappings, and the tracking page sits at 3 << 12. */
   const unsigned va_bits = MMU_FEATURES_VA_BITS(dev->props.mmu_features);

   return (struct pan_kmod_va_range){
      .start = 4ull * 1024 * 1024 * 1024,
      .size = (1ull << va_bits) - (4ull * 1024 * 1024 * 1024),
   };
}

/* ---- buffer objects ------------------------------------------------------------------------
 *
 * THE MEMORY MODEL IS NOT WHAT A FIRST READING OF THE TRAFFIC SUGGESTS, and the first version
 * of this backend got it wrong in a way nothing could catch until something was submitted.
 *
 * MEM_ALLOC's out.gpu_va is NOT always an address. Three vendor records:
 *
 *     flags 0x204f -> 0x41000     flags 0x380f -> 0x41000     (185 pages, and 1 page)
 *     flags 0x17   -> 0x0000007f00001000
 *
 * A value that repeats across allocations of different sizes is not an address. 0x41000 is a
 * COOKIE. The measurement that settles it is MEM_QUERY: the kernel does
 * not recognise 0x41000 as a region, and DOES recognise the address the subsequent mmap
 * returned. So for an ordinary allocation:
 *
 *     the GPU VA is the address mmap GIVES BACK, not the value MEM_ALLOC returned.
 *
 * kbase forces BASE_MEM_SAME_VA on for 64-bit clients -- it ORs 0x2000 into the flags it returns
 * even when we did not ask for it -- and a SAME_VA region's GPU address is its CPU address. The
 * one exception is the odd record above, and it is the only one with the GPU_EX bit:
 * executable allocations come from a separate zone with real addresses. That zone does not exist
 * until ioctl 38 reserves it, which is why the vendor calls it exactly once at startup and why,
 * without it, even flags 0x17 comes back as a cookie. Measured as an A/B.
 *
 * Consequences, each of them measured rather than reasoned:
 *
 *   - An ANCHOR mapping is taken at allocation time. It is what fixes the GPU VA, and it must
 *     outlive every mapping PanVK makes, because for a SAME_VA region munmap IS the free --
 *     MEM_FREE refuses it (EINVAL) in every form. Without the anchor, vkUnmapMemory would
 *     destroy the allocation.
 *   - A SAME_VA region can be mapped AGAIN at offset = its own address; the second mapping lands
 *     at a different CPU address and reads the same memory. That is what lets
 *     bo_get_mmap_offset keep working for PanVK on top of the anchor.
 *   - An EXEC region is the mirror image: MEM_FREE works on it, a second mmap does not (ENOMEM),
 *     and GPU_WR alongside GPU_EX is refused outright, so the zone is for shader code only.
 *
 * pan_kmod still wants a u32 handle to key its BO table on, so one is synthesised per device.
 */

/* The last allocations and frees, for one question a GPU page fault raises: was the address
 * ours, and did we free it while the GPU could still read it? The kernel reports the faulting
 * address with the group error (kbase_csf_ctx_handle_fault puts it in the sideband), and the
 * event thread looks it up here (kbase_bo_history_explain). Lock-free on purpose: a torn record
 * costs one diagnostic, a lock would cost every allocation. */
#define KBASE_BO_HISTORY 32768

struct kbase_bo_record {
   uint64_t va, size, t_ns;
   uint32_t flags;
};

static struct kbase_bo_record kbase_bo_allocs[KBASE_BO_HISTORY];
static struct kbase_bo_record kbase_bo_frees[KBASE_BO_HISTORY];
static uint32_t kbase_bo_alloc_next, kbase_bo_free_next;

static uint64_t kbase_now_ns(void);

static void
kbase_bo_history_add(struct kbase_bo_record *ring, uint32_t *next, uint64_t va, uint64_t size,
                     uint32_t flags)
{
   const uint32_t i = p_atomic_inc_return(next) % KBASE_BO_HISTORY;
   ring[i] = (struct kbase_bo_record){
      .va = va, .size = size, .t_ns = kbase_now_ns(), .flags = flags};
}

/* Writes into buf what the history knows about addr: the newest free and the newest
 * allocation covering it, and how far back each ring reaches. */
static void
kbase_bo_history_explain(uint64_t addr, char *buf, size_t len)
{
   const uint64_t now = kbase_now_ns();
   const struct kbase_bo_record *freed = NULL, *alloced = NULL;
   uint64_t oldest_alloc = now, oldest_free = now;
   for (unsigned i = 0; i < KBASE_BO_HISTORY; i++) {
      const struct kbase_bo_record *f = &kbase_bo_frees[i];
      if (f->size)
         oldest_free = MIN2(oldest_free, f->t_ns);
      if (f->size && addr >= f->va && addr - f->va < f->size &&
          (!freed || f->t_ns > freed->t_ns))
         freed = f;
      const struct kbase_bo_record *a = &kbase_bo_allocs[i];
      if (a->size)
         oldest_alloc = MIN2(oldest_alloc, a->t_ns);
      if (a->size && addr >= a->va && addr - a->va < a->size &&
          (!alloced || a->t_ns > alloced->t_ns))
         alloced = a;
   }
   int n = snprintf(buf, len, "address 0x%" PRIx64 ":", addr);
   if (alloced && n > 0 && (size_t)n < len)
      n += snprintf(buf + n, len - n,
                    " allocated %" PRIu64 " ms ago in BO 0x%" PRIx64 "+0x%" PRIx64
                    " (flags 0x%x);",
                    (now - alloced->t_ns) / 1000000, alloced->va, alloced->size, alloced->flags);
   if (freed && n > 0 && (size_t)n < len)
      n += snprintf(buf + n, len - n,
                    " freed %" PRIu64 " ms ago from BO 0x%" PRIx64 "+0x%" PRIx64 " (flags 0x%x)%s;",
                    (now - freed->t_ns) / 1000000, freed->va, freed->size, freed->flags,
                    alloced && alloced->t_ns > freed->t_ns ? ", since reallocated" : "");
   if (!alloced && !freed && n > 0 && (size_t)n < len)
      n += snprintf(buf + n, len - n, " in none of the last %u allocations or frees;",
                    KBASE_BO_HISTORY);
   if (n > 0 && (size_t)n < len)
      snprintf(buf + n, len - n,
               " the history reaches back %" PRIu64 " ms for allocations, %" PRIu64
               " ms for frees",
               (now - oldest_alloc) / 1000000, (now - oldest_free) / 1000000);
}

/* Every live mapping we made at its own GPU address (BO anchors, imports, repeated mappings),
 * and the executable BOs, so a fault can be matched against what is alive NOW, which the history
 * above cannot do for a BO older than its window. And, with KBASE_CHECK_ANCHORS=1, so the event
 * thread can check that each anchor is still mapped (kbase_live_check): the kernel frees a
 * SAME_VA region as soon as its anchor goes away, whoever unmaps it or maps over it
 * (kbase_cpu_vm_close), and nothing tells us; the GPU then finds no region there ("Memory is
 * not mapped on the GPU"). The process is shared with Wine and box64, which manage the address
 * space themselves. */
struct kbase_live_map {
   uint64_t va, size, t_ns;
   uintptr_t caller; /* a return address into whoever asked for it */
   uint32_t flags;   /* PAN_KMOD_BO_FLAG_* */
   bool anchored;    /* CPU-mapped at va, and freed when that mapping goes */
   bool guard;       /* followed by a reserved, unbacked page (KBASE_GUARD_PAGES=detect) */
   bool reported;    /* found gone once already */
   char label[72];   /* what PanVK called it (bo_set_label) */
};

/* In kbase_live_map.flags: a repeated mapping (pan_kmod_kbase_bo_map_repeated) of a BO. Bit 31
 * is PAN_KMOD_BO_FLAG_KBASE_SYNC. */
#define KBASE_LIVE_REPEATED (1u << 30)

static pthread_mutex_t kbase_live_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kbase_live_map *kbase_live;
static uint32_t kbase_live_count, kbase_live_cap;

static void
kbase_live_add(uint64_t va, uint64_t size, uint32_t flags, bool anchored, bool guard,
               uintptr_t caller)
{
   pthread_mutex_lock(&kbase_live_lock);
   if (kbase_live_count == kbase_live_cap) {
      const uint32_t cap = kbase_live_cap ? kbase_live_cap * 2 : 1024;
      struct kbase_live_map *l = realloc(kbase_live, cap * sizeof(*l));
      if (l) {
         kbase_live = l;
         kbase_live_cap = cap;
      }
   }
   if (kbase_live_count < kbase_live_cap)
      kbase_live[kbase_live_count++] = (struct kbase_live_map){
         .va = va, .size = size, .t_ns = kbase_now_ns(), .caller = caller, .flags = flags,
         .anchored = anchored, .guard = guard};
   pthread_mutex_unlock(&kbase_live_lock);
}

static void
kbase_live_remove(uint64_t va)
{
   pthread_mutex_lock(&kbase_live_lock);
   for (uint32_t i = kbase_live_count; i-- > 0;) {
      if (kbase_live[i].va == va) {
         kbase_live[i] = kbase_live[--kbase_live_count];
         break;
      }
   }
   pthread_mutex_unlock(&kbase_live_lock);
}

/* What PanVK calls the BO at va, for the fault reports. Newest entries are at the end. */
static void
kbase_live_set_label(uint64_t va, const char *label)
{
   pthread_mutex_lock(&kbase_live_lock);
   for (uint32_t i = kbase_live_count; i-- > 0;) {
      if (kbase_live[i].va == va) {
         snprintf(kbase_live[i].label, sizeof(kbase_live[i].label), "%s", label ? label : "");
         break;
      }
   }
   pthread_mutex_unlock(&kbase_live_lock);
}

/* "libvulkan_panfrost.so+0x1234", for llvm-addr2line on the unstripped build. */
static void
kbase_format_code_addr(uintptr_t pc, char *buf, size_t len)
{
   Dl_info info;
   if (pc && dladdr((void *)pc, &info) && info.dli_fname) {
      const char *base = strrchr(info.dli_fname, '/');
      snprintf(buf, len, "%s+0x%" PRIxPTR, base ? base + 1 : info.dli_fname,
               pc - (uintptr_t)info.dli_fbase);
   } else {
      snprintf(buf, len, "0x%" PRIxPTR, pc);
   }
}

/* Appends to out what is mapped now, summed by the label PanVK gave it and, for what has none, by
 * the code that asked for it: where a process's GPU memory is beyond the heap the application
 * allocates from (PANVK_IMAGE_STATS_FILE). */
void
pan_kmod_kbase_mem_report(FILE *out)
{
   struct {
      char label[sizeof(((struct kbase_live_map *)0)->label)];
      uintptr_t caller;
      uint64_t size;
      uint32_t count;
   } sums[64];
   unsigned n = 0;
   uint64_t total = 0;

   pthread_mutex_lock(&kbase_live_lock);
   for (uint32_t i = 0; i < kbase_live_count; i++) {
      const struct kbase_live_map *m = &kbase_live[i];

      /* A repeated mapping shows a BO a second time. */
      if (m->flags & KBASE_LIVE_REPEATED)
         continue;

      unsigned s = 0;
      while (s < n && (strcmp(sums[s].label, m->label) ||
                       (!m->label[0] && sums[s].caller != m->caller)))
         s++;
      if (s == n && n < ARRAY_SIZE(sums)) {
         memcpy(sums[n].label, m->label, sizeof(sums[n].label));
         sums[n].caller = m->caller;
         sums[n].size = 0;
         sums[n].count = 0;
         n++;
      } else if (s == n) {
         /* Past the table: its last entry takes the rest. */
         s = n - 1;
      }
      sums[s].size += m->size;
      sums[s].count++;
      total += m->size;
   }
   pthread_mutex_unlock(&kbase_live_lock);

   fprintf(out, "mapped %" PRIu64 " MiB\n", total >> 20);
   for (unsigned s = 0; s < n; s++) {
      char site[96];

      if (!(sums[s].size >> 20))
         continue;
      kbase_format_code_addr(sums[s].caller, site, sizeof(site));
      fprintf(out, "  %s: %u buffers, %" PRIu64 " MiB\n", sums[s].label[0] ? sums[s].label : site,
              sums[s].count, sums[s].size >> 20);
   }
}

/* Writes into buf what is alive at addr now: the live mapping covering it, or else the nearest
 * one on each side. For an address in the guard page after a BO, tail gets that BO's last 128
 * bytes as 64-bit words (empty otherwise): what was being read past. */
static void
kbase_live_explain(uint64_t addr, char *buf, size_t len, char *tail, size_t tail_len)
{
   const uint64_t now = kbase_now_ns();
   struct kbase_live_map hit = {0}, below = {0}, above = {0};
   bool has_hit = false, has_below = false, has_above = false;
   uint64_t words[16];
   unsigned nr_words = 0;
   pthread_mutex_lock(&kbase_live_lock);
   const uint32_t count = kbase_live_count;
   for (uint32_t i = 0; i < kbase_live_count; i++) {
      const struct kbase_live_map *m = &kbase_live[i];
      if (addr >= m->va && addr - m->va < m->size) {
         hit = *m;
         has_hit = true;
      } else if (m->va < addr) {
         if (!has_below || m->va > below.va) {
            below = *m;
            has_below = true;
         }
      } else if (!has_above || m->va < above.va) {
         above = *m;
         has_above = true;
      }
   }
   const bool in_guard = !has_hit && has_below && below.guard &&
                         addr - (below.va + below.size) < 4096;
   if (in_guard && below.anchored) {
      /* Still under the lock, so the BO cannot be released while it is read. */
      nr_words = MIN2(below.size, sizeof(words)) / 8;
      memcpy(words, (const uint8_t *)(uintptr_t)(below.va + below.size - nr_words * 8),
             nr_words * 8);
   }
   pthread_mutex_unlock(&kbase_live_lock);

   if (tail && tail_len) {
      tail[0] = '\0';
      int t = 0;
      for (unsigned i = 0; i < nr_words && t >= 0 && (size_t)t < tail_len; i++)
         t += snprintf(tail + t, tail_len - t, "%s%016" PRIx64, i ? " " : "", words[i]);
   }

   char site[96];
   if (has_hit) {
      kbase_format_code_addr(hit.caller, site, sizeof(site));
      snprintf(buf, len,
               "inside LIVE mapping 0x%" PRIx64 "+0x%" PRIx64 " \"%s\" (flags 0x%x, made %" PRIu64
               " ms ago from %s)%s",
               hit.va, hit.size, hit.label, hit.flags, (now - hit.t_ns) / 1000000, site,
               hit.anchored ? ", which we never released: something else removed it" : "");
      return;
   }
   if (in_guard) {
      kbase_format_code_addr(below.caller, site, sizeof(site));
      snprintf(buf, len,
               "in the GUARD page after live BO 0x%" PRIx64 "+0x%" PRIx64 " \"%s\" (flags 0x%x, made %"
               PRIu64 " ms ago from %s): an access 0x%" PRIx64 " past that BO's end",
               below.va, below.size, below.label, below.flags, (now - below.t_ns) / 1000000, site,
               addr - (below.va + below.size));
      return;
   }
   int n = snprintf(buf, len, "in none of our %u live mappings", count);
   if (has_below && n > 0 && (size_t)n < len) {
      kbase_format_code_addr(below.caller, site, sizeof(site));
      n += snprintf(buf + n, len - n,
                    "; below it 0x%" PRIx64 "+0x%" PRIx64 " (flags 0x%x, from %s), which ends 0x%" PRIx64
                    " before",
                    below.va, below.size, below.flags, site, addr - (below.va + below.size));
   }
   if (has_above && n > 0 && (size_t)n < len) {
      kbase_format_code_addr(above.caller, site, sizeof(site));
      snprintf(buf + n, len - n,
               "; above it 0x%" PRIx64 "+0x%" PRIx64 " (flags 0x%x, from %s), 0x%" PRIx64
               " after",
               above.va, above.size, above.flags, site, above.va - addr);
   }
}

/* One extra page after every ordinary BO, never used by PanVK. Something in Deep Rock
 * Galactic's render passes has the GPU read a little past the end of a command buffer
 * allocation that sits at the very end of its 64 KiB slab. On panthor BOs sit back to back in
 * the GPU address space and such a read lands in the next BO unnoticed; here the next page is
 * usually a CPU view (bo_mmap) the GPU cannot see, and the read kills the queue group.
 * KBASE_GUARD_PAGES picks what the page is:
 *   absorb (the default): committed, so the read finds zeros, as harmless as on panthor;
 *   detect (or 1): reserved but never backed, so the read faults inside this BO's own region
 *     ("Memory is not growable") and the fault report names the BO and dumps its end;
 *   off (or 0): no page. */
enum kbase_guard {
   KBASE_GUARD_OFF,
   KBASE_GUARD_ABSORB,
   KBASE_GUARD_DETECT,
};

static enum kbase_guard
kbase_guard_mode(void)
{
   static int mode = -1;
   if (mode < 0) {
      const char *s = debug_get_option("KBASE_GUARD_PAGES", "absorb");
      mode = !strcmp(s, "detect") || !strcmp(s, "1")  ? KBASE_GUARD_DETECT
             : !strcmp(s, "off") || !strcmp(s, "0") ? KBASE_GUARD_OFF
                                                      : KBASE_GUARD_ABSORB;
   }
   return mode;
}

static struct pan_kmod_bo *
kbase_kmod_bo_import_dmabuf(struct pan_kmod_dev *dev, struct pan_kmod_vm *exclusive_vm,
                            int fd, uint64_t size, uint32_t flags)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);
   struct kbase_kmod_bo *bo = pan_kmod_dev_alloc(dev, sizeof(*bo));
   if (!bo)
      return NULL;

   bo->dmabuf_fd = fcntl(fd, F_DUPFD_CLOEXEC, 3);
   bo->cpu_mapping = NULL;
   bo->gpu_mapping = NULL;
   if (bo->dmabuf_fd < 0) {
      mesa_loge("kbase: failed to duplicate dma-buf: %s", strerror(errno));
      goto err_free_bo;
   }

   uint64_t import_flags = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR |
                           BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR |
                           BASE_MEM_IMPORT_SHARED | BASE_MEM_COHERENT_SYSTEM;
   if (flags & PAN_KMOD_BO_FLAG_GPU_UNCACHED)
      import_flags |= BASE_MEM_UNCACHED_GPU;
   if (flags & PAN_KMOD_BO_FLAG_WB_MMAP)
      import_flags |= BASE_MEM_CACHED_CPU;

   int import_fd = bo->dmabuf_fd;
   union kbase_ioctl_mem_import req = {
      .in = {
         .flags = import_flags,
         .phandle = (uintptr_t)&import_fd,
         .type = BASE_MEM_IMPORT_TYPE_UMM,
      },
   };
   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_IMPORT, &req)) {
      mesa_loge("kbase: MEM_IMPORT of dma-buf failed: %s", strerror(errno));
      goto err_close_dmabuf;
   }

   const uint64_t bo_size = req.out.va_pages * 4096;
   bo->same_va = (req.out.flags & (BASE_MEM_SAME_VA | BASE_MEM_NEED_MMAP)) != 0;
   if (!bo->same_va || bo_size < size) {
      mesa_loge("kbase: MEM_IMPORT returned flags 0x%" PRIx64 " and size 0x%" PRIx64
                " for request 0x%" PRIx64,
                req.out.flags, bo_size, size);
      goto err_free_region;
   }

   bo->gpu_mapping = os_mmap(NULL, bo_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                             (off_t)req.out.gpu_va);
   if (bo->gpu_mapping == MAP_FAILED) {
      const int err = errno;
      bo->gpu_mapping = NULL;
      mesa_loge("kbase: mmap of imported dma-buf failed: %s", strerror(err));
      errno = err;
      goto err_free_region;
   }

   if (!(flags & PAN_KMOD_BO_FLAG_NO_MMAP)) {
      bo->cpu_mapping = os_mmap(NULL, bo_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                                bo->dmabuf_fd, 0);
      if (bo->cpu_mapping == MAP_FAILED) {
         const int err = errno;
         bo->cpu_mapping = NULL;
         mesa_loge("kbase: CPU mmap of imported dma-buf failed: %s", strerror(err));
         errno = err;
         goto err_free_region;
      }
   }

   bo->imported = true;
   bo->gpu_va = (uint64_t)(uintptr_t)bo->gpu_mapping;
   pan_kmod_bo_init(&bo->base, dev, exclusive_vm, bo_size, flags,
                    p_atomic_inc_return(&kbase_dev->next_handle));
   kbase_bo_history_add(kbase_bo_allocs, &kbase_bo_alloc_next, bo->gpu_va, bo_size, flags);
   kbase_live_add(bo->gpu_va, bo_size, flags, bo->same_va, false,
                  (uintptr_t)__builtin_return_address(0));
   return &bo->base;

err_free_region: {
   if (bo->gpu_mapping) {
      if (os_munmap(bo->gpu_mapping, bo_size))
         mesa_loge("kbase: unmapping failed dma-buf import: %s", strerror(errno));
   } else if (bo->same_va && bo_size) {
      void *mapping =
         os_mmap(NULL, bo_size, PROT_NONE, MAP_SHARED, dev->fd, (off_t)req.out.gpu_va);
      if (mapping != MAP_FAILED) {
         if (os_munmap(mapping, bo_size))
            mesa_loge("kbase: releasing failed SAME_VA import: %s", strerror(errno));
      } else {
         mesa_loge("kbase: could not release failed SAME_VA import: %s", strerror(errno));
      }
   }
   if (!bo->same_va) {
      struct kbase_ioctl_mem_free free_req = {.gpu_addr = req.out.gpu_va};
      if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_FREE, &free_req))
         mesa_logd("kbase: MEM_FREE after failed dma-buf import: %s", strerror(errno));
   }
}
err_close_dmabuf:
   close(bo->dmabuf_fd);
err_free_bo:
   pan_kmod_dev_free(dev, bo);
   return NULL;
}

static struct pan_kmod_bo *
kbase_kmod_bo_alloc_dmabuf(struct pan_kmod_dev *dev, struct pan_kmod_vm *exclusive_vm,
                           uint64_t size, uint32_t flags)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);
   struct dma_heap_allocation_data alloc = {
      .len = DIV_ROUND_UP(size, 4096) * 4096,
      .fd_flags = O_RDWR | O_CLOEXEC,
   };

   if (ioctl(kbase_dev->dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc)) {
      mesa_loge("kbase: DMA_HEAP_IOCTL_ALLOC failed: %s", strerror(errno));
      return NULL;
   }

   struct pan_kmod_bo *bo =
      kbase_kmod_bo_import_dmabuf(dev, exclusive_vm, alloc.fd, alloc.len, flags);
   close(alloc.fd);
   return bo;
}

static struct pan_kmod_bo *
kbase_kmod_bo_alloc(struct pan_kmod_dev *dev, struct pan_kmod_vm *exclusive_vm, uint64_t size,
                    uint32_t flags)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   if (!exclusive_vm && kbase_dev->dma_heap_fd >= 0 &&
       !(flags & (PAN_KMOD_BO_FLAG_EXECUTABLE | PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT |
                  PAN_KMOD_BO_FLAG_KBASE_SYNC)))
      return kbase_kmod_bo_alloc_dmabuf(dev, exclusive_vm, size, flags);

   /* Nothing in the captured traffic shows how to ask for a GPU-uncached mapping on the JM
    * kernels, and guessing a flag bit there would be silently wrong rather than loudly. Refuse, as
    * panfrost_kmod does. The CSF UAPI has it (BASE_MEM_UNCACHED_GPU). */
   if ((flags & PAN_KMOD_BO_FLAG_GPU_UNCACHED) && !kbase_dev->csf)
      return NULL;

   const bool exec = flags & PAN_KMOD_BO_FLAG_EXECUTABLE;
   const bool grow = !exec && (flags & PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT);
   const enum kbase_guard gmode = !exec && !grow && !(flags & PAN_KMOD_BO_FLAG_KBASE_SYNC)
                                     ? kbase_guard_mode()
                                     : KBASE_GUARD_OFF;
   const bool guard = gmode != KBASE_GUARD_OFF;

   struct kbase_kmod_bo *bo = pan_kmod_dev_alloc(dev, sizeof(*bo));
   if (!bo)
      return NULL;
   bo->dmabuf_fd = -1;
   bo->cpu_mapping = NULL;
   bo->gpu_mapping = NULL;
   bo->imported = false;
   bo->same_va = !exec;

   const uint64_t pages = DIV_ROUND_UP(size, 4096);
   union kbase_ioctl_mem_alloc req = {
      .in = {
         .va_pages = pages + guard,
         .commit_pages = grow ? MIN2(pages, KBASE_GROW_INITIAL_PAGES)
                              : pages + (gmode == KBASE_GUARD_ABSORB),
         .extension = grow ? KBASE_GROW_STEP_PAGES : 0,
         .flags = exec ? KBASE_MEM_FLAGS_EXEC
                       : KBASE_MEM_FLAGS_RW | (grow ? BASE_MEM_GROW_ON_GPF : 0) |
                            (kbase_mem_group() << BASEP_MEM_GROUP_ID_SHIFT) |
                            ((flags & PAN_KMOD_BO_FLAG_WB_MMAP) ? BASE_MEM_CACHED_CPU : 0),
      },
   };

   /* Not CSF_EVENT memory, although a queue descheduled while waiting on a sync object is
    * only resumed by kbase re-reading that object through the permanent kernel mapping such
    * memory has: kbase makes CSF_EVENT memory GPU-uncached on a system that is not fully
    * coherent, and the tiler does not work out of it (a draw rasterised nothing, every
    * descriptor and the vertex data in it; a plain BO draws). PanVK's waits are on queues of the
    * same group, one of which is making progress, so the group is not idle while they wait. */
   if (kbase_dev->csf && !exec && (flags & PAN_KMOD_BO_FLAG_GPU_UNCACHED))
      req.in.flags |= BASE_MEM_UNCACHED_GPU;

   /* Sync objects: CSF_EVENT, in exactly the shape csfring allocated its seqnos with (kbase adds
    * CACHED_CPU and COHERENT_SYSTEM itself, and refuses the vendor's RW shape with the flag). */
   if (kbase_dev->csf && !exec && !grow && (flags & PAN_KMOD_BO_FLAG_KBASE_SYNC))
      req.in.flags = 0xfull | BASE_MEM_SAME_VA | BASE_MEM_CSF_EVENT;

   const union kbase_ioctl_mem_alloc first = req;
   int ret = pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_ALLOC, &req);
   if (ret && (first.in.flags & BASEP_MEM_GROUP_ID_MASK)) {
      req = first;
      req.in.flags &= ~BASEP_MEM_GROUP_ID_MASK;
      ret = pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_ALLOC, &req);
      if (!ret)
         kbase_mem_group_refused();
   }
   if (ret) {
      mesa_loge("kbase: MEM_ALLOC of %" PRIu64 " pages, flags 0x%" PRIx64 ", failed (err=%d)",
                pages, (uint64_t)req.in.flags, errno);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   bo->exec = exec;
   bo->anchor = NULL;

   if (exec) {
      /* The exec zone hands back the address itself. Leave the mapping to whoever wants one:
       * this zone allows only a single mmap, so taking an anchor here would spend it. */
      bo->gpu_va = req.out.gpu_va;
   } else {
      /* req.out.gpu_va is a cookie. Spend it on the anchor, and the address that comes back is
       * the GPU VA. The anchor is never unmapped before bo_free, because unmapping it frees the
       * allocation. It must cover the whole reservation, guard page included (kbase refuses a
       * cookie mapping of any other length). */
      bo->anchor_size = (pages + guard) * 4096;
      void *anchor = mmap(NULL, bo->anchor_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                          (off_t)req.out.gpu_va);
      if (anchor == MAP_FAILED) {
         mesa_loge("kbase: could not map the anchor for cookie 0x%" PRIx64 " (err=%d)",
                   (uint64_t)req.out.gpu_va, errno);
         pan_kmod_dev_free(dev, bo);
         return NULL;
      }
      bo->anchor = anchor;
      bo->gpu_va = (uint64_t)(uintptr_t)anchor;
   }

   pan_kmod_bo_init(&bo->base, dev, exclusive_vm, pages * 4096, flags,
                    p_atomic_inc_return(&kbase_dev->next_handle));
   kbase_bo_history_add(kbase_bo_allocs, &kbase_bo_alloc_next, bo->gpu_va, pages * 4096, flags);
   kbase_live_add(bo->gpu_va, pages * 4096, flags, !exec, gmode == KBASE_GUARD_DETECT,
                  (uintptr_t)__builtin_return_address(0));
   return &bo->base;
}

static void
kbase_kmod_bo_free(struct pan_kmod_bo *bo)
{
   struct kbase_kmod_bo *kbase_bo = container_of(bo, struct kbase_kmod_bo, base);

   /* Out of the live table before the anchor goes, so the anchor check never sees our own
    * release as someone else's. */
   kbase_live_remove(kbase_bo->gpu_va);
   kbase_bo_history_add(kbase_bo_frees, &kbase_bo_free_next, kbase_bo->gpu_va, bo->size,
                        bo->flags);

   if (kbase_bo->imported) {
      if (kbase_bo->cpu_mapping &&
          os_munmap(kbase_bo->cpu_mapping, bo->size))
         mesa_loge("kbase: unmapping dma-buf CPU view failed: %s", strerror(errno));
      if (kbase_bo->gpu_mapping &&
          os_munmap(kbase_bo->gpu_mapping, bo->size))
         mesa_loge("kbase: unmapping dma-buf GPU view failed: %s", strerror(errno));
      if (!kbase_bo->same_va) {
         struct kbase_ioctl_mem_free req = {.gpu_addr = kbase_bo->gpu_va};
         if (pan_kmod_ioctl(bo->dev->fd, KBASE_IOCTL_MEM_FREE, &req))
            mesa_loge("kbase: MEM_FREE(0x%" PRIx64 ") failed: %s",
                      kbase_bo->gpu_va, strerror(errno));
      }
      close(kbase_bo->dmabuf_fd);
   } else if (kbase_bo->exec) {
      struct kbase_ioctl_mem_free req = {.gpu_addr = kbase_bo->gpu_va};
      if (pan_kmod_ioctl(bo->dev->fd, KBASE_IOCTL_MEM_FREE, &req))
         mesa_loge("kbase: MEM_FREE(0x%" PRIx64 ") failed (err=%d)", kbase_bo->gpu_va, errno);
   } else if (kbase_bo->anchor) {
      /* Dropping the anchor IS the free for a SAME_VA region. MEM_FREE refuses these. */
      if (munmap(kbase_bo->anchor, kbase_bo->anchor_size))
         mesa_loge("kbase: releasing the anchor at %p failed (err=%d)", kbase_bo->anchor, errno);

   }

   pan_kmod_dev_free(bo->dev, kbase_bo);
}

static void
kbase_kmod_bo_set_label(struct pan_kmod_dev *dev, struct pan_kmod_bo *bo, const char *label)
{
   kbase_live_set_label(container_of(bo, struct kbase_kmod_bo, base)->gpu_va, label);
}

static void *
kbase_kmod_bo_mmap(struct pan_kmod_bo *bo, int prot, int flags, void *host_addr)
{
   struct kbase_kmod_bo *kbase_bo = container_of(bo, struct kbase_kmod_bo, base);

   if (kbase_bo->imported) {
      if (!kbase_bo->cpu_mapping) {
         errno = EINVAL;
         return MAP_FAILED;
      }
      if (!host_addr || host_addr == kbase_bo->cpu_mapping)
         return kbase_bo->cpu_mapping;

      return os_mmap(host_addr, bo->size, prot, MAP_SHARED | MAP_FIXED,
                     kbase_bo->dmabuf_fd, 0);
   }

   /* Both zones map at offset = the BO's GPU VA. For an exec BO that is the address the kernel
    * returned; for an ordinary one it is the anchor's address, and mapping it again yields a
    * second CPU view of the same memory. */
   const off_t offset = kbase_bo->gpu_va;
   return os_mmap(host_addr, bo->size, prot, flags, bo->dev->fd, offset);
}

static int
kbase_kmod_bo_export(struct pan_kmod_bo *bo)
{
   struct kbase_kmod_bo *kbase_bo = container_of(bo, struct kbase_kmod_bo, base);
   if (kbase_bo->dmabuf_fd < 0) {
      errno = ENOTSUP;
      return -1;
   }

   return fcntl(kbase_bo->dmabuf_fd, F_DUPFD_CLOEXEC, 3);
}

static int
kbase_kmod_bo_munmap(struct pan_kmod_bo *bo, void *host_addr, size_t size)
{
   struct kbase_kmod_bo *kbase_bo = container_of(bo, struct kbase_kmod_bo, base);

   if (kbase_bo->imported &&
       (host_addr == kbase_bo->cpu_mapping || host_addr == kbase_bo->gpu_mapping))
      return 0;

   return os_munmap(host_addr, size);
}

static bool
kbase_kmod_bo_wait(struct pan_kmod_bo *bo, int64_t timeout_ns, bool for_read_only_access)
{
   /* NOT IMPLEMENTED, and deliberately reported rather than faked as success. kbase expresses
    * completion through job atoms and its own fence fds, not through a per-BO wait, so this
    * needs the submission path before it can mean anything. Returning true here would turn a
    * missing sync into silent corruption. */
   mesa_loge("kbase: bo_wait is not implemented yet");
   return false;
}

/* Called with dev->handle_to_bo.lock held. kbase has no handle namespace a dma-buf maps back
 * to, so every import is a new BO, registered under a handle of ours. */
static struct pan_kmod_bo *
kbase_kmod_bo_import(struct pan_kmod_dev *dev, int fd)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   const off_t size = lseek(fd, 0, SEEK_END);
   if (size <= 0) {
      mesa_loge("kbase: invalid dmabuf size");
      return NULL;
   }

   struct kbase_kmod_bo *bo = pan_kmod_dev_alloc(dev, sizeof(*bo));
   if (!bo)
      return NULL;

   /* CPU and GPU read/write, SAME_VA. Not COHERENT_LOCAL: that is a property of memory kbase
    * allocates, and an import is the exporter's. */
   int dmabuf = fd;
   union kbase_ioctl_mem_import req;
   memset(&req, 0, sizeof(req));
   req.in.flags = 0xfull | BASE_MEM_SAME_VA;
   req.in.phandle = (uint64_t)(uintptr_t)&dmabuf;
   req.in.type = BASE_MEM_IMPORT_TYPE_UMM;
   static bool import_40;
   int ret = pan_kmod_ioctl(dev->fd, import_40 ? KBASE_IOCTL_MEM_IMPORT_40
                                                : KBASE_IOCTL_MEM_IMPORT, &req);
   if (ret && errno == ENOTTY && !import_40) {
      ret = pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_IMPORT_40, &req);
      if (!ret)
         import_40 = true;
   }
   if (ret) {
      mesa_loge("kbase: MEM_IMPORT of dma-buf fd %d failed (err=%d)", fd, errno);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   const uint64_t pages = req.out.va_pages;
   if (!(req.out.flags & (BASE_MEM_SAME_VA | BASE_MEM_NEED_MMAP)) || pages * 4096 < size) {
      mesa_loge("kbase: MEM_IMPORT returned flags 0x%" PRIx64 ", %" PRIu64 " pages for %" PRIu64
                " bytes; only a SAME_VA import is handled",
                (uint64_t)req.out.flags, pages, size);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   void *anchor = mmap(NULL, pages * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                       (off_t)req.out.gpu_va);
   if (anchor == MAP_FAILED) {
      mesa_loge("kbase: could not map the imported cookie 0x%" PRIx64 " (err=%d)",
                (uint64_t)req.out.gpu_va, errno);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   bo->exec = false;
   bo->anchor = anchor;
   bo->anchor_size = pages * 4096;
   bo->gpu_va = (uint64_t)(uintptr_t)anchor;
   pan_kmod_bo_init(&bo->base, dev, NULL, pages * 4096, PAN_KMOD_BO_FLAG_IMPORTED,
                    p_atomic_inc_return(&kbase_dev->next_handle));
   kbase_bo_history_add(kbase_bo_allocs, &kbase_bo_alloc_next, bo->gpu_va, pages * 4096,
                        PAN_KMOD_BO_FLAG_IMPORTED);
   kbase_live_add(bo->gpu_va, pages * 4096, PAN_KMOD_BO_FLAG_IMPORTED, true, false,
                  (uintptr_t)__builtin_return_address(0));

   struct pan_kmod_bo **slot = util_sparse_array_get(&dev->handle_to_bo.array, bo->base.handle);
   if (!slot) {
      kbase_kmod_bo_free(&bo->base);
      return NULL;
   }
   *slot = &bo->base;
   return &bo->base;
}

/* ---- address space -------------------------------------------------------------------------
 *
 * kbase gives a context exactly one GPU address space and assigns every VA itself at allocation
 * time, so there is nothing to create and nothing to bind. That is not a compromise forced by
 * this backend: DRM panfrost behaves the same way, and panfrost_kmod likewise refuses anything
 * but PAN_KMOD_VM_FLAG_AUTO_VA, one VM per device, and whole-BO mappings. */

static struct pan_kmod_vm *
kbase_kmod_vm_create(struct pan_kmod_dev *dev, uint32_t flags, uint64_t va_start,
                     uint64_t va_range)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   if (kbase_dev->vm) {
      mesa_loge("kbase: only one VM per device is supported");
      return NULL;
   }

   if (!(flags & PAN_KMOD_VM_FLAG_AUTO_VA)) {
      mesa_loge("kbase: only PAN_KMOD_VM_FLAG_AUTO_VA is supported -- the kernel owns the VA");
      return NULL;
   }

   struct kbase_kmod_vm *vm = pan_kmod_dev_alloc(dev, sizeof(*vm));
   if (!vm)
      return NULL;

   pan_kmod_vm_init(&vm->base, dev, 0, flags);
   kbase_dev->vm = vm;
   return &vm->base;
}

static void
kbase_kmod_vm_destroy(struct pan_kmod_vm *vm)
{
   struct kbase_kmod_dev *kbase_dev = container_of(vm->dev, struct kbase_kmod_dev, base);
   struct kbase_kmod_vm *kbase_vm = container_of(vm, struct kbase_kmod_vm, base);

   kbase_dev->vm = NULL;
   pan_kmod_vm_cleanup(vm);
   pan_kmod_dev_free(vm->dev, kbase_vm);
}

static int
kbase_kmod_vm_bind(struct pan_kmod_vm *vm, enum pan_kmod_vm_op_mode mode,
                   struct pan_kmod_vm_op *ops, uint32_t op_count)
{
   if (mode != PAN_KMOD_VM_OP_MODE_IMMEDIATE &&
       mode != PAN_KMOD_VM_OP_MODE_DEFER_TO_NEXT_IDLE_POINT) {
      mesa_loge("kbase: unsupported vm_bind mode %d", mode);
      return -1;
   }

   for (uint32_t i = 0; i < op_count; i++) {
      if (ops[i].type == PAN_KMOD_VM_OP_TYPE_MAP) {
         if (ops[i].va.start != PAN_KMOD_VM_MAP_AUTO_VA) {
            mesa_loge("kbase: only auto-VA mapping is possible -- the kernel picks the address");
            return -1;
         }
         /* The mapping already exists: MEM_ALLOC created it and returned its VA. Report that
          * address back so the caller learns where the kernel put it. */
         struct kbase_kmod_bo *bo = container_of(ops[i].map.bo, struct kbase_kmod_bo, base);
         ops[i].va.start = bo->gpu_va;
      } else if (ops[i].type != PAN_KMOD_VM_OP_TYPE_UNMAP) {
         mesa_loge("kbase: unsupported vm_bind op %d", ops[i].type);
         return -1;
      }
      /* UNMAP is a no-op: the mapping goes away with the allocation, in bo_free. */
   }

   return 0;
}

/* ---- sync objects ----------------------------------------------------------------------------
 *
 * kbase has no syncobjs, and the DRM syncobj ioctls panvk used to issue on the kbase fd only
 * worked on the Exynos kernel because it returns success for DRM ioctls it does not implement
 * (see "THE TRAP" below): every wait "succeeded" at once and nothing was ever signaled. A
 * MediaTek kbase (Galaxy A31, Mali-G52) rejects them, so panvk found no device there at all.
 *
 * So the objects live in this process: binary, in one process-wide table indexed by handle, with
 * waits on a condition variable (CLOCK_MONOTONIC, DRM's absolute timeouts and -ETIME). A GPU
 * signal is done by the kmod's event thread when the submission that carries it completes
 * (pan_kmod_kbase_submit_async); until then the object is SUBMITTED, which is what a DRM
 * syncobj holding an unsignaled fence is, and satisfies WAIT_AVAILABLE. A submitted object can
 * hold a sync file from a kbase fence-trigger atom that signals with the GPU work: that is what
 * an export hands to the compositor, so presenting never waits for the GPU. */

enum ks_state {
   KS_FREE = 0,
   KS_UNSIGNALED,
   KS_SIGNALED,
   KS_SUBMITTED,
};

struct ks_obj {
   uint8_t state;
   /* KS_SUBMITTED from an import: nothing here signals it, the sync file below does. */
   bool external;
   int fd; /* KS_SUBMITTED: a sync file signaling with it, or -1 */
   /* KS_SUBMITTED by a CSF submission: its group and seqno, so a later submission on the same
    * group can wait for it on the GPU (pan_kmod_kbase_sync_csf_wait_seq). */
   const void *csf_group;
   uint64_t csf_seq;
};

static pthread_mutex_t ks_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ks_cond;
static pthread_once_t ks_once = PTHREAD_ONCE_INIT;
static struct ks_obj *ks_objs;
static uint32_t ks_cap, ks_hint;

/* The runtime (vk_drm_syncobj.c) tells a timeout from an error by errno == ETIME, the way
 * libdrm's wrappers leave it: every failure here sets errno too. Without it, a fence that was
 * merely not signaled yet read as a lost device -- which never happened while every submission
 * blocked, and killed Minecraft once they stopped blocking. */
static int
ks_ret(int ret)
{
   if (ret < 0)
      errno = -ret;
   return ret;
}

static void
ks_init(void)
{
   pthread_condattr_t a;
   pthread_condattr_init(&a);
   pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
   pthread_cond_init(&ks_cond, &a);
   pthread_condattr_destroy(&a);
}

static struct ks_obj *
ks_get(uint32_t h)
{
   return h && h <= ks_cap && ks_objs[h - 1].state != KS_FREE ? &ks_objs[h - 1] : NULL;
}

static void
ks_set_state(struct ks_obj *o, uint8_t state, int fd)
{
   if (o->fd >= 0)
      close(o->fd);
   o->state = state;
   o->external = false;
   o->fd = fd;
   o->csf_group = NULL;
}

static int
ks_create(struct util_sync_provider *p, uint32_t flags, uint32_t *handle)
{
   pthread_mutex_lock(&ks_lock);
   uint32_t i = ks_hint;
   while (i < ks_cap && ks_objs[i].state != KS_FREE)
      i++;
   if (i >= ks_cap) {
      for (i = 0; i < ks_cap && ks_objs[i].state != KS_FREE; i++)
         ;
   }
   if (i >= ks_cap) {
      const uint32_t cap = ks_cap ? ks_cap * 2 : 256;
      struct ks_obj *o = realloc(ks_objs, cap * sizeof(*o));
      if (!o) {
         pthread_mutex_unlock(&ks_lock);
         return ks_ret(-ENOMEM);
      }
      for (uint32_t k = ks_cap; k < cap; k++)
         o[k] = (struct ks_obj){.state = KS_FREE, .fd = -1};
      i = ks_cap;
      ks_objs = o;
      ks_cap = cap;
   }
   ks_objs[i] = (struct ks_obj){
      .state = (flags & DRM_SYNCOBJ_CREATE_SIGNALED) ? KS_SIGNALED : KS_UNSIGNALED, .fd = -1};
   ks_hint = i + 1;
   *handle = i + 1;
   pthread_mutex_unlock(&ks_lock);
   return 0;
}

static int
ks_destroy(struct util_sync_provider *p, uint32_t handle)
{
   pthread_mutex_lock(&ks_lock);
   struct ks_obj *o = ks_get(handle);
   if (o) {
      ks_set_state(o, KS_FREE, -1);
      ks_hint = MIN2(ks_hint, handle - 1);
   }
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(o ? 0 : -EINVAL);
}

static int
ks_set(const uint32_t *handles, uint32_t count, uint8_t state)
{
   int ret = 0;
   pthread_mutex_lock(&ks_lock);
   for (uint32_t i = 0; i < count; i++) {
      struct ks_obj *o = ks_get(handles[i]);
      if (o)
         ks_set_state(o, state, -1);
      else
         ret = -EINVAL;
   }
   pthread_cond_broadcast(&ks_cond);
   pthread_mutex_unlock(&ks_lock);
   return ret;
}

static int
ks_signal(struct util_sync_provider *p, const uint32_t *handles, uint32_t count)
{
   return ks_ret(ks_set(handles, count, KS_SIGNALED));
}

static int
ks_reset(struct util_sync_provider *p, const uint32_t *handles, uint32_t count)
{
   return ks_ret(ks_set(handles, count, KS_UNSIGNALED));
}

void
pan_kmod_kbase_sync_submitted(const uint32_t *handles, unsigned count)
{
   pthread_once(&ks_once, ks_init);
   ks_set(handles, count, KS_SUBMITTED);
}

void
pan_kmod_kbase_sync_attach_fd(const uint32_t *handles, unsigned count, int fd)
{
   if (fd < 0)
      return;
   pthread_mutex_lock(&ks_lock);
   for (uint32_t i = 0; i < count; i++) {
      struct ks_obj *o = ks_get(handles[i]);
      /* Only if still submitted: the event thread may have signaled it already. */
      if (o && o->state == KS_SUBMITTED && o->fd < 0)
         o->fd = dup(fd);
   }
   pthread_mutex_unlock(&ks_lock);
}

void
pan_kmod_kbase_sync_attach_csf(const uint32_t *handles, unsigned count, const void *group,
                               uint64_t seq)
{
   pthread_mutex_lock(&ks_lock);
   for (unsigned i = 0; i < count; i++) {
      struct ks_obj *o = ks_get(handles[i]);
      if (o && o->state == KS_SUBMITTED && !o->external) {
         o->csf_group = group;
         o->csf_seq = seq;
      }
   }
   pthread_mutex_unlock(&ks_lock);
}

bool
pan_kmod_kbase_sync_csf_wait_seq(const uint32_t *handles, unsigned count, const void *group,
                                 uint64_t *seq)
{
   bool ok = true;
   *seq = 0;
   pthread_mutex_lock(&ks_lock);
   for (unsigned i = 0; i < count && ok; i++) {
      const struct ks_obj *o = ks_get(handles[i]);
      if (o && o->state == KS_SIGNALED)
         continue;
      if (!o || o->state != KS_SUBMITTED || o->csf_group != group)
         ok = false;
      else
         *seq = MAX2(*seq, o->csf_seq);
   }
   pthread_mutex_unlock(&ks_lock);
   return ok;
}

struct ks_sync_merge_data {
   char name[32];
   int32_t fd2;
   int32_t fence;
   uint32_t flags;
   uint32_t pad;
};
#define KS_SYNC_IOC_MERGE _IOWR('>', 3, struct ks_sync_merge_data)

int
pan_kmod_kbase_sync_merged_fd(const uint32_t *handles, unsigned count)
{
   int fd = -1;
   pthread_mutex_lock(&ks_lock);
   for (unsigned i = 0; i < count; i++) {
      struct ks_obj *o = ks_get(handles[i]);
      if (o && o->state == KS_SIGNALED)
         continue;
      if (!o || o->state != KS_SUBMITTED || o->fd < 0) {
         if (fd >= 0)
            close(fd);
         fd = -2;
         break;
      }
      if (fd < 0) {
         fd = dup(o->fd);
      } else {
         struct ks_sync_merge_data m = {.name = "panvk", .fd2 = o->fd};
         const int r = ioctl(fd, KS_SYNC_IOC_MERGE, &m);
         close(fd);
         fd = r ? -2 : m.fence;
      }
      if (fd < 0) {
         fd = -2;
         break;
      }
   }
   pthread_mutex_unlock(&ks_lock);
   return fd;
}

static uint64_t
ks_now(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* Called with ks_lock held. WAIT_AVAILABLE is satisfied by a submitted object, everything else
 * waits for it to be signaled; WAIT_FOR_SUBMIT needs nothing extra, an unsubmitted object just
 * keeps the wait going. */
static int
ks_wait_locked(uint32_t *handles, unsigned n, int64_t timeout_nsec, unsigned flags,
               uint32_t *first_signaled)
{
   const bool all = flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
   const bool available = flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE;

   for (;;) {
      unsigned done = 0, first = UINT32_MAX;
      bool external = false;
      for (unsigned i = 0; i < n; i++) {
         struct ks_obj *o = ks_get(handles[i]);
         if (!o)
            return -EINVAL;
         if (o->state == KS_SUBMITTED && o->external) {
            struct pollfd pfd = {.fd = o->fd, .events = POLLIN};
            if (o->fd < 0 || poll(&pfd, 1, 0) > 0)
               ks_set_state(o, KS_SIGNALED, -1);
            else
               external = true;
         }
         if (o->state == KS_SIGNALED || (available && o->state == KS_SUBMITTED)) {
            done++;
            first = MIN2(first, i);
         }
      }
      if (all ? done == n : done > 0) {
         if (first_signaled)
            *first_signaled = first == UINT32_MAX ? 0 : first;
         return 0;
      }
      if (timeout_nsec != INT64_MAX && (uint64_t)timeout_nsec <= ks_now())
         return -ETIME;
      if (external) {
         /* An imported sync file signals without telling anyone: look again in a millisecond. */
         const uint64_t next = MIN2((uint64_t)timeout_nsec, ks_now() + 1000000ull);
         const struct timespec ts = {.tv_sec = next / 1000000000ull, .tv_nsec = next % 1000000000ull};
         pthread_cond_timedwait(&ks_cond, &ks_lock, &ts);
      } else if (timeout_nsec == INT64_MAX) {
         pthread_cond_wait(&ks_cond, &ks_lock);
      } else {
         const struct timespec ts = {.tv_sec = timeout_nsec / 1000000000ll,
                                     .tv_nsec = timeout_nsec % 1000000000ll};
         pthread_cond_timedwait(&ks_cond, &ks_lock, &ts);
      }
   }
}

static int
ks_wait(struct util_sync_provider *p, uint32_t *handles, unsigned n, int64_t timeout_nsec,
        unsigned flags, uint32_t *first_signaled)
{
   pthread_mutex_lock(&ks_lock);
   const int ret = ks_wait_locked(handles, n, timeout_nsec, flags, first_signaled);
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(ret);
}

/* Binary objects only (panvk drops the timeline feature before v10), but the runtime takes this
 * path for every VK_SYNC_WAIT_PENDING wait because only it carries WAIT_AVAILABLE. */
static int
ks_timeline_wait(struct util_sync_provider *p, uint32_t *handles, uint64_t *points,
                 unsigned n, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
{
   return ks_wait(p, handles, n, timeout_nsec, flags, first_signaled);
}

static int
ks_query(struct util_sync_provider *p, uint32_t *handles, uint64_t *points, uint32_t count,
         uint32_t flags)
{
   return ks_ret(-ENOSYS);
}

static int
ks_transfer(struct util_sync_provider *p, uint32_t dst, uint64_t dst_point, uint32_t src,
            uint64_t src_point, uint32_t flags)
{
   pthread_mutex_lock(&ks_lock);
   struct ks_obj *d = ks_get(dst), *s = ks_get(src);
   if (d && s && d != s) {
      ks_set_state(d, s->state, s->fd >= 0 ? dup(s->fd) : -1);
      d->external = s->external;
      pthread_cond_broadcast(&ks_cond);
   }
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(d && s ? 0 : -EINVAL);
}

static int
ks_export_sync_file(struct util_sync_provider *p, uint32_t handle, int *out_fd)
{
   const uint64_t deadline = ks_now() + 10ull * 1000000000ull;
   pthread_mutex_lock(&ks_lock);
   /* Wait for the signal operation to be submitted (normally it is already). */
   int ret = ks_wait_locked(&handle, 1, deadline, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE, NULL);
   struct ks_obj *o = ret ? NULL : ks_get(handle);
   if (o && o->state == KS_SUBMITTED) {
      if (o->fd >= 0) {
         /* The GPU is still at it: hand out the fence-trigger sync file. */
         *out_fd = dup(o->fd);
         pthread_mutex_unlock(&ks_lock);
         return *out_fd >= 0 ? 0 : ks_ret(-errno);
      }
      /* Submitted without a sync file: all that is left is to wait for the GPU. */
      ret = ks_wait_locked(&handle, 1, deadline, 0, NULL);
   }
   pthread_mutex_unlock(&ks_lock);
   if (ret) {
      mesa_loge("kbase: sync file export of an object never signaled (%d)", ret);
      return ks_ret(ret);
   }
   *out_fd = -1;
   return 0;
}

/* Not a CPU wait: the object stays pending on a copy of the sync file, and a wait on it polls
 * that. Waiting here held vkAcquireNextImageKHR until the compositor released the buffer, and a
 * present (vkQueueSignalReleaseImageANDROID imports the frame's own fence) until the GPU had
 * finished the frame. A submission waiting on such an object still waits on the CPU, but by then
 * the compositor has normally long released the buffer. */
static int
ks_import_sync_file(struct util_sync_provider *p, uint32_t handle, int fd)
{
   struct pollfd pfd = {.fd = fd, .events = POLLIN};
   if (fd < 0 || poll(&pfd, 1, 0) > 0)
      return ks_signal(p, &handle, 1);
   const int copy = dup(fd);
   if (copy < 0)
      return ks_ret(-errno);
   pthread_mutex_lock(&ks_lock);
   struct ks_obj *o = ks_get(handle);
   if (o) {
      ks_set_state(o, KS_SUBMITTED, copy);
      o->external = true;
      pthread_cond_broadcast(&ks_cond);
   } else {
      close(copy);
   }
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(o ? 0 : -EINVAL);
}

static int
ks_no_fd(struct util_sync_provider *p, uint32_t handle, int *out_fd)
{
   return ks_ret(-ENOSYS);
}

static int
ks_no_handle(struct util_sync_provider *p, int fd, uint32_t *handle)
{
   return ks_ret(-ENOSYS);
}

static void
ks_finalize(struct util_sync_provider *p)
{
   free(p);
}

static struct util_sync_provider *
ks_clone(struct util_sync_provider *p)
{
   return pan_kmod_kbase_sync_provider();
}

struct util_sync_provider *
pan_kmod_kbase_sync_provider(void)
{
   pthread_once(&ks_once, ks_init);

   struct util_sync_provider *p = calloc(1, sizeof(*p));
   if (!p)
      return NULL;

   *p = (struct util_sync_provider){
      .create = ks_create,
      .destroy = ks_destroy,
      .handle_to_fd = ks_no_fd,
      .fd_to_handle = ks_no_handle,
      .import_sync_file = ks_import_sync_file,
      .export_sync_file = ks_export_sync_file,
      .wait = ks_wait,
      .timeline_wait = ks_timeline_wait,
      .query = ks_query,
      .reset = ks_reset,
      .signal = ks_signal,
      .transfer = ks_transfer,
      .finalize = ks_finalize,
      .clone = ks_clone,
   };
   return p;
}

/* ---- job submission -------------------------------------------------------------------------
 *
 * THE TRAP THAT MAKES THIS SECTION NECESSARY. This kernel returns SUCCESS for DRM ioctls it
 * does not implement. Measured, on our own driver, with the spy attached:
 *
 *    [IOCTL] fd=7 NON-KBASE type=0x64 nr=64  size=48 -> 0     DRM_IOCTL_PANFROST_SUBMIT
 *    [IOCTL] fd=7 NON-KBASE type=0x64 nr=191 size=8  -> 0     DRM_IOCTL_SYNCOBJ_CREATE
 *    [IOCTL] fd=7 NON-KBASE type=0x64 nr=195 size=40 -> 0     DRM_IOCTL_SYNCOBJ_WAIT
 *
 * So the unported JM path did not fail -- it reported success, every syncobj wait returned
 * immediately, and vkQueueSubmit and vkQueueWaitIdle both said rc=0 over a GPU that had been
 * asked for nothing at all. The only reason it was not mistaken for a working driver is that
 * the readback was verified. Nothing here may rely on an ioctl return alone.
 *
 * WHAT SUBMISSION LOOKS LIKE, read off the vendor rendering a triangle:
 * one JOB_SUBMIT carrying an array of 64-byte atoms, and completion reported by READING the
 * device fd -- 24-byte records, not a fence. The atom layout below is pinned by an unusually
 * strong check: the two udata words we put in an atom came back BYTE-IDENTICAL in that atom's
 * completion event, which fixes both structures at once against the same evidence.
 */

struct kbase_ioctl_job_submit {
   uint64_t addr;
   uint32_t nr_atoms;
   uint32_t stride;
};

struct kbase_jd_dependency {
   uint8_t atom_id;
   uint8_t dependency_type;
};

/* The atom, as the code below fills it: base_jd_atom_v2's fields. What goes to the kernel is 64
 * bytes either way, laid out as kbase_kmod_dev::atom_layout says (kbase_pack_atoms). */
struct kbase_jd_atom {
   uint64_t jc;
   uint64_t udata[2];
   uint64_t extres_list;
   uint16_t nr_extres;
   uint8_t jit_id[2];
   struct kbase_jd_dependency pre_dep[2];
   uint8_t atom_number;
   int8_t prio;
   uint8_t device_nr;
   uint8_t jobslot;
   uint32_t core_req;
};
static_assert(sizeof(struct kbase_jd_atom) == 48, "base_jd_atom_v2 up to renderpass_id");

/* kbase picks the atom layout by the stride alone, and 64 bytes means two different things:
 *  - ARM's kbase: base_jd_atom, a u64 seq_nr and then the v2 fields, renderpass_id, padding[7].
 *    The vendor driver sends this on the Exynos 9820 (captured).
 *  - MediaTek's kbase with CONFIG_MALI_MTK_GPU_BM_JM: base_jd_atom_v2 grows a u32 frame_nr at
 *    the end, which makes IT 64 bytes, so a 64-byte atom is read as v2 with no seq_nr. A
 *    seq_nr-first atom was then read 8 bytes off (Galaxy Tab A11, Helio G99): the event echoed
 *    the jc as udata[0], every atom number read as 0, and the second atom of a submit waited
 *    forever in JOB_SUBMIT for "atom 0" to be free. 48 (the old v2 layout) is refused there.
 * Which one a kernel uses is measured once per context (kbase_probe_atom_layout). */
enum kbase_atom_layout {
   KBASE_ATOM_SEQ_NR_FIRST = 0,
   KBASE_ATOM_V2_FRAME_NR,
};
#define KBASE_ATOM_STRIDE 64

static void
kbase_pack_atoms(enum kbase_atom_layout layout, const struct kbase_jd_atom *in, unsigned n,
                 uint8_t *out)
{
   memset(out, 0, (size_t)n * KBASE_ATOM_STRIDE);
   for (unsigned i = 0; i < n; i++) {
      /* seq_nr stays 0: the kernel only uses it for its own tracing. */
      uint8_t *dst = out + (size_t)i * KBASE_ATOM_STRIDE;
      memcpy(layout == KBASE_ATOM_SEQ_NR_FIRST ? dst + 8 : dst, &in[i], sizeof(in[i]));
   }
}



/* 24 bytes, and the read that returned two of them for a two-atom frame returned exactly 48. */
struct kbase_jd_event {
   uint32_t event_code;
   uint8_t atom_number;
   uint8_t padding[3];
   uint64_t udata[2];
};
static_assert(sizeof(struct kbase_jd_event) == 24, "two events came back as 48 bytes");

#define KBASE_IOCTL_JOB_SUBMIT _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)

/* The only event code seen, on every atom that completed correctly. Anything else is reported
 * rather than interpreted: a table of fault codes we have not observed would be a guess, and the
 * number itself is what a reader needs. */
#define KBASE_JD_EVENT_DONE 1

bool
pan_kmod_dev_is_kbase(const struct pan_kmod_dev *dev)
{
   return dev->ops == &kbase_kmod_ops;
}

bool
kbase_kmod_supports_dmabuf(const struct pan_kmod_dev *dev)
{
   const struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);
   return kbase_dev->dma_heap_fd >= 0;
}

/* ---- asynchronous submission ------------------------------------------------------------
 *
 * Submitting used to mean waiting: the caller read the device fd until every atom it had sent
 * reported. So the app's thread sat idle while the GPU drew each frame, and the next frame's
 * recording could only start after it -- measured in Minecraft on the S10e, about 13 ms of CPU
 * and 30 ms of GPU per frame, one after the other.
 *
 * Now one thread per device reads the events, and a submission returns once the kernel has its
 * atoms. Atom numbers (1..255, unique among the atoms in flight) come from one allocator. Each
 * submission runs after everything submitted before it: its first atom waits (ORDER, so a fault
 * does not spread) on the newest atom still in flight. When a completion callback is given, it
 * runs on the event thread once every atom of the submission has reported, before
 * pan_kmod_kbase_wait_seq sees it complete.
 *
 * A fence-trigger soft atom (BASE_JD_REQ_SOFT_FENCE_TRIGGER, mali_kbase_softjobs.c) waiting on
 * the submission's last atom gives a sync file at submit time that signals with the work. The
 * kernel runs soft atoms from a worker, measurably later than the GPU atom they wait on, so the
 * fence atom is kept out of everything else: the submission completes, and the next one starts,
 * on its last GPU atom. It is tracked on its own, only to free its number. */

#define KBASE_JD_DEP_ORDER              2u
#define KBASE_JD_REQ_SOFT_FENCE_TRIGGER 0x202u /* BASE_JD_REQ_SOFT_JOB | 0x2 */
#define KBASE_ATOM_TAG                  0x50414e564bull /* "PANVK", comes back in the event */
#define KBASE_ATOM_PROBE_TAG            0x45424f5250ull /* "PROBE" */

/* Which 64-byte atom layout this kernel reads (see enum kbase_atom_layout): one dependency-only
 * atom (no job chain, nothing reaches the GPU) is submitted seq_nr-first, and the completion
 * event says how it was read. udata comes back verbatim from wherever the kernel took it: in
 * place if the layout matched, shifted by the 8 bytes of seq_nr if the kernel read it as v2
 * (udata[0] is then the jc, 0, and udata[1] the tag). Runs before the event thread exists. */
static enum kbase_atom_layout
kbase_probe_atom_layout(int fd)
{
   struct kbase_jd_atom a = {
      .udata = {KBASE_ATOM_PROBE_TAG, 0},
      .atom_number = 1,
      /* core_req 0: BASE_JD_REQ_DEP, an atom that only orders others. */
   };
   uint8_t packed[KBASE_ATOM_STRIDE];
   kbase_pack_atoms(KBASE_ATOM_SEQ_NR_FIRST, &a, 1, packed);
   struct kbase_ioctl_job_submit req = {
      .addr = (uintptr_t)packed, .nr_atoms = 1, .stride = KBASE_ATOM_STRIDE};
   if (pan_kmod_ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &req)) {
      mesa_logw("kbase: atom layout probe refused (err=%d), assuming seq_nr first", errno);
      return KBASE_ATOM_SEQ_NR_FIRST;
   }

   for (unsigned tries = 0; tries < 20; tries++) {
      struct pollfd pfd = {.fd = fd, .events = POLLIN};
      if (poll(&pfd, 1, 50) <= 0)
         continue;
      struct kbase_jd_event ev[4];
      const ssize_t n = read(fd, ev, sizeof(ev));
      for (ssize_t i = 0; n > 0 && i < n / (ssize_t)sizeof(ev[0]); i++) {
         if (ev[i].udata[0] == KBASE_ATOM_PROBE_TAG)
            return KBASE_ATOM_SEQ_NR_FIRST;
         if (ev[i].udata[1] == KBASE_ATOM_PROBE_TAG)
            return KBASE_ATOM_V2_FRAME_NR;
      }
   }
   mesa_logw("kbase: no event for the atom layout probe, assuming seq_nr first");
   return KBASE_ATOM_SEQ_NR_FIRST;
}

static int
kbase_probe_atom_layout_fd(int fd)
{
   return kbase_probe_atom_layout(fd);
}


struct kbase_sub {
   uint64_t seq;
   unsigned pending;
   bool ok;
   pan_kmod_kbase_done_cb done;
   void *data;
};

static struct kbase_queue *
kbase_queue(struct pan_kmod_dev *dev)
{
   return &container_of(dev, struct kbase_kmod_dev, base)->queue;
}

static void *
kbase_event_thread(void *arg)
{
   struct kbase_kmod_dev *kdev = arg;
   struct kbase_queue *q = &kdev->queue;

   for (;;) {
      pthread_mutex_lock(&q->lock);
      const bool stop = q->stop && !q->in_flight;
      pthread_mutex_unlock(&q->lock);
      if (stop)
         break;

      struct pollfd pfd = {.fd = kdev->base.fd, .events = POLLIN};
      if (poll(&pfd, 1, 100) <= 0)
         continue;

      struct kbase_jd_event ev[16];
      const ssize_t n = read(kdev->base.fd, ev, sizeof(ev));
      if (n <= 0 || n % sizeof(struct kbase_jd_event)) {
         if (n < 0 && (errno == EAGAIN || errno == EINTR))
            continue;
         mesa_loge("kbase: event read returned %zd, which is not a whole number of records", n);
         continue;
      }

      struct kbase_sub *finished[16];
      unsigned nr_finished = 0;

      pthread_mutex_lock(&q->lock);
      for (unsigned i = 0; i < n / sizeof(struct kbase_jd_event); i++) {
         if (ev[i].udata[0] != KBASE_ATOM_TAG)
            continue; /* not one of ours */
         const uint8_t an = ev[i].atom_number;
         struct kbase_sub *sub = q->owner[an];
         if (!sub) {
            mesa_loge("kbase: event for atom %u, which is not in flight", an);
            continue;
         }
         q->owner[an] = NULL;
         q->free_atoms++;
         for (unsigned t = 0; t < PAN_KMOD_KBASE_NR_TRACKS; t++) {
            if (q->track[t] == an)
               q->track[t] = 0;
         }
         if (ev[i].event_code != KBASE_JD_EVENT_DONE) {
            if (q->is_fence[an] && !sub->done) {
               if (!q->fence_warned) {
                  mesa_loge("kbase: fence-trigger atom finished with event code %u; presents "
                            "will wait for the GPU", ev[i].event_code);
                  q->fence_warned = true;
               }
            } else {
               mesa_loge("kbase: atom %u finished with event code %u, not DONE", an,
                         ev[i].event_code);
               sub->ok = false;
            }
         }
         q->is_fence[an] = false;
         if (--sub->pending == 0)
            finished[nr_finished++] = sub;
      }
      pthread_cond_broadcast(&q->cond);
      pthread_mutex_unlock(&q->lock);

      for (unsigned i = 0; i < nr_finished; i++) {
         struct kbase_sub *sub = finished[i];
         if (sub->done)
            sub->done(sub->data, sub->ok);
         pthread_mutex_lock(&q->lock);
         /* seq 0: a fence atom of its own, not a submission. */
         for (unsigned k = 0; sub->seq && k < q->nr_open_seq; k++) {
            if (q->open_seq[k] == sub->seq) {
               q->open_seq[k] = q->open_seq[--q->nr_open_seq];
               break;
            }
         }
         q->in_flight--;
         pthread_cond_broadcast(&q->cond);
         pthread_mutex_unlock(&q->lock);
         free(sub);
      }
   }
   return NULL;
}

static void
kbase_queue_init(struct kbase_queue *q)
{
   pthread_mutex_init(&q->lock, NULL);
   /* CLOCK_MONOTONIC, for pan_kmod_kbase_wait_seq_timeout. */
   pthread_condattr_t attr;
   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   pthread_cond_init(&q->cond, &attr);
   pthread_condattr_destroy(&attr);
   q->free_atoms = 255;
   q->next_seq = 1;
}

static void
kbase_queue_finish(struct kbase_kmod_dev *kdev)
{
   struct kbase_queue *q = &kdev->queue;

   pthread_mutex_lock(&q->lock);
   q->stop = true;
   const bool started = q->thread_started;
   pthread_mutex_unlock(&q->lock);
   if (started)
      pthread_join(q->thread, NULL);
   pthread_cond_destroy(&q->cond);
   pthread_mutex_destroy(&q->lock);
}

/* Adds an ORDER dependency on atom id (0: nothing to wait for) unless the atom already waits for
 * it. False when both slots are taken. */
static bool
kbase_add_order_dep(struct kbase_jd_atom *jd, uint8_t id)
{
   if (!id)
      return true;
   for (unsigned s = 0; s < 2; s++) {
      if (jd->pre_dep[s].atom_id == id)
         return true;
      if (!jd->pre_dep[s].atom_id) {
         jd->pre_dep[s].atom_id = id;
         jd->pre_dep[s].dependency_type = KBASE_JD_DEP_ORDER;
         return true;
      }
   }
   return false;
}

uint64_t
pan_kmod_kbase_submit_async(struct pan_kmod_dev *dev, const struct pan_kmod_kbase_atom *atoms,
                            unsigned nr_atoms, int *fence_fd, pan_kmod_kbase_done_cb done,
                            void *data)
{
   struct kbase_kmod_dev *kdev = container_of(dev, struct kbase_kmod_dev, base);
   struct kbase_queue *q = &kdev->queue;

   if (fence_fd)
      *fence_fd = -1;

   if (nr_atoms > PAN_KMOD_KBASE_MAX_ATOMS) {
      mesa_loge("kbase: %u atoms in one submit is more than this path handles", nr_atoms);
      return 0;
   }

   pthread_mutex_lock(&q->lock);

   if (!q->thread_started) {
      if (pthread_create(&q->thread, NULL, kbase_event_thread, kdev)) {
         pthread_mutex_unlock(&q->lock);
         mesa_loge("kbase: could not start the event thread");
         return 0;
      }
      q->thread_started = true;
   }

   const uint64_t seq = q->next_seq++;
   const bool fence =
      fence_fd && (nr_atoms || q->track[0] || q->track[1]);
   static_assert(PAN_KMOD_KBASE_TRACK_VTC == 1 && PAN_KMOD_KBASE_TRACK_FRAG == 2,
                 "track 0 is VTC and track 1 FRAG");
   const unsigned need = nr_atoms + fence;

   if (!need) {
      /* Nothing to run and nothing on the GPU: complete once earlier submissions have run their
       * callbacks, so completion stays in order. */
      while (q->nr_open_seq)
         pthread_cond_wait(&q->cond, &q->lock);
      pthread_mutex_unlock(&q->lock);
      if (done)
         done(data, true);
      return seq;
   }

   while (q->free_atoms < need)
      pthread_cond_wait(&q->cond, &q->lock);


   /* With GPU atoms, the fence atom gets a record of its own (seq 0, no callback); a fence
    * alone is the submission and completes with it. */
   const bool split = fence && nr_atoms;
   struct kbase_sub *sub = calloc(1, sizeof(*sub));
   struct kbase_sub *fsub = split ? calloc(1, sizeof(*fsub)) : NULL;
   if (!sub || (split && !fsub)) {
      free(sub);
      free(fsub);
      pthread_mutex_unlock(&q->lock);
      return 0;
   }
   *sub = (struct kbase_sub){.seq = seq, .pending = split ? nr_atoms : need, .ok = true,
                             .done = done, .data = data};
   if (split)
      *fsub = (struct kbase_sub){.pending = 1, .ok = true};

   uint8_t real[PAN_KMOD_KBASE_MAX_ATOMS + 1];
   uint8_t rel_to_real[256] = {0};
   unsigned next = 1;
   for (unsigned i = 0; i < need; i++) {
      while (q->owner[next])
         next++;
      assert(next <= 255);
      real[i] = next;
      q->owner[next] = split && i == nr_atoms ? fsub : sub;
      q->is_fence[next] = fence && i == nr_atoms;
      next++;
   }
   for (unsigned i = 0; i < nr_atoms; i++)
      rel_to_real[atoms[i].atom_number] = real[i];

   /* The tracks as they will be once this submission is in, committed if the kernel takes it. */
   uint8_t track[PAN_KMOD_KBASE_NR_TRACKS];
   memcpy(track, q->track, sizeof(track));

   struct kbase_jd_atom jd[PAN_KMOD_KBASE_MAX_ATOMS + 1];
   memset(jd, 0, need * sizeof(jd[0]));
   for (unsigned i = 0; i < nr_atoms; i++) {
      assert(atoms[i].atom_number != 0);
      jd[i].jc = atoms[i].jc;
      jd[i].core_req = atoms[i].core_req;
      jd[i].atom_number = real[i];
      /* udata is opaque to the kernel and comes back verbatim in the completion event: the tag
       * tells our events from anyone else's on the context. */
      jd[i].udata[0] = KBASE_ATOM_TAG;
      jd[i].udata[1] = seq;
      const bool all = !atoms[i].wait && !atoms[i].join;
      if (atoms[i].dep_atom) {
         jd[i].pre_dep[0].atom_id = rel_to_real[atoms[i].dep_atom];
         jd[i].pre_dep[0].dependency_type = KBASE_JD_DEP_DATA;
      }
      /* Everything before: the newest vertex/tiler and fragment atoms, which each come after
       * every earlier one of their kind. */
      const uint8_t wait = all ? (atoms[i].dep_atom ? 0 : PAN_KMOD_KBASE_TRACK_VTC |
                                                         PAN_KMOD_KBASE_TRACK_FRAG)
                               : atoms[i].wait;
      for (unsigned t = 0; t < PAN_KMOD_KBASE_NR_TRACKS; t++) {
         if ((wait & (1u << t)) && !kbase_add_order_dep(&jd[i], track[t])) {
            mesa_loge("kbase: atom %u waits for more than two atoms", atoms[i].atom_number);
            assert(!"too many atom waits");
         }
      }
      const uint8_t join = all ? (1u << PAN_KMOD_KBASE_NR_TRACKS) - 1 : atoms[i].join;
      for (unsigned t = 0; t < PAN_KMOD_KBASE_NR_TRACKS; t++) {
         if (join & (1u << t))
            track[t] = real[i];
      }
   }

   /* The kernel reads the struct at submit time and writes the sync file's fd back into it. */
   struct {
      int fd;
      int stream_fd;
   } base_fence = {-1, -1};
   if (fence) {
      struct kbase_jd_atom *f = &jd[nr_atoms];
      f->jc = (uintptr_t)&base_fence;
      f->core_req = KBASE_JD_REQ_SOFT_FENCE_TRIGGER;
      f->atom_number = real[nr_atoms];
      f->udata[0] = KBASE_ATOM_TAG;
      f->udata[1] = seq;
      /* After everything, this submission's atoms included. */
      kbase_add_order_dep(f, track[0]);
      kbase_add_order_dep(f, track[1]);
   }

   uint8_t packed[(PAN_KMOD_KBASE_MAX_ATOMS + 1) * KBASE_ATOM_STRIDE];
   kbase_pack_atoms(kdev->atom_layout, jd, need, packed);
   struct kbase_ioctl_job_submit req = {
      .addr = (uintptr_t)packed,
      .nr_atoms = need,
      .stride = KBASE_ATOM_STRIDE,
   };

   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_JOB_SUBMIT, &req)) {
      mesa_loge("kbase: JOB_SUBMIT of %u atoms failed (err=%d)", need, errno);
      for (unsigned i = 0; i < need; i++) {
         q->owner[real[i]] = NULL;
         q->is_fence[real[i]] = false;
      }
      pthread_mutex_unlock(&q->lock);
      free(sub);
      free(fsub);
      return 0;
   }

   q->free_atoms -= need;
   /* What later atoms wait on: GPU atoms, never a fence atom. */
   memcpy(q->track, track, sizeof(track));
   q->in_flight += split ? 2 : 1;
   q->open_seq[q->nr_open_seq++] = seq;
   pthread_mutex_unlock(&q->lock);

   if (fence_fd)
      *fence_fd = base_fence.fd;
   return seq;
}

void
pan_kmod_kbase_wait_seq(struct pan_kmod_dev *dev, uint64_t seq)
{
   struct kbase_queue *q = kbase_queue(dev);

   pthread_mutex_lock(&q->lock);
   if (!seq)
      seq = q->next_seq - 1;
   for (;;) {
      bool open = false;
      for (unsigned k = 0; k < q->nr_open_seq && !open; k++)
         open = q->open_seq[k] <= seq;
      if (!open)
         break;
      pthread_cond_wait(&q->cond, &q->lock);
   }
   pthread_mutex_unlock(&q->lock);
}

bool
pan_kmod_kbase_wait_seq_timeout(struct pan_kmod_dev *dev, uint64_t seq, int64_t abs_timeout_ns)
{
   struct kbase_queue *q = kbase_queue(dev);

   if (!seq)
      return true;

   bool done = false;
   pthread_mutex_lock(&q->lock);
   for (;;) {
      bool open = false;
      for (unsigned k = 0; k < q->nr_open_seq && !open; k++)
         open = q->open_seq[k] <= seq;
      if (!open) {
         done = true;
         break;
      }
      if (abs_timeout_ns == INT64_MAX) {
         pthread_cond_wait(&q->cond, &q->lock);
         continue;
      }
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      const int64_t now_ns = (int64_t)now.tv_sec * 1000000000ll + now.tv_nsec;
      if (now_ns >= abs_timeout_ns)
         break;
      const struct timespec until = {.tv_sec = abs_timeout_ns / 1000000000ll,
                                     .tv_nsec = abs_timeout_ns % 1000000000ll};
      pthread_cond_timedwait(&q->cond, &q->lock, &until);
   }
   pthread_mutex_unlock(&q->lock);
   return done;
}

static void
kbase_blocking_done(void *data, bool ok)
{
   *(bool *)data = ok;
}

bool
pan_kmod_kbase_submit(struct pan_kmod_dev *dev, const struct pan_kmod_kbase_atom *atoms,
                      unsigned nr_atoms)
{
   bool ok = true;
   const uint64_t seq =
      pan_kmod_kbase_submit_async(dev, atoms, nr_atoms, NULL, kbase_blocking_done, &ok);
   if (!seq)
      return false;
   pan_kmod_kbase_wait_seq(dev, seq);
   return ok;
}

/* ---- repeated mappings -------------------------------------------------------------------------
 *
 * PanVK's CSF descriptor ring is one BO mapped twice back to back, so an allocation that runs
 * off the end continues at the start. kbase picks every VA itself, but MEM_ALIAS builds one new
 * region out of pieces of existing native allocations: count entries of the same BO, a stride
 * of the BO's size apart, is exactly that layout. On a 64-bit context the alias is SAME_VA, so
 * its cookie is mapped like any allocation, and the mapping's address is the GPU VA. The caller
 * wants it aligned (the ring is addressed with 32-bit wraparound), and kbase picks the address
 * itself (MAP_FIXED and hints are refused), but it aligns any SAME_VA mapping of 2 MB or more to
 * 2 MB. So the BO is repeated until the alias is at least 2 MB: the extra repeats are never
 * addressed, and an aligned 2 MB range never crosses a 4 GB boundary. */

union kbase_ioctl_mem_alias {
   struct {
      uint64_t flags;
      uint64_t stride;
      uint64_t nents;
      uint64_t aliasing_info;
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
      uint64_t va_pages;
   } out;
};
#define KBASE_IOCTL_MEM_ALIAS _IOWR(KBASE_IOCTL_TYPE, 21, union kbase_ioctl_mem_alias)
#define KBASE_SAME_VA_LARGE_ALIGN (2ull * 1024 * 1024)

struct kbase_mem_aliasing_info {
   uint64_t handle; /* the aliased region's GPU VA */
   uint64_t offset; /* pages */
   uint64_t length; /* pages */
};

static unsigned
kbase_repeat_count(uint64_t bo_size, unsigned count)
{
   return MAX2(count, DIV_ROUND_UP(KBASE_SAME_VA_LARGE_ALIGN, bo_size));
}

uint64_t
pan_kmod_kbase_bo_map_repeated(struct pan_kmod_bo *bo, unsigned count, uint64_t align)
{
   struct kbase_kmod_bo *kbo = container_of(bo, struct kbase_kmod_bo, base);
   const uint64_t pages = bo->size / 4096;
   const unsigned n = kbase_repeat_count(bo->size, count);
   struct kbase_mem_aliasing_info ai[8];

   if (n > ARRAY_SIZE(ai) || kbo->exec || align > KBASE_SAME_VA_LARGE_ALIGN)
      return 0;
   for (unsigned i = 0; i < n; i++)
      ai[i] = (struct kbase_mem_aliasing_info){.handle = kbo->gpu_va, .length = pages};

   /* COHERENT_LOCAL as the aliased BO has it: kbase refuses an alias whose coherency differs
    * from what it aliases. */
   union kbase_ioctl_mem_alias req = {
      .in = {
         .flags = BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR | BASE_MEM_PROT_CPU_RD |
                  BASE_MEM_COHERENT_LOCAL,
         .stride = pages,
         .nents = n,
         .aliasing_info = (uint64_t)(uintptr_t)ai,
      },
   };
   if (pan_kmod_ioctl(bo->dev->fd, KBASE_IOCTL_MEM_ALIAS, &req)) {
      mesa_loge("kbase: MEM_ALIAS of %u x %" PRIu64 " pages failed (err=%d)", n, pages, errno);
      return 0;
   }

   const uint64_t size = pages * n * 4096;
   void *p = mmap(NULL, size, PROT_READ, MAP_SHARED, bo->dev->fd, (off_t)req.out.gpu_va);
   if (p == MAP_FAILED) {
      mesa_loge("kbase: mapping the alias cookie 0x%" PRIx64 " failed (err=%d)",
                (uint64_t)req.out.gpu_va, errno);
      return 0;
   }
   if ((uintptr_t)p & (align - 1)) {
      mesa_loge("kbase: the alias landed at %p, not aligned to 0x%" PRIx64, p, align);
      munmap(p, size);
      return 0;
   }
   kbase_live_add((uint64_t)(uintptr_t)p, size, bo->flags | KBASE_LIVE_REPEATED, true, false,
                  (uintptr_t)__builtin_return_address(0));
   return (uint64_t)(uintptr_t)p;
}

void
pan_kmod_kbase_unmap_repeated(struct pan_kmod_bo *bo, uint64_t va, unsigned count)
{
   /* As for any SAME_VA region, dropping the mapping frees the alias. */
   kbase_live_remove(va);
   munmap((void *)(uintptr_t)va, bo->size * kbase_repeat_count(bo->size, count));
}

/* ---- CSF -------------------------------------------------------------------------------------
 *
 * Valhall 2nd gen and later have no job manager: work is a command stream that the GPU's
 * firmware runs from a ring buffer. kbase (UAPI 1.x, r44p1 on the Dimensity 9300) exposes that
 * as queue GROUPS of up to eight queues, each queue a ring in our own memory plus two pages the
 * firmware and we share (CS_INSERT/CS_EXTRACT_INIT in, CS_EXTRACT/CS_ACTIVE out), and a KICK
 * ioctl that has the kernel scheduler put the group on a slot. Everything panthor's kernel does
 * between a GROUP_SUBMIT and the ring is done here instead, measured first on the device:
 *
 *   per call:  MOVE32 flush_id=0; FLUSH_CACHE2 clean+invalidate; MOVE48/MOVE32 stream; WAIT;
 *              CALL
 *   at the end of a submission, on every queue in it:
 *              FLUSH_CACHE2 clean; WAIT all; SYNC_ADD64 (system scope) on the queue's seqno;
 *              ERROR_BARRIER
 *
 * The clean before the signal is not optional: without it the seqno (in uncached CSF_EVENT
 * memory) is visible before what the stream wrote, which csfring measured as a round whose
 * signal arrived with the previous round's data. The ring code uses the last four CS registers,
 * which are the ones a CS builder leaves to the kernel (unpreserved_cs_reg_count).
 *
 * Completion: the firmware raises an event per sync update and kbase turns it into POLLIN on
 * the context's fd (one per round in csfring), so one thread per device polls, drains the
 * notifications (a GPU_QUEUE_GROUP_ERROR marks its group faulted) and completes, in order, the
 * submissions whose queues have all reached their seqno targets. */

struct kbase_ioctl_cs_get_glb_iface {
   union {
      struct {
         uint32_t max_group_num;
         uint32_t max_total_stream_num;
         uint64_t groups_ptr;
         uint64_t streams_ptr;
      } in;
      struct {
         uint32_t glb_version;
         uint32_t features;
         uint32_t group_num;
         uint32_t prfcnt_size;
         uint32_t total_stream_num;
         uint32_t instr_features;
      } out;
   };
};

struct kbase_cs_group_control {
   uint32_t features, stream_num, suspend_size, padding;
};

struct kbase_cs_stream_control {
   uint32_t features, padding;
};

union kbase_ioctl_cs_queue_group_create {
   struct {
      uint64_t tiler_mask;
      uint64_t fragment_mask;
      uint64_t compute_mask;
      uint8_t cs_min;
      uint8_t priority;
      uint8_t tiler_max;
      uint8_t fragment_max;
      uint8_t compute_max;
      uint8_t csi_handlers;
      uint16_t reserved;
      uint64_t dvs_buf;
      uint64_t padding[9];
   } in;
   struct {
      uint8_t group_handle;
      uint8_t padding[3];
      uint32_t group_uid;
   } out;
};

struct kbase_ioctl_cs_queue_group_term {
   uint8_t group_handle;
   uint8_t padding[7];
};

struct kbase_ioctl_cs_queue_register {
   uint64_t buffer_gpu_addr;
   uint32_t buffer_size;
   uint8_t priority;
   uint8_t padding[3];
};

union kbase_ioctl_cs_queue_bind {
   struct {
      uint64_t buffer_gpu_addr;
      uint8_t group_handle;
      uint8_t csi_index;
      uint8_t padding[6];
   } in;
   struct {
      uint64_t mmap_handle;
   } out;
};

struct kbase_ioctl_cs_queue_addr {
   uint64_t buffer_gpu_addr;
};

union kbase_ioctl_cs_tiler_heap_init {
   struct {
      uint32_t chunk_size;
      uint32_t initial_chunks;
      uint32_t max_chunks;
      uint16_t target_in_flight;
      uint8_t group_id;
      uint8_t padding;
      uint64_t buf_desc_va;
   } in;
   struct {
      uint64_t gpu_heap_va;
      uint64_t first_chunk_va;
   } out;
};

struct kbase_ioctl_cs_tiler_heap_term {
   uint64_t gpu_heap_va;
};

#define KBASE_IOCTL_CS_QUEUE_REGISTER                                                             \
   _IOW(KBASE_IOCTL_TYPE, 36, struct kbase_ioctl_cs_queue_register)
#define KBASE_IOCTL_CS_QUEUE_KICK _IOW(KBASE_IOCTL_TYPE, 37, struct kbase_ioctl_cs_queue_addr)
#define KBASE_IOCTL_CS_QUEUE_BIND _IOWR(KBASE_IOCTL_TYPE, 39, union kbase_ioctl_cs_queue_bind)
#define KBASE_IOCTL_CS_QUEUE_TERMINATE                                                            \
   _IOW(KBASE_IOCTL_TYPE, 41, struct kbase_ioctl_cs_queue_addr)
#define KBASE_IOCTL_CS_QUEUE_GROUP_TERMINATE                                                      \
   _IOW(KBASE_IOCTL_TYPE, 43, struct kbase_ioctl_cs_queue_group_term)
#define KBASE_IOCTL_CS_TILER_HEAP_INIT                                                            \
   _IOWR(KBASE_IOCTL_TYPE, 48, union kbase_ioctl_cs_tiler_heap_init)
#define KBASE_IOCTL_CS_TILER_HEAP_TERM                                                            \
   _IOW(KBASE_IOCTL_TYPE, 49, struct kbase_ioctl_cs_tiler_heap_term)
#define KBASE_IOCTL_CS_GET_GLB_IFACE                                                              \
   _IOWR(KBASE_IOCTL_TYPE, 51, struct kbase_ioctl_cs_get_glb_iface)
#define KBASE_IOCTL_CS_QUEUE_GROUP_CREATE                                                         \
   _IOWR(KBASE_IOCTL_TYPE, 58, union kbase_ioctl_cs_queue_group_create)

/* BASE_CSF_NOTIFICATION_*: what a read() of the context's fd returns, 64 bytes each. */
#define KBASE_CSF_NOTIFICATION_SIZE              64
#define KBASE_CSF_NOTIFICATION_GPU_QUEUE_GROUP_ERROR 1

/* CS opcodes and fields (genxml v12, "CS ..."). */
#define CS_OP_MOVE48        1u
#define CS_OP_MOVE32        2u
#define CS_OP_WAIT          3u
#define CS_OP_CALL          32u
#define CS_OP_FLUSH_CACHE2  36u
#define CS_OP_ERROR_BARRIER 47u
#define CS_OP_SYNC_ADD64    51u
#define CS_OP_SYNC_WAIT64   53u

#define KBASE_CSF_RING_SIZE (64 * 1024)
#define KBASE_CSF_MAX_SUBS  256

/* Deferred vertex shading (RUN_IDVS2, Vertex Shading Mode = Deferred) keeps each shader core's
 * vertex positions in a per-group work buffer between the binning and the main phase. kbase hands
 * it to the firmware only when dvs_buf is non-zero (program_csg_slot); with zero, the CSG slot
 * keeps the buffer of whichever group ran there before, usually another process's, and deferred
 * draws fault there. Layout as the Arm driver sets it up on G720 MC12: one slot of this many pages
 * per core index up to the highest present core, one page more, and the slot size in pages in
 * the low 12 bits of dvs_buf (CSG_DVS_BUF_BUFFER_SIZE). */
#define KBASE_CSF_DVS_PAGES_PER_CORE 6

struct kbase_csf_ring {
   uint64_t va;
   uint64_t *cpu;
   void *io;
   volatile uint64_t *input;        /* [0] CS_INSERT, [1] CS_EXTRACT_INIT */
   volatile const uint64_t *output; /* [0] CS_EXTRACT, [1] CS_ACTIVE */
   uint64_t insert;                 /* bytes written, never wrapped */
   uint64_t signals;                /* SYNC_ADD64s emitted */
};

struct kbase_csf_sub {
   uint64_t seq;
   uint32_t mask;
   uint64_t target[PAN_KMOD_KBASE_CSF_MAX_QUEUES];
   pan_kmod_kbase_done_cb done;
   void *data;
};

struct pan_kmod_kbase_csf_group {
   struct kbase_kmod_dev *kdev;
   struct list_head link;
   uint8_t handle;
   unsigned nr_queues;
   struct kbase_csf_ring ring[PAN_KMOD_KBASE_CSF_MAX_QUEUES];
   /* One u64 seqno per queue, 64 bytes apart, in CSF_EVENT memory. */
   uint64_t seq_va;
   volatile uint64_t *seqno;
   /* The deferred vertex shading work buffer (KBASE_CSF_DVS_PAGES_PER_CORE). */
   void *dvs;
   uint64_t dvs_va, dvs_size;
   /* Submissions in flight, oldest first. */
   struct kbase_csf_sub subs[KBASE_CSF_MAX_SUBS];
   unsigned first_sub, nr_subs;
   uint64_t next_seq, done_seq;
   bool faulted;
};

/* A fence handed out by pan_kmod_kbase_csf_fence_fd whose submission is not done yet.
 *
 * On the Tab S10 Ultra's kernel (MediaTek/Samsung r44p1) a fence signal that waits in a KCPU
 * queue for 2 s trips kcpu_queue_timeout_worker, whose queue dump overruns a buffer and panics
 * the device (kbasep_csf_sync_kcpu_dump_locked). The timer starts when the signal is enqueued
 * (or when the one before it signals), so the poll thread drains the queue itself once the
 * oldest export has waited KBASE_CSF_FENCE_WATCHDOG_NS, or as soon as its group faults:
 * deleting a KCPU queue skips its waits and signals its fences (delete_queue). A GPU job that
 * really runs longer than that shows a frame early; a hang or a fault no longer takes the
 * device down. */
/* Every submission that signals anything exports one (panvk's gpu_queue_submit), so this has
 * to hold everything in flight: KBASE_CSF_MAX_SUBS per group, a few groups. Filling it drains
 * early, which shows frames before they are done. */
#define KBASE_CSF_MAX_EXPORTS        1024
#define KBASE_CSF_FENCE_WATCHDOG_NS  1000000000ull

struct kbase_csf_export {
   struct pan_kmod_kbase_csf_group *g;
   uint64_t t_ns;
   uint32_t mask;
   uint64_t target[PAN_KMOD_KBASE_CSF_MAX_QUEUES];
};

struct kbase_csf_dev {
   pthread_mutex_t lock;
   pthread_cond_t cond;
   struct list_head groups;
   pthread_t thread;
   bool thread_started, stop;
   /* The KCPU queue the submission fences come from (pan_kmod_kbase_csf_fence_fd). */
   bool kcpu_created;
   uint8_t kcpu_id;
   /* Exports still waiting on the GPU, oldest first (see kbase_csf_fence_watchdog). */
   struct kbase_csf_export exports[KBASE_CSF_MAX_EXPORTS];
   unsigned nr_exports;
};

static uint64_t
kbase_now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* KBASE_LOG_FILE=<path>: fault and watchdog events are also appended there and synced, so they
 * survive what they are about (logcat does not survive a kernel panic). */
static void PRINTFLIKE(1, 2)
kbase_log_persist(const char *fmt, ...)
{
   const char *path = getenv("KBASE_LOG_FILE");
   if (!path || !*path)
      return;
   char line[512];
   va_list ap;
   va_start(ap, fmt);
   int n = vsnprintf(line, sizeof(line) - 1, fmt, ap);
   va_end(ap);
   if (n < 0)
      return;
   n = MIN2(n, (int)sizeof(line) - 2);
   line[n++] = '\n';
   const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
   if (fd < 0)
      return;
   char stamp[48];
   const int s = snprintf(stamp, sizeof(stamp), "%" PRIu64 " pid %d: ", kbase_now_ns(),
                          (int)getpid());
   if (s > 0 && write(fd, stamp, s) < 0) {
      close(fd);
      return;
   }
   if (write(fd, line, n) >= 0)
      fsync(fd);
   close(fd);
}

/* /proc/self/maps, whole and NUL-terminated (free() it), or NULL. */
static char *
kbase_read_maps(void)
{
   const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return NULL;
   size_t cap = 256 * 1024, len = 0;
   char *buf = malloc(cap);
   while (buf) {
      if (cap - len < 64 * 1024) {
         char *b = realloc(buf, cap * 2);
         if (!b) {
            free(buf);
            buf = NULL;
            break;
         }
         buf = b;
         cap *= 2;
      }
      const ssize_t r = read(fd, buf + len, cap - len - 1);
      if (r <= 0)
         break;
      len += r;
   }
   close(fd);
   if (buf)
      buf[len] = '\0';
   return buf;
}

/* One /proc/self/maps line: "start-end perms offset dev inode path". */
struct kbase_maps_line {
   uint64_t start, end, offset;
   const char *text;
};

static bool
kbase_parse_maps_line(const char *line, struct kbase_maps_line *out)
{
   unsigned long long s, e, off;
   if (sscanf(line, "%llx-%llx %*s %llx", &s, &e, &off) != 3)
      return false;
   *out = (struct kbase_maps_line){.start = s, .end = e, .offset = off, .text = line};
   return true;
}

/* Logs the /proc/self/maps lines overlapping [lo, hi), at most max of them. */
static void
kbase_log_maps_range(uint64_t lo, uint64_t hi, unsigned max)
{
   char *text = kbase_read_maps();
   if (!text) {
      mesa_loge("kbase: could not read /proc/self/maps (err=%d)", errno);
      return;
   }
   unsigned shown = 0;
   for (char *line = text; *line && shown < max;) {
      char *nl = strchr(line, '\n');
      if (nl)
         *nl = '\0';
      struct kbase_maps_line m;
      if (kbase_parse_maps_line(line, &m) && m.start < hi && m.end > lo) {
         mesa_loge("kbase:   maps: %s", line);
         kbase_log_persist("  maps: %s", line);
         shown++;
      }
      if (!nl)
         break;
      line = nl + 1;
   }
   if (!shown) {
      mesa_loge("kbase:   maps: nothing between 0x%" PRIx64 " and 0x%" PRIx64, lo, hi);
      kbase_log_persist("  maps: nothing between 0x%" PRIx64 " and 0x%" PRIx64, lo, hi);
   }
   free(text);
}

/* Whether maps[k...] is still the anchor of [va, va + size): kbase's node, mapped at the
 * region's own GPU address (kbase sets a SAME_VA mapping's page offset to its region's start,
 * so offset == start), over the whole range. Another mapping of ours that landed on top of it
 * (a placed vkMapMemory of a different BO, say) fails the offset test. */
static bool
kbase_maps_is_anchor(const struct kbase_maps_line *maps, uint32_t nr, uint32_t k, uint64_t va,
                     uint64_t size)
{
   uint64_t end = va;
   for (; k < nr && maps[k].start <= end; k++) {
      if (!strstr(maps[k].text, "/dev/mali") || maps[k].offset != maps[k].start)
         return false;
      end = MAX2(end, maps[k].end);
      if (end >= va + size)
         return true;
   }
   return false;
}

/* KBASE_CHECK_ANCHORS=1: called every couple of seconds by the event thread, checks that each
 * anchored live mapping is still the mapping we made. One that is not was unmapped or mapped
 * over by someone else in the process, and the kernel freed its region with it. Reported once
 * each, with what holds the address now. */
static void
kbase_live_check(void)
{
   static unsigned runs;
   const uint64_t t0 = kbase_now_ns();
   char *text = kbase_read_maps();
   if (!text)
      return;

   uint32_t nr = 0, cap = 4096;
   struct kbase_maps_line *maps = malloc(cap * sizeof(*maps));
   for (char *line = text; maps && *line;) {
      char *nl = strchr(line, '\n');
      if (nl)
         *nl = '\0';
      if (nr == cap) {
         struct kbase_maps_line *m = realloc(maps, 2 * cap * sizeof(*maps));
         if (!m) {
            free(maps);
            maps = NULL;
            break;
         }
         maps = m;
         cap *= 2;
      }
      if (kbase_parse_maps_line(line, &maps[nr]))
         nr++;
      if (!nl)
         break;
      line = nl + 1;
   }
   if (!maps) {
      free(text);
      return;
   }

   struct kbase_live_map gone[8];
   const char *gone_now[8];
   unsigned nr_gone = 0;
   uint32_t checked = 0;
   uint64_t lowest = UINT64_MAX, highest = 0;
   pthread_mutex_lock(&kbase_live_lock);
   for (uint32_t i = 0; i < kbase_live_count; i++) {
      struct kbase_live_map *l = &kbase_live[i];
      /* Made after the snapshot was read: not in it. */
      if (!l->anchored || l->reported || l->t_ns >= t0)
         continue;
      checked++;
      lowest = MIN2(lowest, l->va);
      highest = MAX2(highest, l->va + l->size);
      /* The first line ending above va: the one holding it, if any. The lines are sorted and
       * do not overlap. */
      uint32_t lo = 0, hi = nr;
      while (lo < hi) {
         const uint32_t mid = lo + (hi - lo) / 2;
         if (maps[mid].end <= l->va)
            lo = mid + 1;
         else
            hi = mid;
      }
      if (lo < nr && maps[lo].start <= l->va &&
          kbase_maps_is_anchor(maps, nr, lo, l->va, l->size))
         continue;
      l->reported = true;
      if (nr_gone < ARRAY_SIZE(gone)) {
         gone[nr_gone] = *l;
         gone_now[nr_gone] = lo < nr && maps[lo].start < l->va + l->size ? maps[lo].text : NULL;
         nr_gone++;
      }
   }
   pthread_mutex_unlock(&kbase_live_lock);

   const uint64_t ms = (kbase_now_ns() - t0) / 1000000;
   if (runs++ % 30 == 0) {
      mesa_logw("kbase: anchor check: %u live mappings between 0x%" PRIx64 " and 0x%" PRIx64
                ", %u maps lines, %" PRIu64 " ms",
                checked, lowest, highest, nr, ms);
      kbase_log_persist("anchor check: %u live mappings between 0x%" PRIx64 " and 0x%" PRIx64
                        ", %u maps lines, %" PRIu64 " ms",
                        checked, lowest, highest, nr, ms);
   }
   for (unsigned i = 0; i < nr_gone; i++) {
      char site[96];
      kbase_format_code_addr(gone[i].caller, site, sizeof(site));
      const uint64_t age = (t0 - gone[i].t_ns) / 1000000;
      mesa_loge("kbase: our mapping 0x%" PRIx64 "+0x%" PRIx64 " (flags 0x%x, made %" PRIu64
                " ms ago from %s) is GONE, and its GPU memory with it; there now: %s",
                gone[i].va, gone[i].size, gone[i].flags, age, site,
                gone_now[i] ? gone_now[i] : "nothing");
      kbase_log_persist("our mapping 0x%" PRIx64 "+0x%" PRIx64 " (flags 0x%x, made %" PRIu64
                        " ms ago from %s) is GONE; there now: %s",
                        gone[i].va, gone[i].size, gone[i].flags, age, site,
                        gone_now[i] ? gone_now[i] : "nothing");
   }
   free(maps);
   free(text);
}

/* A CPU view of one of our regions: a mapping of the kbase node that is NOT at its region's own
 * address (offset != start), which is what bo_mmap hands PanVK. Its offset is the region's GPU
 * address. The GPU has nothing at a view's addresses, so a view address in GPU hands is a CPU
 * pointer that was written where a GPU address belongs. */
struct kbase_cpu_view {
   uint64_t start, end, gpu;
};

/* Our CPU views from /proc/self/maps, sorted by start (malloc'ed; free() it). */
static uint32_t
kbase_read_cpu_views(struct kbase_cpu_view **out)
{
   *out = NULL;
   char *text = kbase_read_maps();
   if (!text)
      return 0;
   uint32_t n = 0, cap = 1024;
   struct kbase_cpu_view *v = malloc(cap * sizeof(*v));
   for (char *line = text; v && *line;) {
      char *nl = strchr(line, '\n');
      if (nl)
         *nl = '\0';
      struct kbase_maps_line m;
      if (kbase_parse_maps_line(line, &m) && m.offset != m.start && strstr(line, "/dev/mali")) {
         if (n == cap) {
            struct kbase_cpu_view *g = realloc(v, 2 * cap * sizeof(*v));
            if (!g) {
               free(v);
               v = NULL;
               break;
            }
            v = g;
            cap *= 2;
         }
         v[n++] = (struct kbase_cpu_view){.start = m.start, .end = m.end, .gpu = m.offset};
      }
      if (!nl)
         break;
      line = nl + 1;
   }
   free(text);
   *out = v;
   return v ? n : 0;
}

static const struct kbase_cpu_view *
kbase_find_cpu_view(const struct kbase_cpu_view *v, uint32_t n, uint64_t addr)
{
   uint32_t lo = 0, hi = n;
   while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (v[mid].end <= addr)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo < n && v[lo].start <= addr ? &v[lo] : NULL;
}

/* Says whether a GPU fault address is one of our CPU views, i.e. a CPU pointer that the GPU was
 * given as an address. */
static void
kbase_explain_cpu_view(uint64_t addr)
{
   struct kbase_cpu_view *views;
   const uint32_t nv = kbase_read_cpu_views(&views);
   const struct kbase_cpu_view *v = kbase_find_cpu_view(views, nv, addr);
   if (v) {
      mesa_loge("kbase: fault address 0x%" PRIx64 " is our CPU VIEW 0x%" PRIx64 "-0x%" PRIx64
                " of the region at GPU 0x%" PRIx64 " (+0x%" PRIx64 "): a CPU pointer was used as"
                " a GPU address; its GPU address would be 0x%" PRIx64,
                addr, v->start, v->end, v->gpu, addr - v->start, v->gpu + (addr - v->start));
      kbase_log_persist("fault address 0x%" PRIx64 " is our CPU VIEW 0x%" PRIx64 "-0x%" PRIx64
                        " of the region at GPU 0x%" PRIx64 " (+0x%" PRIx64 "): a CPU pointer was"
                        " used as a GPU address; its GPU address would be 0x%" PRIx64,
                        addr, v->start, v->end, v->gpu, addr - v->start,
                        v->gpu + (addr - v->start));
   } else {
      mesa_loge("kbase: fault address 0x%" PRIx64 " is in none of our %u CPU views", addr, nv);
      kbase_log_persist("fault address 0x%" PRIx64 " is in none of our %u CPU views", addr, nv);
   }
   free(views);
}

/* Reads every live BO PanVK writes from the CPU, looking for 64-bit words whose low 48 bits are
 * an address inside one of our CPU views: a CPU pointer stored where the GPU will take it for a
 * GPU address (a descriptor, a command stream immediate). Words pointing within 64 KiB of near
 * are reported first. Holds kbase_live_lock while reading, so nothing is released under it;
 * meant for after a fault, when the device is lost anyway. */
static void
kbase_scan_cpu_pointers(uint64_t near)
{
   struct kbase_cpu_view *views;
   const uint32_t nv = kbase_read_cpu_views(&views);
   if (!nv) {
      free(views);
      return;
   }

   struct hit {
      uint64_t bo_va, bo_size, off, word, view_gpu, view_off;
      uint64_t ctx[6];
      uint32_t flags;
      uintptr_t caller;
      bool near;
   } hits[16];
   unsigned nr_hits = 0, total = 0, total_near = 0;
   uint64_t scanned = 0;
   const uint64_t t0 = kbase_now_ns();
   /* Not worth reading: no CPU writes (NO_MMAP), only partly backed (ALLOC_ON_FAULT, a read past
    * the backing faults), somebody else's (IMPORTED), our own sync objects and second mappings. */
   const uint32_t skip = PAN_KMOD_BO_FLAG_NO_MMAP | PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT |
                         PAN_KMOD_BO_FLAG_IMPORTED | PAN_KMOD_BO_FLAG_EXECUTABLE |
                         PAN_KMOD_BO_FLAG_KBASE_SYNC | KBASE_LIVE_REPEATED;

   pthread_mutex_lock(&kbase_live_lock);
   for (uint32_t i = 0; i < kbase_live_count && scanned < (1ull << 30); i++) {
      const struct kbase_live_map *l = &kbase_live[i];
      if (!l->anchored || (l->flags & skip) || l->size > (4u << 20))
         continue;
      const uint64_t *w = (const uint64_t *)(uintptr_t)l->va;
      const uint64_t nw = l->size / 8;
      for (uint64_t k = 0; k < nw; k++) {
         const uint64_t a = w[k] & 0xffffffffffffull;
         if (a < views[0].start || a >= views[nv - 1].end)
            continue;
         const struct kbase_cpu_view *v = kbase_find_cpu_view(views, nv, a);
         if (!v)
            continue;
         const bool is_near = near && a + 0x10000 > near && a < near + 0x10000;
         total++;
         total_near += is_near;
         /* Once the table is full, a near one replaces one that is not. */
         unsigned slot = nr_hits;
         if (slot == ARRAY_SIZE(hits)) {
            if (!is_near)
               continue;
            for (slot = 0; slot < ARRAY_SIZE(hits) && hits[slot].near; slot++)
               ;
            if (slot == ARRAY_SIZE(hits))
               continue;
         } else {
            nr_hits++;
         }
         struct hit *h = &hits[slot];
         *h = (struct hit){.bo_va = l->va, .bo_size = l->size, .off = k * 8, .word = a,
                           .view_gpu = v->gpu, .view_off = a - v->start, .flags = l->flags,
                           .caller = l->caller, .near = is_near};
         for (unsigned c = 0; c < 6; c++) {
            const int64_t j = (int64_t)k - 2 + c;
            h->ctx[c] = j >= 0 && (uint64_t)j < nw ? w[j] : 0;
         }
      }
      scanned += l->size;
   }
   pthread_mutex_unlock(&kbase_live_lock);

   const uint64_t ms = (kbase_now_ns() - t0) / 1000000;
   mesa_loge("kbase: CPU pointer scan: %u word(s) hold one of our CPU view addresses (%u near "
             "0x%" PRIx64 "), %" PRIu64 " MB read in %" PRIu64 " ms",
             total, total_near, near, scanned >> 20, ms);
   kbase_log_persist("CPU pointer scan: %u word(s) hold one of our CPU view addresses (%u near "
                     "0x%" PRIx64 "), %" PRIu64 " MB read in %" PRIu64 " ms",
                     total, total_near, near, scanned >> 20, ms);
   for (unsigned i = 0; i < nr_hits; i++) {
      const struct hit *h = &hits[i];
      char site[96];
      kbase_format_code_addr(h->caller, site, sizeof(site));
      mesa_loge("kbase:   0x%" PRIx64 " (CPU view of GPU 0x%" PRIx64 " +0x%" PRIx64 ") at BO "
                "0x%" PRIx64 "+0x%" PRIx64 " (size 0x%" PRIx64 ", flags 0x%x, from %s); words "
                "%016" PRIx64 " %016" PRIx64 " [%016" PRIx64 "] %016" PRIx64 " %016" PRIx64
                " %016" PRIx64,
                h->word, h->view_gpu, h->view_off, h->bo_va, h->off, h->bo_size, h->flags, site,
                h->ctx[0], h->ctx[1], h->ctx[2], h->ctx[3], h->ctx[4], h->ctx[5]);
      kbase_log_persist("  0x%" PRIx64 " (CPU view of GPU 0x%" PRIx64 " +0x%" PRIx64 ") at BO "
                        "0x%" PRIx64 "+0x%" PRIx64 " (size 0x%" PRIx64 ", flags 0x%x, from %s); "
                        "words %016" PRIx64 " %016" PRIx64 " [%016" PRIx64 "] %016" PRIx64
                        " %016" PRIx64 " %016" PRIx64,
                        h->word, h->view_gpu, h->view_off, h->bo_va, h->off, h->bo_size,
                        h->flags, site, h->ctx[0], h->ctx[1], h->ctx[2], h->ctx[3], h->ctx[4],
                        h->ctx[5]);
   }
   free(views);
}

/* KCPU queues (UAPI 1.x): the kernel runs their commands in order, so a CQS wait on the seqno
 * objects our rings signal, followed by a fence signal, gives a sync file that signals when the
 * GPU is done with a submission, with no CPU wait. */
struct kbase_ioctl_kcpu_queue_new {
   uint8_t id;
   uint8_t padding[7];
};
struct kbase_ioctl_kcpu_queue_delete {
   uint8_t id;
   uint8_t padding[7];
};
struct kbase_ioctl_kcpu_queue_enqueue {
   uint64_t addr;
   uint32_t nr_commands;
   uint8_t id;
   uint8_t padding[3];
};
#define KBASE_IOCTL_KCPU_QUEUE_CREATE _IOR(KBASE_IOCTL_TYPE, 45, struct kbase_ioctl_kcpu_queue_new)
#define KBASE_IOCTL_KCPU_QUEUE_DELETE _IOW(KBASE_IOCTL_TYPE, 46, struct kbase_ioctl_kcpu_queue_delete)
#define KBASE_IOCTL_KCPU_QUEUE_ENQUEUE                                                              _IOW(KBASE_IOCTL_TYPE, 47, struct kbase_ioctl_kcpu_queue_enqueue)
#define BASE_KCPU_COMMAND_TYPE_FENCE_SIGNAL        0
#define BASE_KCPU_COMMAND_TYPE_CQS_WAIT_OPERATION  4
struct base_cqs_wait_operation_info {
   uint64_t addr;
   uint64_t val;
   uint8_t operation; /* 1: wait while *addr <= val */
   uint8_t data_type; /* 1: u64 */
   uint8_t padding[6];
};
struct base_kcpu_command {
   uint8_t type;
   uint8_t padding[7];
   union {
      struct {
         uint64_t fence; /* struct base_fence * */
      } fence;
      struct {
         uint64_t objs;
         uint32_t nr_objs;
         uint32_t inherit_err_flags;
      } cqs_wait_operation;
      uint64_t pad[2];
   } info;
};
struct base_fence {
   struct {
      int fd;
      int stream_fd;
   } basep;
};

/* Under c->lock: drops finished exports; drains the KCPU queue if one has waited too long, its
 * group faulted, or `why` says so. */
static void
kbase_csf_fence_watchdog(struct kbase_kmod_dev *kdev, struct kbase_csf_dev *c, const char *why)
{
   const uint64_t now = kbase_now_ns();
   unsigned keep = 0;
   const char *stale = why;
   uint64_t waited = 0;

   for (unsigned i = 0; i < c->nr_exports; i++) {
      const struct kbase_csf_export *e = &c->exports[i];
      bool done = true;
      u_foreach_bit(q, e->mask) {
         if (e->g->seqno[q * 8] < e->target[q])
            done = false;
      }
      if (done)
         continue;
      if (!stale && e->g->faulted)
         stale = "its queue group faulted";
      if (!stale && now - e->t_ns > KBASE_CSF_FENCE_WATCHDOG_NS)
         stale = "it waited too long";
      waited = MAX2(waited, now - e->t_ns);
      c->exports[keep++] = *e;
   }
   c->nr_exports = keep;

   if (!stale || !c->kcpu_created)
      return;

   struct kbase_ioctl_kcpu_queue_delete del = {.id = c->kcpu_id};
   const int ret = pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_KCPU_QUEUE_DELETE, &del);
   const int err = ret ? errno : 0;
   mesa_loge("kbase: draining the fence queue (%u pending, oldest %" PRIu64 " ms): %s%s",
             c->nr_exports, waited / 1000000, stale, err ? " -- delete failed" : "");
   kbase_log_persist("draining the fence queue (%u pending, oldest %" PRIu64 " ms): %s, "
                     "delete %s (err %d)",
                     c->nr_exports, waited / 1000000, stale, err ? "failed" : "ok", err);
   c->kcpu_created = false;
   c->nr_exports = 0;
}

static uint64_t
cs_ins(unsigned op, uint64_t bits)
{
   return ((uint64_t)op << 56) | bits;
}

bool
pan_kmod_kbase_is_csf(const struct pan_kmod_dev *dev)
{
   return pan_kmod_dev_is_kbase(dev) &&
          container_of(dev, struct kbase_kmod_dev, base)->csf;
}

const struct drm_panthor_csif_info *
pan_kmod_kbase_csif_props(const struct pan_kmod_dev *dev)
{
   return &container_of(dev, struct kbase_kmod_dev, base)->csif;
}

bool
pan_kmod_kbase_frag_slot_writes(const struct pan_kmod_dev *dev)
{
   return pan_kmod_dev_is_kbase(dev) &&
          (container_of(dev, struct kbase_kmod_dev, base)->js0_features & (1u << 2));
}

static bool
kbase_csf_dev_init(struct kbase_kmod_dev *kdev)
{
   /* More than the firmware has is refused (EINVAL with 32 groups; csfring asks for 16). */
   struct kbase_cs_group_control groups[16];
   struct kbase_cs_stream_control streams[128];
   struct kbase_ioctl_cs_get_glb_iface req = {
      .in = {
         .max_group_num = ARRAY_SIZE(groups),
         .max_total_stream_num = ARRAY_SIZE(streams),
         .groups_ptr = (uint64_t)(uintptr_t)groups,
         .streams_ptr = (uint64_t)(uintptr_t)streams,
      },
   };
   if (pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_CS_GET_GLB_IFACE, &req) || !req.out.group_num) {
      mesa_loge("kbase: CS_GET_GLB_IFACE failed (err=%d)", errno);
      return false;
   }

   /* CS_FEATURES: work registers - 1 in bits 0..7, scoreboards in 8..15 (0x7107f on the G720:
    * 128 registers, 16 scoreboards). Four unpreserved registers, as panthor's firmware
    * interface reports. */
   const uint32_t cs_features = streams[0].features;
   kdev->csif = (struct drm_panthor_csif_info){
      .csg_slot_count = req.out.group_num,
      .cs_slot_count = groups[0].stream_num,
      .cs_reg_count = (cs_features & 0xff) + 1,
      .scoreboard_slot_count = (cs_features >> 8) & 0xff,
      .unpreserved_cs_reg_count = 4,
   };

   struct kbase_csf_dev *c = calloc(1, sizeof(*c));
   if (!c)
      return false;
   pthread_mutex_init(&c->lock, NULL);
   pthread_condattr_t attr;
   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   pthread_cond_init(&c->cond, &attr);
   pthread_condattr_destroy(&attr);
   list_inithead(&c->groups);
   kdev->csfq = c;
   return true;
}

static void
kbase_csf_dev_finish(struct kbase_kmod_dev *kdev)
{
   struct kbase_csf_dev *c = kdev->csfq;
   if (!c)
      return;

   pthread_mutex_lock(&c->lock);
   c->stop = true;
   const bool started = c->thread_started;
   pthread_mutex_unlock(&c->lock);
   if (started)
      pthread_join(c->thread, NULL);
   if (c->kcpu_created) {
      struct kbase_ioctl_kcpu_queue_delete del = {.id = c->kcpu_id};
      pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_KCPU_QUEUE_DELETE, &del);
   }
   pthread_cond_destroy(&c->cond);
   pthread_mutex_destroy(&c->lock);
   free(c);
   kdev->csfq = NULL;
}

/* SAME_VA memory for the rings and seqnos: the anchor mapping is the GPU VA and the free. */
static void *
kbase_csf_alloc(struct kbase_kmod_dev *kdev, uint64_t size, uint64_t flags, uint64_t *va)
{
   const uint64_t pages = DIV_ROUND_UP(size, 4096);
   union kbase_ioctl_mem_alloc req = {
      .in = {.va_pages = pages, .commit_pages = pages, .flags = flags},
   };
   if (pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_MEM_ALLOC, &req)) {
      mesa_loge("kbase: CSF MEM_ALLOC of %" PRIu64 " pages failed (err=%d)", pages, errno);
      return NULL;
   }
   void *p = mmap(NULL, pages * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, kdev->base.fd,
                  (off_t)req.out.gpu_va);
   if (p == MAP_FAILED) {
      mesa_loge("kbase: CSF mapping of cookie 0x%" PRIx64 " failed (err=%d)",
                (uint64_t)req.out.gpu_va, errno);
      return NULL;
   }
   *va = (uint64_t)(uintptr_t)p;
   return p;
}

/* Under c->lock: completes, in order, the leading submissions of g whose queues are all past
 * their targets (all of them if the group faulted). Returns them in out[] for the caller to run
 * outside the lock. */
static unsigned
kbase_csf_collect(struct pan_kmod_kbase_csf_group *g, struct kbase_csf_sub *out, unsigned max)
{
   unsigned n = 0;
   while (g->nr_subs && n < max) {
      struct kbase_csf_sub *s = &g->subs[g->first_sub];
      bool done = true;
      for (unsigned q = 0; q < g->nr_queues && done && !g->faulted; q++) {
         if (!(s->mask & BITFIELD_BIT(q)))
            continue;
         done = g->seqno[q * 8] >= s->target[q];
      }
      if (!done)
         break;
      out[n++] = *s;
      g->first_sub = (g->first_sub + 1) % KBASE_CSF_MAX_SUBS;
      g->nr_subs--;
   }
   return n;
}

static void *
kbase_csf_thread(void *arg)
{
   struct kbase_kmod_dev *kdev = arg;
   struct kbase_csf_dev *c = kdev->csfq;
   /* KBASE_CSF_DEBUG=1: the state of every ring about once a second while work is pending. */
   const bool dbg = debug_get_bool_option("KBASE_CSF_DEBUG", false);
   unsigned polls = 0;
   /* KBASE_CHECK_ANCHORS=1: check our anchors every 2 s (kbase_live_check); KBASE_SCAN_PTRS=1:
    * scan for CPU pointers every 5 s (kbase_scan_cpu_pointers). Both run here, on the thread that
    * completes submissions, and both cost time with many BOs (the anchor check took 53 ms with
    * 11600 mappings in Deep Rock Galactic), so neither comes with KBASE_LOG_FILE. */
   const bool check_anchors = debug_get_bool_option("KBASE_CHECK_ANCHORS", false);
   const bool scan_ptrs = debug_get_bool_option("KBASE_SCAN_PTRS", false);
   uint64_t next_check = 0, next_scan = kbase_now_ns() + 5000000000ull;

   for (;;) {
      if (check_anchors) {
         const uint64_t now = kbase_now_ns();
         if (now >= next_check) {
            kbase_live_check();
            next_check = now + 2000000000ull;
         }
      }
      if (scan_ptrs) {
         const uint64_t now = kbase_now_ns();
         if (now >= next_scan) {
            kbase_scan_cpu_pointers(0);
            next_scan = now + 5000000000ull;
         }
      }

      pthread_mutex_lock(&c->lock);
      bool busy = false;
      list_for_each_entry(struct pan_kmod_kbase_csf_group, g, &c->groups, link)
         busy |= g->nr_subs != 0;
      const bool stop = c->stop && !busy;
      pthread_mutex_unlock(&c->lock);
      if (stop)
         break;

      /* The event is a wakeup, not the completion record: the seqnos are. A short timeout
       * covers an update whose event was consumed before its submission was recorded. */
      struct pollfd pfd = {.fd = kdev->base.fd, .events = POLLIN};
      if (poll(&pfd, 1, busy ? 2 : 50) > 0 && (pfd.revents & POLLIN)) {
         uint8_t notes[KBASE_CSF_NOTIFICATION_SIZE * 8];
         const ssize_t got = read(kdev->base.fd, notes, sizeof(notes));
         for (ssize_t o = 0; o + KBASE_CSF_NOTIFICATION_SIZE <= got;
              o += KBASE_CSF_NOTIFICATION_SIZE) {
            if (notes[o] != KBASE_CSF_NOTIFICATION_GPU_QUEUE_GROUP_ERROR)
               continue;
            /* struct base_csf_notification: the group handle at 8, then a
             * base_gpu_queue_group_error: error_type at 16, sideband at 24, status at 32 (the
             * exception type in its low byte), and for a queue error the CS index at 36. */
            static const char *const kinds[] = {"group fault", "queue fault",
                                                "progress timeout", "tiler heap out of memory"};
            const uint8_t handle = notes[o + 8], kind = notes[o + 16];
            uint64_t sideband;
            uint32_t status;
            memcpy(&sideband, &notes[o + 24], sizeof(sideband));
            memcpy(&status, &notes[o + 32], sizeof(status));
            mesa_loge("kbase: queue group %u: %s (type %u), status 0x%08x (exception 0x%02x), "
                      "sideband 0x%016" PRIx64 ", queue %u",
                      handle, kind < ARRAY_SIZE(kinds) ? kinds[kind] : "unknown", kind, status,
                      status & 0xff, sideband, kind == 1 ? notes[o + 36] : 0);
            kbase_log_persist("queue group %u: %s (type %u), status 0x%08x, sideband 0x%016" PRIx64,
                              handle, kind < ARRAY_SIZE(kinds) ? kinds[kind] : "unknown", kind,
                              status, sideband);
            /* A group fault's sideband is the faulting GPU address (kbase_csf_ctx_handle_fault):
             * say whether it was ours, and whether we had freed it. */
            if (kind == 0 && sideband) {
               char what[480];
               kbase_bo_history_explain(sideband, what, sizeof(what));
               mesa_loge("kbase: fault %s", what);
               kbase_log_persist("fault %s", what);
               char tail[16 * 17 + 1];
               kbase_live_explain(sideband, what, sizeof(what), tail, sizeof(tail));
               mesa_loge("kbase: fault address now: %s", what);
               kbase_log_persist("fault address now: %s", what);
               if (tail[0]) {
                  mesa_loge("kbase:   its last bytes: %s", tail);
                  kbase_log_persist("  its last bytes: %s", tail);
               }
               const uint64_t around = 256ull << 10;
               kbase_log_maps_range(sideband > around ? sideband - around : 0,
                                    sideband + around, 16);
               /* Before the group is marked faulted: until then nothing completes, so the
                * command buffers that faulted are still intact for the scan. */
               kbase_explain_cpu_view(sideband);
               kbase_scan_cpu_pointers(sideband);
            }
            pthread_mutex_lock(&c->lock);
            list_for_each_entry(struct pan_kmod_kbase_csf_group, g, &c->groups, link) {
               if (g->handle == handle)
                  g->faulted = true;
            }
            /* The exported fences wait on this group's seqnos, which a faulted group never
             * advances: drain them now rather than at the kernel's timeout. */
            kbase_csf_fence_watchdog(kdev, c, "a queue group faulted");
            pthread_mutex_unlock(&c->lock);
         }
      }

      if (dbg && busy && ++polls % 500 == 0) {
         pthread_mutex_lock(&c->lock);
         list_for_each_entry(struct pan_kmod_kbase_csf_group, g, &c->groups, link) {
            for (unsigned q = 0; q < g->nr_queues; q++) {
               mesa_logi("kbase csf: group %u queue %u: seqno %" PRIu64 "/%" PRIu64
                         " insert %" PRIu64 " extract %" PRIu64 " active %" PRIu64
                         " subs %u faulted %d error 0x%x",
                         g->handle, q, (uint64_t)g->seqno[q * 8], g->ring[q].signals,
                         g->ring[q].insert, (uint64_t)g->ring[q].output[0],
                         (uint64_t)g->ring[q].output[1], g->nr_subs, g->faulted,
                         ((volatile uint32_t *)&g->seqno[q * 8])[2]);
            }
         }
         pthread_mutex_unlock(&c->lock);
      }

      /* Collected under the lock, run outside it; done_seq moves only after the callbacks, so
       * a wait (and a group destroy) returns once the signals are in. */
      struct kbase_csf_sub fin[64];
      struct pan_kmod_kbase_csf_group *fin_group[64];
      bool fin_ok[64];
      unsigned n = 0;
      pthread_mutex_lock(&c->lock);
      list_for_each_entry(struct pan_kmod_kbase_csf_group, g, &c->groups, link) {
         const unsigned k = kbase_csf_collect(g, fin + n, ARRAY_SIZE(fin) - n);
         if (dbg && k) {
            for (unsigned q = 0; q < g->nr_queues; q++)
               mesa_logi("kbase csf: group %u queue %u done to seqno %" PRIu64 ", error 0x%x",
                         g->handle, q, (uint64_t)g->seqno[q * 8],
                         ((volatile uint32_t *)&g->seqno[q * 8])[2]);
         }
         for (unsigned i = 0; i < k; i++) {
            fin_group[n + i] = g;
            fin_ok[n + i] = !g->faulted;
         }
         n += k;
      }
      kbase_csf_fence_watchdog(kdev, c, NULL);
      pthread_mutex_unlock(&c->lock);

      for (unsigned i = 0; i < n; i++) {
         if (fin[i].done)
            fin[i].done(fin[i].data, fin_ok[i]);
      }
      if (n) {
         pthread_mutex_lock(&c->lock);
         for (unsigned i = 0; i < n; i++)
            fin_group[i]->done_seq = MAX2(fin_group[i]->done_seq, fin[i].seq);
         pthread_cond_broadcast(&c->cond);
         pthread_mutex_unlock(&c->lock);
      }
   }
   return NULL;
}

static void kbase_csf_emit(struct kbase_csf_ring *r, uint64_t ins);

struct pan_kmod_kbase_csf_group *
pan_kmod_kbase_csf_group_create(struct pan_kmod_dev *dev, unsigned nr_queues,
                                uint64_t compute_mask, uint64_t fragment_mask,
                                uint8_t max_compute, uint8_t max_fragment, uint8_t priority)
{
   struct kbase_kmod_dev *kdev = container_of(dev, struct kbase_kmod_dev, base);
   struct kbase_csf_dev *c = kdev->csfq;

   if (!c || nr_queues > PAN_KMOD_KBASE_CSF_MAX_QUEUES)
      return NULL;

   struct pan_kmod_kbase_csf_group *g = calloc(1, sizeof(*g));
   if (!g)
      return NULL;
   g->kdev = kdev;
   g->nr_queues = nr_queues;
   g->next_seq = 1;

   g->seqno = kbase_csf_alloc(kdev, 4096, 0xfull | BASE_MEM_SAME_VA | BASE_MEM_CSF_EVENT,
                              &g->seq_va);
   if (!g->seqno)
      goto err_free;
   for (unsigned q = 0; q < nr_queues; q++)
      g->seqno[q * 8] = 0;

   g->dvs_size =
      (util_last_bit64(dev->props.shader_present) * KBASE_CSF_DVS_PAGES_PER_CORE + 1) * 4096ull;
   g->dvs = kbase_csf_alloc(kdev, g->dvs_size, 0xfull | BASE_MEM_SAME_VA, &g->dvs_va);
   if (!g->dvs)
      goto err_free_seq;

   union kbase_ioctl_cs_queue_group_create gc = {
      .in = {
         .tiler_mask = 1,
         .fragment_mask = fragment_mask,
         .compute_mask = compute_mask,
         .cs_min = nr_queues,
         .priority = priority,
         .tiler_max = 1,
         .fragment_max = max_fragment,
         .compute_max = max_compute,
         .dvs_buf = g->dvs_va | KBASE_CSF_DVS_PAGES_PER_CORE,
      },
   };
   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_QUEUE_GROUP_CREATE, &gc)) {
      mesa_loge("kbase: CS_QUEUE_GROUP_CREATE failed (err=%d)", errno);
      goto err_free_dvs;
   }
   g->handle = gc.out.group_handle;

   for (unsigned q = 0; q < nr_queues; q++) {
      struct kbase_csf_ring *r = &g->ring[q];
      /* GPU-uncached, as panthor maps its rings: the CPU writes them while the GPU runs them,
       * so no ring line may be served stale from the GPU's L2. See also the cache line padding
       * in pan_kmod_kbase_csf_submit. */
      r->cpu = kbase_csf_alloc(kdev, KBASE_CSF_RING_SIZE,
                               0xfull | BASE_MEM_SAME_VA | BASE_MEM_UNCACHED_GPU, &r->va);
      if (!r->cpu)
         goto err_term;
      memset(r->cpu, 0, KBASE_CSF_RING_SIZE);

      struct kbase_ioctl_cs_queue_register reg = {
         .buffer_gpu_addr = r->va,
         .buffer_size = KBASE_CSF_RING_SIZE,
         .priority = 1,
      };
      union kbase_ioctl_cs_queue_bind bind = {
         .in = {.buffer_gpu_addr = r->va, .group_handle = g->handle, .csi_index = q},
      };
      if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_QUEUE_REGISTER, &reg) ||
          pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_QUEUE_BIND, &bind)) {
         mesa_loge("kbase: queue %u register/bind failed (err=%d)", q, errno);
         goto err_term;
      }
      r->io = mmap(NULL, 3 * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                   (off_t)bind.out.mmap_handle);
      if (r->io == MAP_FAILED) {
         r->io = NULL;
         mesa_loge("kbase: queue %u user IO pages could not be mapped (err=%d)", q, errno);
         goto err_term;
      }
      /* Page 0 is the doorbell, 1 the input page, 2 the output page. */
      r->input = (volatile uint64_t *)((uint8_t *)r->io + 4096);
      r->output = (volatile const uint64_t *)((uint8_t *)r->io + 2 * 4096);
   }

   /* PanVK's streams assume every CS register starts at zero (its subqueue seqno registers
    * are compared against sync objects initialised to 1). Nothing guarantees that for a kbase
    * queue, so the first thing each ring does is zero them all. */
   for (unsigned q = 0; q < nr_queues; q++) {
      struct kbase_csf_ring *r = &g->ring[q];
      for (unsigned reg = 0; reg < kdev->csif.cs_reg_count; reg += 2)
         kbase_csf_emit(r, cs_ins(CS_OP_MOVE48, (uint64_t)reg << 48));
   }

   pthread_mutex_lock(&c->lock);
   list_addtail(&g->link, &c->groups);
   if (!c->thread_started) {
      if (pthread_create(&c->thread, NULL, kbase_csf_thread, kdev) == 0)
         c->thread_started = true;
      else
         mesa_loge("kbase: could not start the CSF event thread");
   }
   pthread_mutex_unlock(&c->lock);
   if (!c->thread_started) {
      pthread_mutex_lock(&c->lock);
      list_del(&g->link);
      pthread_mutex_unlock(&c->lock);
      goto err_term;
   }
   return g;

err_term:
   for (unsigned q = 0; q < nr_queues; q++) {
      struct kbase_csf_ring *r = &g->ring[q];
      if (r->io)
         munmap(r->io, 3 * 4096);
   }
   {
      struct kbase_ioctl_cs_queue_group_term t = {.group_handle = g->handle};
      pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_QUEUE_GROUP_TERMINATE, &t);
   }
   for (unsigned q = 0; q < nr_queues; q++) {
      struct kbase_csf_ring *r = &g->ring[q];
      if (r->cpu) {
         struct kbase_ioctl_cs_queue_addr a = {.buffer_gpu_addr = r->va};
         pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_QUEUE_TERMINATE, &a);
         munmap(r->cpu, KBASE_CSF_RING_SIZE);
      }
   }
err_free_dvs:
   munmap(g->dvs, g->dvs_size);
err_free_seq:
   munmap((void *)g->seqno, 4096);
err_free:
   free(g);
   return NULL;
}

void
pan_kmod_kbase_csf_group_destroy(struct pan_kmod_kbase_csf_group *g)
{
   if (!g)
      return;

   struct kbase_kmod_dev *kdev = g->kdev;
   struct kbase_csf_dev *c = kdev->csfq;

   /* Whatever is still in flight completes (or the group faults) before the rings go. */
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   pan_kmod_kbase_csf_wait(g, 0, (int64_t)now.tv_sec * 1000000000ll + now.tv_nsec +
                                    5000000000ll);

   pthread_mutex_lock(&c->lock);
   list_del(&g->link);
   /* The watchdog reads its exports' groups: anything still pending on this one is drained
    * first, and none of its exports outlive it. */
   bool pending = false;
   for (unsigned i = 0; i < c->nr_exports; i++)
      pending |= c->exports[i].g == g;
   if (pending) {
      g->faulted = true;
      kbase_csf_fence_watchdog(kdev, c, "its queue group is being destroyed");
   }
   unsigned keep = 0;
   for (unsigned i = 0; i < c->nr_exports; i++) {
      if (c->exports[i].g != g)
         c->exports[keep++] = c->exports[i];
   }
   c->nr_exports = keep;
   pthread_mutex_unlock(&c->lock);

   struct kbase_ioctl_cs_queue_group_term t = {.group_handle = g->handle};
   pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_CS_QUEUE_GROUP_TERMINATE, &t);
   for (unsigned q = 0; q < g->nr_queues; q++) {
      struct kbase_csf_ring *r = &g->ring[q];
      munmap(r->io, 3 * 4096);
      struct kbase_ioctl_cs_queue_addr a = {.buffer_gpu_addr = r->va};
      pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_CS_QUEUE_TERMINATE, &a);
      munmap(r->cpu, KBASE_CSF_RING_SIZE);
   }
   munmap(g->dvs, g->dvs_size);
   munmap((void *)g->seqno, 4096);
   free(g);
}

bool
pan_kmod_kbase_csf_tiler_heap_create(struct pan_kmod_dev *dev, uint32_t chunk_size,
                                     uint32_t initial_chunks, uint32_t max_chunks,
                                     uint16_t target_in_flight, uint64_t *heap_ctx_va,
                                     uint64_t *first_chunk_va)
{
   union kbase_ioctl_cs_tiler_heap_init req = {
      .in = {
         .chunk_size = chunk_size,
         .initial_chunks = initial_chunks,
         .max_chunks = max_chunks,
         .target_in_flight = target_in_flight,
         .group_id = kbase_mem_group(),
      },
   };
   const union kbase_ioctl_cs_tiler_heap_init first = req;
   int ret = pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_TILER_HEAP_INIT, &req);
   if (ret && first.in.group_id) {
      req = first;
      req.in.group_id = 0;
      ret = pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_TILER_HEAP_INIT, &req);
      if (!ret)
         kbase_mem_group_refused();
   }
   if (ret) {
      mesa_loge("kbase: CS_TILER_HEAP_INIT failed (err=%d)", errno);
      return false;
   }
   *heap_ctx_va = req.out.gpu_heap_va;
   *first_chunk_va = req.out.first_chunk_va;
   return true;
}

void
pan_kmod_kbase_csf_tiler_heap_destroy(struct pan_kmod_dev *dev, uint64_t heap_ctx_va)
{
   struct kbase_ioctl_cs_tiler_heap_term req = {.gpu_heap_va = heap_ctx_va};
   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_CS_TILER_HEAP_TERM, &req))
      mesa_loge("kbase: CS_TILER_HEAP_TERM failed (err=%d)", errno);
}

/* Room for `bytes` more in the ring: waits for the firmware's CS_EXTRACT to move on. */
static bool
kbase_csf_ring_reserve(struct pan_kmod_kbase_csf_group *g, struct kbase_csf_ring *r,
                       unsigned bytes)
{
   for (unsigned spins = 0; r->insert + bytes - r->output[0] > KBASE_CSF_RING_SIZE; spins++) {
      if (g->faulted)
         return false;
      if (spins > 200000) {
         mesa_loge("kbase: a ring stayed full for 10 s (insert %" PRIu64 ", extract %" PRIu64 ")",
                   r->insert, (uint64_t)r->output[0]);
         return false;
      }
      usleep(50);
   }
   return true;
}

static void
kbase_csf_emit(struct kbase_csf_ring *r, uint64_t ins)
{
   r->cpu[(r->insert / 8) % (KBASE_CSF_RING_SIZE / 8)] = ins;
   r->insert += 8;
}

/* Ring instructions that wait until queue w of the group has signaled target times, through
 * registers A (address) and V (value). */
static void
kbase_csf_emit_seq_wait(struct kbase_csf_ring *r, const struct pan_kmod_kbase_csf_group *g,
                        unsigned w, uint64_t target, uint64_t A, uint64_t V)
{
   kbase_csf_emit(r, cs_ins(CS_OP_MOVE48, (A << 48) | (g->seq_va + w * 64)));
   kbase_csf_emit(r, cs_ins(CS_OP_MOVE48, (V << 48) | (target - 1)));
   kbase_csf_emit(r, cs_ins(CS_OP_SYNC_WAIT64, (A << 40) | (V << 32) | (1u << 28)));
}

uint64_t
pan_kmod_kbase_csf_submit(struct pan_kmod_kbase_csf_group *g,
                          const struct pan_kmod_kbase_cs_call *calls, unsigned nr_calls,
                          uint32_t signal_queue_mask, uint64_t wait_seq,
                          const uint32_t *prior_waits, pan_kmod_kbase_done_cb done, void *data)
{
   struct kbase_kmod_dev *kdev = g->kdev;
   struct kbase_csf_dev *c = kdev->csfq;
   const unsigned K = kdev->csif.cs_reg_count - 4;
   const uint64_t A = (uint64_t)K, V = (uint64_t)K + 2;

   uint32_t mask = signal_queue_mask;
   for (unsigned i = 0; i < nr_calls; i++)
      mask |= BITFIELD_BIT(calls[i].queue);
   mask &= BITFIELD_MASK(g->nr_queues);

   pthread_mutex_lock(&c->lock);
   while (g->nr_subs == KBASE_CSF_MAX_SUBS && !g->faulted)
      pthread_cond_wait(&c->cond, &c->lock);
   const bool faulted = g->faulted;
   /* The submission to wait for, if it is still pending: every queue it signals. */
   uint32_t wait_mask = 0;
   uint64_t wait_target[PAN_KMOD_KBASE_CSF_MAX_QUEUES];
   for (unsigned i = 0; wait_seq && i < g->nr_subs; i++) {
      const struct kbase_csf_sub *ws = &g->subs[(g->first_sub + i) % KBASE_CSF_MAX_SUBS];
      if (ws->seq == wait_seq) {
         wait_mask = ws->mask;
         memcpy(wait_target, ws->target, sizeof(wait_target));
      }
   }
   pthread_mutex_unlock(&c->lock);
   if (faulted || !mask)
      return 0;

   /* Same-group semaphore waits, on the GPU: each queue of this submission first waits until
    * every queue of the waited one has signaled. */
   if (wait_mask) {
      u_foreach_bit(q, mask) {
         struct kbase_csf_ring *r = &g->ring[q];
         if (!kbase_csf_ring_reserve(g, r, 3 * 8 * PAN_KMOD_KBASE_CSF_MAX_QUEUES))
            return 0;
         u_foreach_bit(w, wait_mask)
            kbase_csf_emit_seq_wait(r, g, w, wait_target[w], A, V);
      }
   }

   /* The caller's waits on earlier work of other queues of this group, also on the GPU. */
   if (prior_waits) {
      u_foreach_bit(q, mask) {
         const uint32_t ws = prior_waits[q] & BITFIELD_MASK(g->nr_queues);
         if (!ws)
            continue;
         struct kbase_csf_ring *r = &g->ring[q];
         if (!kbase_csf_ring_reserve(g, r, 3 * 8 * PAN_KMOD_KBASE_CSF_MAX_QUEUES))
            return 0;
         u_foreach_bit(w, ws) {
            if (g->ring[w].signals)
               kbase_csf_emit_seq_wait(r, g, w, g->ring[w].signals, A, V);
         }
      }
   }

   /* The rings are only written from here, one submission at a time per group: the caller
    * (a vk_queue) serialises submissions. */
   for (unsigned i = 0; i < nr_calls; i++) {
      if (!calls[i].size)
         continue;
      struct kbase_csf_ring *r = &g->ring[calls[i].queue];
      if (!kbase_csf_ring_reserve(g, r, 6 * 8))
         return 0;
      kbase_csf_emit(r, cs_ins(CS_OP_MOVE32, V << 48));
      kbase_csf_emit(r, cs_ins(CS_OP_FLUSH_CACHE2,
                               (V << 40) | (2u << 8) | (3u << 4) | 3u)); /* clean+inv, slot 0 */
      kbase_csf_emit(r, cs_ins(CS_OP_MOVE48, (A << 48) | calls[i].addr));
      kbase_csf_emit(r, cs_ins(CS_OP_MOVE32, (V << 48) | calls[i].size));
      kbase_csf_emit(r, cs_ins(CS_OP_WAIT, 1ull << 16));
      kbase_csf_emit(r, cs_ins(CS_OP_CALL, (A << 40) | (V << 32)));
   }

   struct kbase_csf_sub sub = {.mask = mask, .done = done, .data = data};
   u_foreach_bit(q, mask) {
      struct kbase_csf_ring *r = &g->ring[q];
      /* After a fault the firmware skips everything up to the next ERROR_BARRIER, so a signal
       * before it never lands and the submission would never complete (measured: CS_EXTRACT at
       * CS_INSERT, the seqno one short). The barrier goes before the signal. */
      if (!kbase_csf_ring_reserve(g, r, 16 * 8))
         return 0;
      kbase_csf_emit(r, cs_ins(CS_OP_MOVE32, V << 48));
      kbase_csf_emit(r, cs_ins(CS_OP_FLUSH_CACHE2,
                               (V << 40) | (0xffffull << 16) | (1u << 4) | 1u)); /* clean */
      kbase_csf_emit(r, cs_ins(CS_OP_WAIT, 0xffffull << 16));
      kbase_csf_emit(r, cs_ins(CS_OP_ERROR_BARRIER, 0));
      kbase_csf_emit(r, cs_ins(CS_OP_MOVE48, (A << 48) | (g->seq_va + q * 64)));
      kbase_csf_emit(r, cs_ins(CS_OP_MOVE48, (V << 48) | 1));
      kbase_csf_emit(r, cs_ins(CS_OP_SYNC_ADD64, (A << 40) | (V << 32) | 1u)); /* propagate */
      kbase_csf_emit(r, cs_ins(CS_OP_WAIT, 0));
      /* Up to the next cache line with NOPs, as panthor pads its jobs: the command frontend's
       * prefetcher reads whole 64-byte lines, and the part of a line written after it read the
       * line is never seen. Measured on the Mali-G720: a submission whose first instructions
       * shared a line with the previous one's end ran without its CALLs (the subqueue sync
       * objects skipped the frame), and the next one then waited forever for it. */
      while (r->insert % 64)
         kbase_csf_emit(r, 0);
      sub.target[q] = ++r->signals;
   }

   pthread_mutex_lock(&c->lock);
   sub.seq = g->next_seq++;
   g->subs[(g->first_sub + g->nr_subs) % KBASE_CSF_MAX_SUBS] = sub;
   g->nr_subs++;
   pthread_mutex_unlock(&c->lock);

   __sync_synchronize();
   u_foreach_bit(q, mask) {
      struct kbase_csf_ring *r = &g->ring[q];
      r->input[0] = r->insert;
   }
   __sync_synchronize();
   u_foreach_bit(q, mask) {
      struct kbase_ioctl_cs_queue_addr k = {.buffer_gpu_addr = g->ring[q].va};
      if (pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_CS_QUEUE_KICK, &k))
         mesa_loge("kbase: CS_QUEUE_KICK of queue %u failed (err=%d)", q, errno);
   }
   return sub.seq;
}

bool
pan_kmod_kbase_csf_wait(struct pan_kmod_kbase_csf_group *g, uint64_t seq, int64_t abs_timeout_ns)
{
   struct kbase_csf_dev *c = g->kdev->csfq;
   bool done = false;

   pthread_mutex_lock(&c->lock);
   if (!seq)
      seq = g->next_seq - 1;
   for (;;) {
      if (g->done_seq >= seq) {
         done = true;
         break;
      }
      if (abs_timeout_ns == INT64_MAX) {
         pthread_cond_wait(&c->cond, &c->lock);
         continue;
      }
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      if ((int64_t)now.tv_sec * 1000000000ll + now.tv_nsec >= abs_timeout_ns)
         break;
      const struct timespec until = {.tv_sec = abs_timeout_ns / 1000000000ll,
                                     .tv_nsec = abs_timeout_ns % 1000000000ll};
      pthread_cond_timedwait(&c->cond, &c->lock, &until);
   }
   pthread_mutex_unlock(&c->lock);
   return done;
}

int
pan_kmod_kbase_csf_fence_fd(struct pan_kmod_kbase_csf_group *g, uint64_t seq)
{
   struct kbase_kmod_dev *kdev = g->kdev;
   struct kbase_csf_dev *c = kdev->csfq;
   struct base_cqs_wait_operation_info objs[PAN_KMOD_KBASE_CSF_MAX_QUEUES];
   unsigned nr_objs = 0;

   /* Under the lock throughout, enqueues included (they do not block): the watchdog may delete
    * the queue, and a KCPU queue id is reused as soon as it is free. */
   pthread_mutex_lock(&c->lock);
   /* A faulted group's seqnos never move: a wait on them would sit in the KCPU queue until the
    * kernel's fence-signal timeout, which panics this device (kbase_csf_fence_watchdog). The
    * work is lost either way, so there is nothing to wait for. */
   if (g->faulted) {
      pthread_mutex_unlock(&c->lock);
      return -1;
   }
   /* Still pending? Then wait for every queue it signals to pass its target. */
   const struct kbase_csf_sub *sub = NULL;
   for (unsigned i = 0; i < g->nr_subs && !sub; i++) {
      const struct kbase_csf_sub *s = &g->subs[(g->first_sub + i) % KBASE_CSF_MAX_SUBS];
      if (s->seq == seq)
         sub = s;
   }
   if (sub) {
      u_foreach_bit(q, sub->mask) {
         objs[nr_objs++] = (struct base_cqs_wait_operation_info){
            .addr = g->seq_va + q * 64,
            .val = sub->target[q] - 1,
            .operation = 1,
            .data_type = 1,
         };
      }
   }
   if (!nr_objs) {
      pthread_mutex_unlock(&c->lock);
      if (getenv("KBASE_FENCE_DEBUG"))
         mesa_logi("kbase fence: seq %" PRIu64 " done already", seq);
      return -1; /* done already */
   }
   if (c->nr_exports == KBASE_CSF_MAX_EXPORTS)
      kbase_csf_fence_watchdog(kdev, c, "too many fences were waiting");
   if (!c->kcpu_created) {
      struct kbase_ioctl_kcpu_queue_new nq = {0};
      if (pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_KCPU_QUEUE_CREATE, &nq)) {
         pthread_mutex_unlock(&c->lock);
         mesa_loge("kbase: KCPU_QUEUE_CREATE failed (err=%d)", errno);
         return -1;
      }
      c->kcpu_id = nq.id;
      c->kcpu_created = true;
   }

   struct base_fence fence = {.basep = {.fd = -1, .stream_fd = -1}};
   struct base_kcpu_command cmds[2] = {
      {.type = BASE_KCPU_COMMAND_TYPE_CQS_WAIT_OPERATION,
       .info.cqs_wait_operation = {.objs = (uintptr_t)objs, .nr_objs = nr_objs}},
      {.type = BASE_KCPU_COMMAND_TYPE_FENCE_SIGNAL, .info.fence = {.fence = (uintptr_t)&fence}},
   };
   /* One command per enqueue: this kernel refuses more (EINVAL). */
   for (unsigned i = 0; i < 2; i++) {
      struct kbase_ioctl_kcpu_queue_enqueue enq = {
         .addr = (uintptr_t)&cmds[i], .nr_commands = 1, .id = c->kcpu_id};
      if (pan_kmod_ioctl(kdev->base.fd, KBASE_IOCTL_KCPU_QUEUE_ENQUEUE, &enq)) {
         const int err = errno;
         /* A wait without its signal (the second enqueue failed) would sit there until the
          * timeout: drain it now. */
         if (i == 1)
            kbase_csf_fence_watchdog(kdev, c, "a fence enqueue failed");
         pthread_mutex_unlock(&c->lock);
         mesa_loge("kbase: KCPU_QUEUE_ENQUEUE failed (err=%d)", err);
         return -1;
      }
   }
   /* Tracked from now: the kernel's 2 s start no earlier than the enqueue. */
   c->exports[c->nr_exports++] = (struct kbase_csf_export){
      .g = g, .t_ns = kbase_now_ns(), .mask = sub->mask};
   memcpy(c->exports[c->nr_exports - 1].target, sub->target, sizeof(sub->target));
   pthread_mutex_unlock(&c->lock);
   if (getenv("KBASE_FENCE_DEBUG"))
      mesa_logi("kbase fence: seq %" PRIu64 " objs %u fd %d", seq, nr_objs, fence.basep.fd);
   return fence.basep.fd;
}

bool
pan_kmod_kbase_csf_group_faulted(const struct pan_kmod_kbase_csf_group *g)
{
   return g->faulted;
}


static uint64_t
kbase_kmod_query_timestamp(const struct pan_kmod_dev *dev)
{
#if defined(__aarch64__)
   uint64_t v;
   __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
   return v;
#else
   return 0;
#endif
}

/* Cache maintenance of WB_MMAP mappings is done from userspace (pan_kmod_queue_bo_map_sync), so
 * nothing is ever queued for the kernel. */
static int
kbase_kmod_flush_bo_map_syncs(struct pan_kmod_dev *dev)
{
   return 0;
}

const struct pan_kmod_ops kbase_kmod_ops = {
   .dev_create = kbase_kmod_dev_create,
   .dev_destroy = kbase_kmod_dev_destroy,
   .dev_query_user_va_range = kbase_kmod_dev_query_user_va_range,
   .bo_alloc = kbase_kmod_bo_alloc,
   .bo_free = kbase_kmod_bo_free,
   .bo_import = kbase_kmod_bo_import,
   .bo_export = kbase_kmod_bo_export,
   .bo_mmap = kbase_kmod_bo_mmap,
   .bo_munmap = kbase_kmod_bo_munmap,
   .bo_set_label = kbase_kmod_bo_set_label,
   .bo_wait = kbase_kmod_bo_wait,
   .vm_create = kbase_kmod_vm_create,
   .vm_destroy = kbase_kmod_vm_destroy,
   .vm_bind = kbase_kmod_vm_bind,
   .query_timestamp = kbase_kmod_query_timestamp,
   .flush_bo_map_syncs = kbase_kmod_flush_bo_map_syncs,
};
