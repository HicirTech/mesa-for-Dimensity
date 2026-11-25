

Not a conformant Vulkan implementation; for testing.

## Where the code comes from

- Upstream Mesa `main` (https://gitlab.freedesktop.org/mesa/mesa).
- The kbase backend (job manager and CSF), the PanVK queue and sync changes,
  gralloc/AHardwareBuffer import, BCn emulation, geometry, tessellation and
  transform feedback on compute, and the Android build and packaging scripts
  come from [mali-panvk](https://github.com/JimVulkan/mali-panvk) by
  JimVulkan at `58f2c55`,
## Building

On Linux or WSL:

```sh
./build.sh /path/to/android-ndk
```

Requirements are those listed in `android/README-mali-panvk.md` (Android NDK
r29 tested, meson, ninja, Python with mako, packaging and pyyaml,
glslangValidator, and LLVM with clang, libclc and the SPIR-V LLVM
translator for the host tools), plus `rustc` with the
`aarch64-linux-android` target and `bindgen` for the Kraid compiler.
`KRAID=0` builds without Kraid and without Rust.

The result is `dist/panvk-9300-<version>.zip` with `meta.json`,
`libvulkan_panfrost.so` and `NOTICE.txt`, the package format of
adrenotools-style driver loaders.

## Runtime switches

| Environment | Effect |
|---|---|
| `PAN_USE_KRAID` | Defaults to all stages when built with Rust support; set to `0` to disable or `cs`, `fs`, `vs`, or `internal` to select stages, see `src/panfrost/compiler/pan_compiler.c` |
| `PANVK_DEBUG=startup` | logs why the device could not be created |
| `PANVK_FORCE_IDVS=1` | turns index-driven vertex shading back on (off by default on Bifrost) |
| `PANVK_NO_MEDIUMP=1` | compiles `mediump` shader code at 32 bits |
| `PANVK_PACING=0` | disables per-swapchain EMA frame pacing before WSI present (enabled by default) |
| `PANVK_PACING_HZ` | sets the pacing refresh-rate fallback in Hz (default `60`) |
| `PANVK_PACING_DEBUG=1` | logs EMA, sleep delay, timing source, and pacing decision |
| `MESA_SHADER_CACHE_DISABLE=true` | turns the disk shader cache off |
| `MESA_LOG=file` | writes the driver's log, including GPU fault details, to a file |

`VK_EXT_present_timing` is available on WSI-enabled builds across PanVK
architectures that support GPU timestamps. Surface-specific timing modes are
reported by Mesa's WSI backend. When enabled, pacing observes timing results
returned to the application; CPU monotonic deltas are used until timing results
are available.

## License

Mesa files keep the licenses in their headers (see `docs/license.rst`). The
changes imported from mali-panvk are MIT (`LICENSE`); the AOSP gralloc
headers are Apache-2.0 (`LICENSES/`, `THIRD_PARTY_NOTICES.md`). Driver
packages carry `android/NOTICE.txt`, which covers everything compiled into
`libvulkan_panfrost.so`.
