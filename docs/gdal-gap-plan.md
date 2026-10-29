# What geoplumb is missing against GDAL, and how to add it

A checked box is built and in `DESIGN.md`, an unchecked one is not. The engine design
(pull graph, kinded caps, resolution ladder, coalescing cache, disk tier,
invalidation, time as a pull parameter, async range reads) is finished and
described in `DESIGN.md`. What follows is the gap between that engine and a
raster and vector stack that needs no GDAL, ordered by coverage gained per
day of work. Estimates are from the design doc and the trait signatures, not
from a full read of every file, so take them as order of magnitude.

Where GDAL still is today, across the org:

- `geolang` (Python): `rasterio` and `pyogrio` for layer reads and
  `read_info`, the QGIS `gdal` provider under `run_qgis_algorithm`. `GDAL_SKIP`
  removes `VRT`, `OGR_VRT` and `GDALG` at import so an uploaded document
  cannot name a local file.
- `verne-gdb`: 317 lines over `gdal-sys` for the OpenFileGDB driver, behind
  the `gdb` feature. The owner decision of 2026-07-29 in
  `viewtopia/DESIGN_TODO.md`: GDAL stays behind one crate or adapter trait,
  feature-gated, shipped in a container with GDAL from the distro.
- geoplumb: nothing. Its sources are `stac`, `cog`, `geojson` and `las`.

Pure-Rust coverage today: terrano reads and writes COGs (deflate and raw,
horizontal and floating-point predictors, u8 through f64, multi-band pixel
interleaved), GRIB2 simple packing, NetCDF classic and 64-bit offset. geodukt
reads and writes GeoJSON, GeoPackage, Shapefile and CSV. projicio covers 5935
EPSG codes with Helmert and NTv2 grid shifts. nubis reads LAS and has an
octree.

## 1. A feature-gated `gdal` raster source

Two to three days. The item with the most coverage per day: every raster
format GDAL opens becomes a geoplumb layer, and the engine's cache, ladder
and halos apply unchanged.

Shape: a `GdalSrc` in `crates/geoplumb/src/elements/gdal.rs` implementing
`Source`, mirroring `CogSrc`.

- [x] `constraint`: `Produces` one raster pattern from the dataset. Dtype
      is `F64`, the one dtype the caps have, every band read as f64 whatever
      the file stores. Bands from the band count. CRS from the
      dataset's spatial reference as an EPSG authority code, resolved through
      projicio. A dataset whose SRS carries no EPSG code fails at open rather
      than guessing. Chunk size from the first band's block size, rounded to
      the ladder's power of two.
- [x] `grid`: origin and base resolution from the geotransform. A rotated or
      sheared geotransform fails at open, the ladder has no row for it.
- [x] `read`: pick the overview whose pixel width is nearest at or below
      the request, as `CogReader::select_level` does, read the window with
      `RasterBand::read_as` at that overview, block-average the remainder
      with the same code `CogSrc` uses. Nodata from `RasterBand::no_data_value`
      mapped to NaN. `read` is a `BoxFuture`, the GDAL call runs inside
      `spawn_blocking`. GDAL's `AVERAGE` overviews treat NaN as a value
      unless the file declares NaN as nodata, so a NaN-hole file without a
      nodata tag serves NaN at partial-hole overview pixels where
      `RasterSrc` averages the finite ones.
- [x] Concurrency: `gdal::Dataset` is `Send` and not `Sync` (gdal 0.18
      docs), so it cannot sit behind an `Arc` across tasks. Hold a pool of
      opened datasets per path, the pattern `StacSrc` uses for its per-asset
      readers (take one, open fresh when empty, put it back).
- [x] Feature `gdal` on the `geoplumb` crate pulling the `gdal` crate, off by
      default, so the default build and the wasm-adjacent crates keep no C
      dependency. The `gdal` crate's newest release is 0.19.0 (2025-12-23,
      MSRV 1.80) with bindings for GDAL 3.4 through 3.11. Pin the GDAL
      version the Dockerfile installs to one in that range (bookworm's
      3.6, behind the `GDAL=1` build arg).
- [x] `SourceConfig::Gdal { path, band: Option<u16> }` in `geoplumb-server`,
      behind the same feature, a layer naming it in a build without the
      feature failing at parse with a message that names the feature.
- [x] Dockerfile: a second stage or a build arg that installs `libgdal` from
      bookworm and builds with the feature. The default image stays GDAL
      free.
- [x] CI: the ubuntu leg adds a job with `libgdal-dev` and `--features gdal`.
      Windows and macOS stay on the default features.
- [x] Test: a striped (non-COG) GeoTIFF with an overview and a nodata value
      written by terrano, pulled at two ladder levels through `GdalSrc`,
      compared pixel for pixel against `RasterSrc` over the same file.

A vector counterpart, `OgrSrc` over `gdal::vector`, is a second two to three
days with the same pool and feature. It only pays once `VecSrc` reads
windowed (section 4), otherwise it holds the whole layer resident the way
`VecSrc` does.

## 2. `RangeRead` by shared reference

One to two days, most of it in terrano. `CogSrc` serializes every range read
on its file behind a mutex because `RangeRead::read_range` and
`CogReader::read_window` take `&mut self`. `HttpRange` is already
multiplexed internally, so the mutex is only the trait signature.

- [x] terrano: `RangeRead::read_range` and `read_ranges` take `&self`. The
      `std::fs::File` impl uses `read_exact_at` on unix and `seek_read` on
      windows instead of seek plus read. The `&[u8]` impl is unchanged. The
      test `RecordingReader` counts through a `Cell`.
- [x] terrano: `CogReader::open` keeps `&mut` on its own `source` only during
      open. `read_window` and `read_window_bands` take `&self`. `levels` and
      `meta` are read-only already.
- [x] geoplumb: `CogSrc` drops the `Arc<Mutex<_>>`, concurrent chunks of one
      pull read concurrently. `HttpRange` drops its `&mut`.
- [x] A test that pulls two chunks of one COG from two tasks and asserts the
      range reads interleave (the recording reader logs order).

This is a breaking change to terrano's public trait, so it lands as a
terrano release (0.3.0) and a geoplumb bump together. tiletopia's cog route
is adapted in its own commit and takes the change at its next lock bump.

## 3. Windowed point source

- [x] Plain LAS tile index. At open, point indices are binned by tile of
      the base grid into a `HashMap<(i64, i64), Vec<usize>>`, and a tile read
      scans the bins its bbox touches instead of the whole cloud. The cloud
      stays resident.
- [ ] COPC, about a week. A `CopcSrc` that reads the octree hierarchy and
      fetches only the nodes a window at a ladder level touches, over the
      same `RangeRead` transport as COGs, so a remote COPC over HTTP works.
      LAZ decoding is already in the org: tiletopia-server depends on the
      `las` crate 0.9 with its `laz` feature (`crates/tiletopia-server`),
      and the newest `las` is 0.11.2 (2026-10-27). `copc-rs` 0.5.0
      (2025-05-15) reads the COPC hierarchy. Neither has been evaluated
      against nubis's `read_las`. The COPC level maps onto the ladder level
      directly, which replaces the per-level voxel thinning for this source.

## 4. Windowed vector source

Two to three days. `VecSrc` holds the collection resident with an in-memory
rtree over envelopes. A GeoPackage carries its own rtree in SQLite.

- [x] `GpkgSrc` over rusqlite, opening the file read-only and reading a
      tile as one query against the layer's `rtree_<table>_<geom>` virtual
      table, decoding only those rows. The WKB and GeoPackage-binary
      decoder that geodukt-io carried moved to topoi-core as `wkb`, and
      geodukt-io calls it, so the two cannot drift. `ptolemy-geopackage`
      was not used: its `DataStore` is async over ptolemy-core's geo-types
      features, which would need a geometry conversion and pull ptolemy-core
      in for one query. Resident memory is a pool of open connections.
- [x] Simplification per ladder level is unchanged, it already happens per
      tile read.
- [x] A fixture of a few thousand features and an assertion on the row
      count read per tile, plus a whole-window pull compared fragment for
      fragment against `VecSrc` over the same features.

## 5. Small items

- [ ] STAC assets within an item read in parallel, one day. The per-asset
      loop in `StacSrc` reads sequentially. Join the asset futures the same
      way item reads already fan out, under the same 64-in-flight cap.
- [ ] Disk cache across processes, one to two days. The spill store is a
      fresh directory per engine because nothing identifies the graph. Add
      an element identity hash: each element contributes a stable
      description (`CogSrc` the file path and its IFD byte length, `StacSrc`
      the api, collection, assets and composite, each transform its
      parameters) and the engine hashes those with the fixated caps. The
      store directory is named by the hash, a matching directory on open is
      reused, a mismatch is a new directory. Element identity goes on the
      `Source`, `Transform` and `Fanin` traits as a required method so a new
      element cannot forget it.

## 6. Memory domains and the GPU tensor path

Two to three weeks before any inference. The solver is kind-blind, so the
caps change is small. The cost is in the chunk type, the cache and the
runtime.

- [ ] `TensorPattern` gains a `domain: SetField<Domain>` with `System` and
      `Wgpu`, defaulting to `OneOf([System])`. Raster, point and vector
      patterns stay system-only. This is glass2glass's memory-domain idea,
      narrowed to the one kind that has a GPU consumer.
- [ ] `TensorChunk::data` becomes an enum: `System(Vec<f32>)` or
      `Wgpu(wgpu::Buffer)` with its device. `Chunk` stays an enum over kinds.
- [ ] Two auto-plugged adapters, `Upload` and `Download`, registered through
      `Graph::register_adapter` the way the reproject adapter is, so a
      `System` producer wired into a `Wgpu` consumer gets an upload inserted
      by the solver and a mismatch never reaches `compute`.
- [ ] A `WgpuConvolve` transform: the 3x3 kernel as a compute shader over
      the chunk, halo handled by `plan` as today. `Convolve` on the CPU stays
      as the `System` variant of the same element.
- [ ] Cache rules: a `Wgpu` chunk counts against a separate GPU byte budget,
      never spills (the disk tier downloads and spills the system copy, or
      drops it), and eviction frees the buffer. `Entry::Ready` carries which
      budget it is in.
- [ ] Batching: a pull's chunks at one node are submitted as one command
      encoder when they share a device, which is where the GPU wins over
      per-chunk dispatch.
- [ ] `geoplumb` gains a `wgpu` feature. The server builds without it by
      default. A GPU test skips without an adapter, as glass2glass's do.

An inference element (ONNX through `ort`, following g2g-ml's element) is
another one to two weeks on top. `DESIGN.md` currently puts inference
outside this repo, so that is a decision before it is a task.

## 7. The long horizon: can GDAL go to nothing

The question splits by who brings the file.

For the stack's own paths, yes, in two to three years at one or two
engineers, and section 1 is the mechanism rather than a detour. The
feature-gated `gdal` source counts which drivers it opens in production. Each
driver with hits gets a pure-Rust reader, by hit count. When the counter has
been zero for a year the feature is deleted. What that porting list looks
like today, by what the org already touches:

- COG tile compressions beyond deflate: LZW (`weezl`, pure Rust), JPEG and
  WebP tiles (`image` crates), ZSTD (`ruzstd` for decode, pure Rust). LERC
  has a C binding crate and no pure-Rust decoder that I know of. Each is one
  to two weeks in terrano including predictor and band interleave cases.
- BigTIFF and striped GeoTIFF in `CogReader`, one week.
- NetCDF-4 and HDF5, the largest raster gap. HDF5 is a C library with no
  complete pure-Rust reader. Zarr is the way around it: `zarrs` is pure Rust
  and the EO catalogs that matter publish Zarr beside NetCDF. One week for a
  `ZarrSrc`, and NetCDF-4 stays on GDAL or is converted at ingest.
- GRIB2 complex and JPEG 2000 packing. JPEG 2000 has no pure-Rust decoder
  that I know of, so GRIB2 JPEG 2000 messages stay on a C codec or GDAL.
- FileGDB, the one verne needs. OpenFileGDB is a documented
  reverse-engineered format and GDAL's driver is the reference. A Rust reader
  for tables, geometry and the catalogue XML is two to four months, and it
  removes the last `gdal-sys` in the org.
- Vector long tail: FlatGeobuf (`flatgeobuf` crate exists), GeoParquet
  (`ptolemy-api` already depends on `parquet`, read what it does before
  adding another path), KML (verne has it), GML and WFS (XML, two weeks),
  DXF (`dxf` crate), MapInfo TAB (no crate, a month).

What the other repos already hold, checked 2026-10-28 over every
`Cargo.toml` and README in the org: GDAL only in verne. ONNX inference in
panoptes, which is where the design doc sends inference. `wgpu` and Vello
in fenestra and jung for rendering, no compute path. `object_store` in
tiletopia behind its `martin` feature only. tiletopia consumes terrano's
`CogReader` for its `/api/v1/cog` window route and geoplumb's STAC source
for its analysis tiles, so neither is duplicated there. No Zarr, HDF5,
LZW or LERC anywhere.
- Coordinate transforms: projicio already covers EPSG codes, Helmert and
  NTv2. What PROJ still has is geoid and vertical grids, time-dependent
  transforms and the CDN grid fetch. Each is weeks, none is blocking.
- The QGIS `gdal` provider in `geolang`'s `run_qgis_algorithm`. This is not a
  format, it is every gdal processing algorithm a user can name. It goes
  away only when the geodukt and geoplumb operator sets cover what users
  ask for, which `viewtopia/DESIGN_TODO.md` already says to grow by demand.

For a public user who brings any file GDAL opens, no. MrSID and ECW are
proprietary SDKs and never pure Rust by license. JPEG 2000 through Kakadu
is the same. The remaining hundred-plus GDAL drivers have no users in this
stack, and matching GDAL's behaviour where it is under-specified (nodata
and mask semantics, color tables, PAM sidecars, metadata domains, warp
kernel edge handling) is open-ended work that only differential testing
against GDAL would bound. The honest end state is the one the owner
decision already names: pure Rust for every format that has a user, GDAL
behind a feature for the rest, and the feature off in every default build.

## Order

1. Section 1, the `gdal` source: the coverage, and the hit counter that
   drives section 7.
2. Section 2, `RangeRead` by shared reference: a small change that removes
   the one serialization point on the hot path.
3. Sections 3 and 4, windowed point and vector sources: they stop two
   sources from holding whole files resident.
4. Section 5, the small items.
5. Section 6, GPU, when a workload needs it. Nothing above depends on it.

Everything except inference is roughly two months, with section 6 taking
more than half of that.
