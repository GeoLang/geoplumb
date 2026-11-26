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
- geoplumb: `GdalSrc` over the `gdal` crate, behind the `gdal` feature and
  off in the default build (section 1). Its other layer sources, `stac`,
  `cog`, `geojson`, `gpkg`, `las` and `copc`, use no GDAL.

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
- [x] COPC. `CopcSrc` reads the octree hierarchy at open and fetches only
      the nodes a window at a ladder level touches, through `copc-reader`
      0.9 over a local file or geoplumb's `HttpRange`, so a remote COPC over
      HTTP works. The COPC level maps onto the ladder level directly, which
      replaces the per-level voxel thinning for this source. copc-reader
      merges chunk fetches less than 64 KiB apart, so a tile of a small
      file reads most of it. The crs comes from the layer, nothing in the
      org parses the WKT CRS VLR.

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

- [x] STAC assets within an item read in parallel, each asset read under
      the same 64-in-flight cap item reads used, bands joined in asset order.
- [x] Disk cache across processes, one to two days. The spill store is a
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

The solver is kind-blind, so the caps change is small. The cost is in the
chunk type, the cache and the runtime. A 3x3 convolve alone does not pay
for the upload and download around it, the path pays when a tensor stays
on the device for something heavier, so the convolve here is the proof
that the solver splices uploads correctly, not a speedup. Decisions:
inference stays a separate decision (DESIGN.md puts it outside this repo,
panoptes holds the ONNX code); one `wgpu` device per graph, handed in by
the driver through `Graph::with_gpu`, never a process-wide device; the
ubuntu CI leg installs lavapipe so the shader path runs on every push;
two slices, batching alone in the second, recorded per `GpuContext` with no
engine change.

Slice 1, the domain:

- [x] `TensorPattern` gains a `domain: SetField<Domain>` with `System` and
      `Wgpu`, defaulting to `OneOf([System])`. Raster, point and vector
      patterns stay system-only. This is glass2glass's memory-domain idea,
      narrowed to the one kind that has a GPU consumer.
- [x] `TensorChunk::data` becomes an enum: `System(Vec<f32>)` or
      `Wgpu(wgpu::Buffer)` with its context. `Chunk` stays an enum over kinds.
- [x] Two adapters, `Upload` and `Download`, registered by
      `Graph::with_gpu` the way the reproject adapter is, so a `System`
      producer wired into a `Wgpu` consumer gets an upload inserted by the
      solver and a mismatch never reaches `compute`.
- [x] A `WgpuConvolve` transform: the 3x3 kernel as a compute shader over
      the chunk, halo handled by `plan` as today. `TensorConv` on the CPU
      stays as the `System` element.
- [x] Cache rules: a `Wgpu` chunk counts against the context's own GPU byte
      budget, never spills, and is dropped on eviction. `Entry::Ready`
      carries which budget it is in.
- [x] `geoplumb` gains a `wgpu` feature. The server builds without it. The
      gpu tests skip without an adapter unless `GEOPLUMB_REQUIRE_GPU` is
      set, which the ubuntu CI leg sets after installing lavapipe.

Slice 2, batching:

- [x] Dispatches are recorded into one open command encoder per
      `GpuContext` and submitted only when a download needs the results or
      sixty-four dispatches are pending, so a pull's chunks share a
      submission whichever node they belong to. The engine stays out of
      it, which is where the GPU wins over per-chunk submit.

An inference element (ONNX through `ort`, following g2g-ml's element) is
another one to two weeks on top. `DESIGN.md` currently puts inference
outside this repo, so that is a decision before it is a task.

## 7. The long horizon: can GDAL go to nothing

The question splits by who brings the file.

For the stack's own paths, yes, in two to three years at one or two
engineers, and section 1 is the mechanism rather than a detour.
`GdalSrc::driver()` returns the short name of the driver that opened a file,
and `geoplumb-server` prints it at startup for each `gdal` layer. No counter
of opens per driver exists. The plan needs one in production: each driver
with opens gets a pure-Rust reader, most opens first, and when the counter
has been zero for a year the feature is deleted. What that porting list looks
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

## 8. Output, correctness and algorithm gaps

Sections 1 to 6 are about reading. These are about what comes out, checked
2026-11-26 against every repo in the org. Where another repo already does the
job, geoplumb takes it over and that repo calls geoplumb, so there is one
code path (owner decision 2026-11-26).

Waves run in parallel worktrees under `/home/aaron/src/GeoLang/.worktrees/`,
each with its own `CARGO_TARGET_DIR`. Agents write code and run `cargo fmt`
only. A scripted gate builds and tests each settled tree, and a box is ticked
only after a proof script passes on real input, recorded in
`/home/aaron/tmp/geolang-fix/proof-ledger.tsv`.

### Wave 1

- [x] 8.0 terrano-core lock bump, 357778f to 62842f8. The locked `CogReader`
      reads a GDAL-written overview's pixel width as a denormal (about
      4e-321 on ESA WorldCover), so `select_level` never picks an overview
      and a z6 tile reads the full-resolution level: geoplumb-server went
      past 4 GB and was killed. terrano master reads the widths right and
      tiletopia already locks it. The bump also brings `write_cog_tiles`
      for 8.4. Lands alone, before the wave 1 merges, together with the
      `needless_range_loop` fix in `TensorConv::compute` that Rust 1.100
      clippy (stable since 2026-11-09) flags, which fails master's next CI
      run otherwise. Both sit in the `geoplumb-baseline` worktree.
- [x] 8.1 Categorical rasters. Every resampling path is bilinear or block
      average, so land cover, `reclassify` and `rasterize` output blend class
      codes at coarse levels and across a reprojection. Raster caps gain a
      `value_kind` field, `Continuous` or `Categorical`, negotiated like dtype
      and covered by a new `FieldMask` bit. Sources declare it, `cog`, `gdal`
      and `stac` taking `value_kind = "categorical"` in a layer file.
      `reclassify` and `rasterize` pin `Categorical` on their output, the
      terrain, focal, bandmath, unary, convolve and idw elements pin
      `Continuous`, everything else passes it through. Categorical data
      decimates by mode (ties to the smaller code) and warps and resamples by
      nearest neighbour, in `RasterSrc`, `cog::decimate` (`CogSrc`,
      `GdalSrc`), `warp_to_grid` (`Reproject`, cross-crs `StacSrc` items),
      `resample_to_grid`, `Mosaic` and `Combine`. Continuous output stays
      byte-identical. Proof: an ESA WorldCover COG served categorical, every
      tile value at z6 to z12 and in EPSG:4326 and 3857 a WorldCover class
      code or nodata.
- [x] 8.2 One MVT encoder. `fenestra-core/src/mvt.rs` holds the org's only
      MVT encoder. Its format-only part (`MvtFeature`, `MvtGeometry`,
      `MvtValue`, `MvtLayer`, `encode_tile`) moves to `jung-mvt` beside the
      decoder, fenestra calls it there, and the GeoJSON conversions stay in
      fenestra. Proof: fenestra's tile route output byte-identical before and
      after, and a jung-mvt round trip test.
- [x] 8.3 NetCDF and GRIB2 sources. terrano reads NetCDF classic and 64-bit
      offset and simple-packed GRIB2, and no repo calls either reader. A
      `NetcdfSrc` (one variable, georeferenced from its lat/lon or x/y
      coordinate variables, a time dimension resolved per pull time) and a
      `GribSrc` (one message, regular lat/lon grid template), both layer
      file source kinds. Proof: a public NOAA or Copernicus file served as
      tiles whose values match the reader's own decode at sampled pixels.
      terrano's readers fail on real files (GRIB2 section 3 read three
      octets off, sign-magnitude integers read as two's complement, no
      bitmap or multi-field messages; NetCDF record variables and 1-D
      coordinate variables unreadable, NC_BYTE unsigned), so they are fixed
      in terrano (worktree `terrano-grib-netcdf`) and geoplumb holds no
      format parsing. Real data: NOAA OISST v2.1 NetCDF classic from
      CoastWatch ERDDAP, JMA GSM GRIB2 (simple packing, 103 fields).

- [ ] 8.1b Value kind per band. `value_kind` is a link field, so a STAC
      source naming `["red", "scl"]` resamples its SCL quality codes like
      reflectance. Below native zoom the codes blend (4.5) and `QualityMask`'s
      exact match drops the pixel, which the `s2-red-clear` example layer
      hits. Needs a kind per band in raster caps and chunks, not per link.

### Wave 2

- [ ] 8.4 Export. tiletopia's `/api/v1/analysis/export/{op}` pulls three
      fixed ops whole into memory as a web mercator COG under a 4096x4096
      cap. geoplumb gets `export::write_cog`: a grid in the node's own crs,
      streamed through terrano's `write_cog_tiles` one tile at a time, the
      overviews pulled from the engine at coarser resolutions rather than
      averaged, resampling per `value_kind`. `geoplumb-server` gets
      `GET /export/{layer}?bbox&crs&resolution&t`, bbox in the target crs as
      gdalwarp's `-te`, building an engine for the layer ending in that crs.
      tiletopia's export route then calls `export::write_cog` and its own
      grid and encode code is deleted. Proof: an export in EPSG:32632 passes
      GDAL's `validate_cloud_optimized_geotiff.py --full-check`, and its
      pixels match the same window pulled from the engine.
- [x] 8.5 Vector layers. A layer whose chain ends in vector features is
      served as `/tiles/{layer}/{z}/{x}/{y}.mvt` through jung-mvt, with the
      `rasterize` requirement on `geojson` and `gpkg` dropped for it.
      `Reproject` and `VecReproject` are one element over a new
      `Constraint::AnyOf`, and vector features from a geographic crs are
      clipped to web mercator's latitude band before projecting. Arrays and
      objects go out as JSON text, nulls are left out. Proof: `ogrinfo` reads
      served Natural Earth tiles at z0, z2 and z5, every required country is
      present, a `vec_filter` lets only its continent through, `mvt_id` is the
      source index and every polygon winds exterior positive, holes negative.
- [ ] 8.6 Raster to vector elements. `Contours` and `Polygonize` over
      terrano's kernels as cross-kind transforms, seam-consistent across
      tiles, servable through 8.5. `Viewshed` as a transform whose plan is
      the observer's radius, not the output window. terrano's `contours`
      builds levels by repeated addition from the window minimum and places
      crossings as `col + t`, so adjacent tiles round differently, and its
      saddle choice looks inverted. Fixed in terrano a4cc373 (levels from
      `base + k * interval`, vertices as cell plus fraction, saddles paired
      as `gdal_contour` pairs them) rather than padded around in geoplumb,
      since tiletopia calls it too.

### Wave 3

- [ ] 8.7 Global operations. Flow accumulation, watershed and stream order
      depend on a whole upstream basin, so no fixed halo works. tiletopia
      runs them over one bounded window in memory. Design before building:
      a transform over a declared region computed once per ladder level and
      pull time, under a pixel cap. Then tiletopia's
      `/api/v1/analysis/terrain` and `/viewshed` call geoplumb and its
      copies go.
- [ ] 8.8 Zarr source over `zarrs`, after checking the crate is still the
      current choice.
- [ ] 8.9 Native dtype. Every source serves f64. terrano's `Raster` is f64
      only, so this starts as a design: where a typed sample buffer lives
      (terrano or a geoplumb chunk), and what compute does with it.
- [ ] 8.10 Colour ramps on png tiles, dogfooding `jung-core`'s `ColorRamp`,
      and CRS input beyond EPSG codes in projicio.

## Order

1. Section 1, the `gdal` source: the coverage, and the driver name that
   section 7's counter needs.
2. Section 2, `RangeRead` by shared reference: a small change that removes
   the one serialization point on the hot path.
3. Sections 3 and 4, windowed point and vector sources: they stop two
   sources from holding whole files resident.
4. Section 5, the small items.
5. Section 6, GPU, when a workload needs it. Nothing above depends on it.

Every box in sections 1 to 6 is ticked. What remains is `OgrSrc` (section 1),
the inference decision (section 6) and section 7.
