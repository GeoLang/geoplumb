# TODO

- geodukt executor swap onto the engine. geoplumb serves the window-local head of a pipeline (sources, reproject, filter, schema, clip, rasterize), the executor pulls full-extent, dissolves at the boundary, then runs the whole-feature ops on geodukt's existing topoi-based transforms: buffer, centroid, group-by dissolve, `$area` expressions, fixed-tolerance simplify
