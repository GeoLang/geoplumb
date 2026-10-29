FROM rust:bookworm AS builder

# GDAL=1 builds the gdal layer source against bookworm's libgdal
ARG GDAL=0

WORKDIR /app
RUN case "$GDAL" in 0|1) ;; *) echo "GDAL must be 0 or 1, got $GDAL" >&2; exit 1 ;; esac \
    && if [ "$GDAL" = 1 ]; then \
        apt-get update && apt-get install -y --no-install-recommends libgdal-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi
COPY . .
RUN if [ "$GDAL" = 1 ]; then \
        cargo build --release -p geoplumb-server --features gdal; \
    else \
        cargo build --release -p geoplumb-server; \
    fi

FROM debian:bookworm-slim

ARG GDAL=0

# ca-certificates for https stac searches and cog range requests, curl for
# the healthcheck. no PROJ: projicio is pure rust (proj4rs plus embedded
# crs definitions), so there is no native library or grid share to stage.
# GDAL=1 adds libgdal32, the runtime half of the builder's libgdal-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl $(if [ "$GDAL" = 1 ]; then echo libgdal32; fi) \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false geoplumb \
    && mkdir -p /cache && chown geoplumb:geoplumb /cache

COPY --from=builder /app/target/release/geoplumb-server /usr/local/bin/geoplumb-server

USER geoplumb

ENV PORT=3000

EXPOSE 3000

# GEOPLUMB_LAYERS is deliberately unset: the server refuses to start
# without a layer file, mount one and name it
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["geoplumb-server"]
