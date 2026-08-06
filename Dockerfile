FROM rust:bookworm AS builder

WORKDIR /app
COPY . .
RUN cargo build --release -p geoplumb-server

FROM debian:bookworm-slim

# ca-certificates for https stac searches and cog range requests, curl for
# the healthcheck. no PROJ: projicio is pure rust (proj4rs plus embedded
# crs definitions), so there is no native library or grid share to stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false geoplumb

COPY --from=builder /app/target/release/geoplumb-server /usr/local/bin/geoplumb-server

USER geoplumb

ENV PORT=3000

EXPOSE 3000

# GEOPLUMB_LAYERS is deliberately unset: the server refuses to start
# without a layer file, mount one and name it
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["geoplumb-server"]
