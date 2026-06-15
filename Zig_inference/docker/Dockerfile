FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git curl libssl-dev pkg-config libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://ziglang.org/download/0.16.0/zig-linux-x86_64-0.16.0.tar.xz -o zig.tar.xz && \
    tar -xf zig.tar.xz && mv zig-linux-x86_64-0.16.0 /usr/local/zig && \
    ln -s /usr/local/zig/zig /usr/local/bin/zig && rm zig.tar.xz

COPY . /app
WORKDIR /app
RUN zig build --release=true
EXPOSE 8080
CMD ["./zig-inference", "--model", "/models/model.gguf", "--shards", "1", "--port", "8080"]
