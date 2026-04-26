# Keryx Miner

A high-performance miner for **Keryx**, optimized for **GPU** using **CUDA** and **OpenCL**.

---

## Installation

### Precompiled Binaries

Download the version corresponding to your system from the repository's **Releases** page.

---

### Build from Source

#### Requirements

- Install **Rust** and **Cargo**: https://rustup.rs/

#### Steps

```bash
git clone https://github.com/Keryx-Labs/keryx-miner.git
cd keryx-miner
cargo build --release --all
```

The compiled binaries will be available in: target/release/

## Usage

To start mining with default settings on all detected devices:

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

## Main Options

To view all available options:

```bash
./keryx-miner --help
```

## Connect with us

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

> "Intelligence is the message. Keryx is the messenger."

## Dev Fund

By default, 2% of mining rewards are allocated to support development.

You can modify this percentage with:

```bash
--devfund-percent XX.YY
```

## Donations

Support the original developers:

Elichai 
```bash
kaspa:qzvqtx5gkvl3tc54up6r8pk5mhuft9rtr0lvn624w9mtv4eqm9rvc9zfdmmpu
```

HauntedCook 
```bash
kaspa:qz4jdyu04hv4hpyy00pl6trzw4gllnhnwy62xattejv2vaj5r0p5quvns058f
```